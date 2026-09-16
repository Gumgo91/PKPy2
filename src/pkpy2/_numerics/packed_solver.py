"""Packed conditional-mode evaluator with bounded persistent workers.

One instance owns one executor; use as a context manager. Structural/covariate
typical values arrive as a matrix. Population penalties are added by the caller
once, never in each subject's contribution. The configured determinant uses
observed Hessian Laplace or expected-information curvature; these are distinct
approximations and are recorded in fit provenance.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from .packed_bfgs import solve_bfgs
from .packed_derivatives import refine_mode
from .packed_kernels import joint_nll, joint_value_gradient, joint_hessian
from .blas_runtime import BlasLease


class IndividualModeError(RuntimeError):
    """A numerical candidate failure, with all attempted mode diagnostics."""
    def __init__(self, failures):
        self.failures = failures
        super().__init__(f'no converged individual mode in deterministic search: {failures}')

    def __reduce__(self):
        return (type(self), (self.failures,))


@dataclass(frozen=True)
class InnerSettings:
    max_iter: int = 80
    ftol: float = 1e-6
    gtol: float = 1e-5
    fallback_xatol: float = 5e-3
    fallback_fatol: float = 5e-3
    method: str = 'L-BFGS-B'
    refine_modes: bool = False
    curvature: str = 'observed'
    mode_search: str = 'single_start'

    def __post_init__(self):
        if self.mode_search not in ('single_start','prior_axes'):
            raise ValueError('unsupported individual mode search')
        if self.mode_search=='prior_axes' and not self.refine_modes:
            raise ValueError('mode search requires precise refinement')
        if self.curvature not in ('observed','expected_information'):
            raise ValueError('unsupported curvature approximation')
        if self.curvature == 'expected_information' and not self.refine_modes:
            raise ValueError('expected information requires mode refinement')
        if self.method not in ('L-BFGS-B', 'compiled-BFGS'):
            raise ValueError('unsupported inner optimizer')
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError('positive integer iteration budget required')
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.ftol,self.gtol,self.fallback_xatol,self.fallback_fatol)):
            raise ValueError('positive finite tolerances required')


def _search_refined_mode(initial, args, settings):
    seeds=[initial]
    if settings.mode_search=='prior_axes':
        seeds.append(np.zeros(len(initial)))
        for j in range(len(initial)):
            for sign in (-1.,1.):
                seed=np.zeros(len(initial))
                seed[j]=sign*np.sqrt(args[3][j])
                seeds.append(seed)
    best=None
    accepted=0
    failures=[]
    primary_value=None
    for index,seed in enumerate(seeds):
        try:
            eta,hessian,status,nit,score,step=refine_mode(
                seed,*args,information_metric=settings.curvature=='expected_information')
            if status!=0:
                failures.append(dict(start_index=index,status=int(status),score_norm=float(score)))
                continue
            value=float(joint_nll(eta,*args))
            if not math.isfinite(value):continue
        except (ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
            failures.append(dict(start_index=index,error_type=type(error).__name__,message=str(error)))
            continue
        accepted+=1
        if index==0:primary_value=value
        # Select a posterior mode by joint NLL, never by its determinant
        # correction. Stable tie handling avoids arbitrary mode changes.
        if best is None or value<best[0]-1e-10:
            best=(value,eta,hessian,dict(status=0,iterations=int(nit),
                  score_norm=float(score),newton_step_norm=float(step),start_index=index))
    if best is None:
        raise IndividualModeError(failures)
    best[3].update(mode_search=settings.mode_search,starts=len(seeds),converged_starts=accepted,
                   primary_joint_nll=primary_value,
                   improvement_over_primary=(primary_value-best[0] if primary_value is not None else None),
                   unsuccessful_starts=failures)
    return best


def _solve(job):
    index, args, settings = job
    zero = np.zeros(len(args[2]))  # args: code, typical, eta_indices, ...
    if len(zero):
        recovery = None
        final_method = settings.method
        try:
            if settings.method == 'compiled-BFGS':
                x, value, gradient, status, nit, calls = solve_bfgs(
                    zero, *args, max_iter=settings.max_iter, gtol=settings.gtol)
                messages = ('gradient converged', 'iteration limit',
                            'line search failed', 'nonfinite initial value')
                fit = OptimizeResult(x=x, fun=value, jac=gradient,
                                     success=status == 0, status=status,
                                     nit=nit, nfev=calls, message=messages[status])
            else:
                fit = minimize(joint_value_gradient, zero, args=args, jac=True,
                               method='L-BFGS-B', options=dict(maxiter=settings.max_iter,
                               ftol=settings.ftol,gtol=settings.gtol))
            primary = dict(success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev))
            fallback = not fit.success or not np.isfinite(fit.fun)
        except (ValueError, OverflowError, FloatingPointError) as error:
            primary = dict(success=False,message=str(error),error_type=type(error).__name__,nfev=None)
            fallback = True
        if fallback and settings.method == 'compiled-BFGS':
            # Recover with the established gradient solver before the less
            # precise derivative-free retry. Preserve both attempts in audit.
            try:
                fit = minimize(joint_value_gradient, zero, args=args, jac=True,
                               method='L-BFGS-B', options=dict(maxiter=settings.max_iter,
                               ftol=settings.ftol, gtol=settings.gtol))
                recovery = dict(success=bool(fit.success), message=str(fit.message),
                                nfev=int(fit.nfev), method='L-BFGS-B')
                fallback = not fit.success or not np.isfinite(fit.fun)
                final_method = 'L-BFGS-B'
            except (ValueError, OverflowError, FloatingPointError) as error:
                recovery = dict(success=False, message=str(error),
                                error_type=type(error).__name__, method='L-BFGS-B')
        if fallback:
            final_method = 'Nelder-Mead'
            fit = minimize(joint_nll,zero,args=args,method='Nelder-Mead',
                           options=dict(maxiter=settings.max_iter,adaptive=True,
                           xatol=settings.fallback_xatol,fatol=settings.fallback_fatol))
        primary_valid=bool(fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all())
        if not primary_valid and settings.mode_search!='prior_axes':
            raise RuntimeError(f'packed subject {index}: inner optimization failed: {fit.message}')
        eta = fit.x.copy() if primary_valid else zero
        value = float(joint_nll(eta,*args))
        if primary_valid and not math.isclose(value,float(fit.fun),rel_tol=0.,abs_tol=1e-10):
            raise RuntimeError('packed inner objective recheck failed')
        refinement = None
        if settings.refine_modes:
            value,eta,hessian,refinement=_search_refined_mode(eta,args,settings)
        else:
            hessian=joint_hessian(eta,*args)
        if not np.isfinite(hessian).all():
            raise RuntimeError('nonfinite packed Hessian')
        hessian = .5*(hessian+hessian.T)
        minimum = float(np.linalg.eigvalsh(hessian).min())
        if settings.refine_modes and minimum <= 0:
            raise RuntimeError('nonpositive curvature at refined individual mode')
        ridge = -minimum+1e-6 if minimum<=0 else 0.
        hessian += ridge*np.eye(len(eta))
        if refinement is not None:
            refinement['minimum_eigenvalue']=minimum
        sign, logdet = np.linalg.slogdet(hessian)
        if sign <= 0 or not np.isfinite(logdet):
            raise RuntimeError('invalid packed Hessian determinant')
        diagnostics = dict(primary=primary,recovery=recovery,
                           curvature_approximation=settings.curvature,
                           mode_refinement=refinement,
                           used_fallback=fallback or recovery is not None,
                           final_success=True,nfev=int(fit.nfev),ridge=ridge,
                           final_method=final_method,
                           final_status=int(fit.status),final_message=str(fit.message),
                           final_nit=int(fit.nit),final_objective=value,
                           final_nfev=int(fit.nfev))
    else:
        eta=zero;value=float(joint_nll(eta,*args));logdet=0.
        hessian=np.empty((0,0));diagnostics=dict(final_success=True,used_fallback=False,ridge=0.)
    contribution=2.*value+logdet-len(eta)*math.log(2.*math.pi)
    if not math.isfinite(contribution):
        raise RuntimeError('nonfinite packed subject contribution')
    return dict(subject_index=index,eta=eta,hessian=hessian,nll=value,
                contribution=contribution,diagnostics=diagnostics)


def _solve_chunk(jobs):
    """Amortize executor scheduling while retaining deterministic subject order."""
    return [_solve(job) for job in jobs]


class PackedEvaluator:
    def __init__(self, study, eta_indices, *, workers=1, settings=None):
        codes={'1cmt_iv':0,'1cmt_oral':1,'2cmt_iv':2,'2cmt_oral':3}
        if study.model not in codes:
            raise ValueError('unsupported packed solver model')
        if type(workers) is not int or workers < 1:
            raise ValueError('positive integer worker budget required')
        indices=tuple(eta_indices)
        count=2 if study.model=='1cmt_iv' else 3 if study.model=='1cmt_oral' else 5 if study.model=='2cmt_oral' else 4
        if len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<count for i in indices):
            raise ValueError('unique valid ETA indices required; ALAG cannot have ETA')
        self.study=study;self.code=codes[study.model]
        self.indices=np.array(indices,dtype=np.int64)
        self.settings=settings or InnerSettings()
        self.workers=min(workers,len(study.subject_ids))
        self.blas_lease=BlasLease() if self.workers>1 else None
        try:
            self.pool=ThreadPoolExecutor(max_workers=self.workers) if self.workers>1 else None
        except BaseException:
            if self.blas_lease is not None:self.blas_lease.close()
            raise
        self.closed=False
        self.python_gradient_ready=False

    def __enter__(self): return self

    def __exit__(self,*exc): self.close()

    def close(self):
        try:
            if self.pool is not None: self.pool.shutdown(wait=True)
        finally:
            if self.blas_lease is not None:self.blas_lease.close()
            self.closed=True

    def evaluate(self,typical,omega,sigma_prop,sigma_add):
        if self.closed: raise RuntimeError('packed evaluator is closed')
        p=np.array(typical,dtype=float,order='C',copy=True)
        om=np.array(omega,dtype=float,copy=True)
        columns=2 if self.code==0 else 6 if self.code==3 else 4
        if p.shape!=(len(self.study.subject_ids),columns) or not np.isfinite(p).all():
            raise ValueError('invalid typical parameter matrix')
        if np.any((p[:,:-1] if self.code in (1,3) else p)<=0) or (self.code in (1,3) and np.any(p[:,-1]<0)):
            raise ValueError('invalid structural parameters')
        if om.shape!=(len(self.indices),) or not np.isfinite(om).all() or np.any(om<=0):
            raise ValueError('positive ETA variances required')
        if any(not math.isfinite(v) or v<0 for v in (sigma_prop,sigma_add)) or sigma_prop+sigma_add<=0:
            raise ValueError('invalid residual standard deviations')
        s=self.study;jobs=[]
        for i in range(len(p)):
            lo,hi=s.observation_offsets[i:i+2];dl,dh=s.dose_offsets[i:i+2]
            args=(self.code,p[i],self.indices,om,s.time[lo:hi],s.observation[lo:hi],
                  s.dose_time[dl:dh],s.dose_amount[dl:dh],sigma_prop,sigma_add)
            jobs.append((i,args,self.settings))
        if jobs and not self.python_gradient_ready:
            # Initialize the Python-facing compiled return adapter before a
            # nested compiled call or worker first specializes this signature.
            # SciPy recovery calls this entry point directly.
            joint_value_gradient(np.zeros(len(self.indices)),*jobs[0][1])
            self.python_gradient_ready=True
        if self.pool:
            futures=[]
            try:
                chunk_size=(len(jobs)+self.workers-1)//self.workers
                for start in range(0,len(jobs),chunk_size):
                    futures.append(self.pool.submit(_solve_chunk,jobs[start:start+chunk_size]))
                rows=[row for future in futures for row in future.result()]
            except BaseException:
                # A failed evaluation must not leave queued/running solves
                # consuming the worker budget of the next objective call.
                for future in futures:
                    future.cancel()
                wait(futures)
                raise
        else:
            rows=list(map(_solve,jobs))
        return math.fsum(r['contribution'] for r in rows),rows
