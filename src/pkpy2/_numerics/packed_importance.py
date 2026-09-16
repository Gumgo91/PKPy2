"""Experimental fixed-proposal randomized QMC marginal likelihood.

This is importance integration, not SAEM or a Laplace approximation. Prepare
proposals outside optimization, then keep samples and proposal densities fixed
throughout each optimization stage. Independent scrambles audit integration
error; ESS alone cannot certify accuracy or coverage of posterior modes.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.special import ndtri, logsumexp
from scipy.stats import qmc
from .packed_kernels import joint_nll
from .subject_work import ordered_map


@njit(cache=True, nogil=True)
def joint_batch(samples, code, typical, indices, omega, times, observed,
                dt, amounts, sp, sa):
    values = np.empty(len(samples))
    for i in range(len(samples)):
        values[i] = joint_nll(samples[i], code, typical, indices, omega,
                              times, observed, dt, amounts, sp, sa)
    return values


def normal_logpdf(samples, center, covariance):
    chol = np.linalg.cholesky(covariance)
    z = np.linalg.solve(chol, (samples-center).T)
    return -.5*(len(center)*np.log(2*np.pi) + 2*np.log(np.diag(chol)).sum()
                + (z*z).sum(axis=0))


@dataclass(frozen=True)
class FixedImportance:
    samples: np.ndarray
    log_proposal: np.ndarray
    seed: int
    components: int = 1
    centers: tuple = ()
    covariances: tuple = ()

    @classmethod
    def prepare(cls, omega, *, power=14, seed=0, center=None, covariance=None):
        """Equal mixture of reference ETA prior and an optional pilot Gaussian.

        Stratified components have equal counts; the full mixture density is
        used in every weight. The prior component guards against an overly
        narrow pilot. No optimization or mode selection occurs in this method.
        """
        omega = np.asarray(omega, dtype=float)
        if omega.ndim != 1 or not np.all(np.isfinite(omega)) or np.any(omega <= 0):
            raise ValueError('positive finite diagonal ETA variances required')
        if type(power) is not int or not 2 <= power <= 24:
            raise ValueError('sample power must be an integer in [2, 24]')
        d = len(omega)
        if not d:
            return cls(np.zeros((1, 0)), np.zeros(1), seed)
        centers = [np.zeros(d)]
        covariances = [np.diag(omega)]
        if center is not None or covariance is not None:
            center, covariance = np.asarray(center), np.asarray(covariance)
            if (center.shape != (d,) or covariance.shape != (d,d)
                    or not np.all(np.isfinite(center))
                    or not np.all(np.isfinite(covariance))
                    or not np.allclose(covariance, covariance.T)):
                raise ValueError('invalid pilot Gaussian')
            centers.append(center)
            covariances.append(covariance)
        parts = []
        streams = np.random.SeedSequence(seed).spawn(len(centers))
        for c, v, stream in zip(centers, covariances, streams):
            u = qmc.Sobol(d, scramble=True, seed=np.random.default_rng(stream)).random_base2(
                power - (len(centers)==2))
            z = ndtri(np.clip(u, np.finfo(float).eps, 1-np.finfo(float).eps))
            parts.append(c + z @ np.linalg.cholesky(v).T)
        samples = np.ascontiguousarray(np.concatenate(parts))
        logq = logsumexp(np.array([normal_logpdf(samples,c,v)
                                  for c,v in zip(centers,covariances)]),axis=0)-np.log(len(parts))
        samples.setflags(write=False)
        logq.setflags(write=False)
        return cls(samples, logq, seed,len(centers),tuple(centers),tuple(covariances))

    def evaluate(self, *args):
        return self.summarize(-joint_batch(self.samples, *args))

    def summarize(self, log_target):
        log_target = np.asarray(log_target, dtype=float)
        if log_target.shape != self.log_proposal.shape or np.any(np.isnan(log_target)):
            raise ValueError('invalid target density values')
        lw = log_target-self.log_proposal
        total = logsumexp(lw)
        if not np.isfinite(total):
            raise ValueError('nonfinite importance integral')
        weights = np.exp(lw-total)
        mean = weights @ self.samples
        centered = self.samples-mean
        covariance = (centered.T*weights) @ centered
        return dict(ofv=float(-2*(total-np.log(len(lw)))),
                    ess=float(1/np.dot(weights,weights)),
                    max_weight=float(weights.max()), samples=len(lw),
                    posterior_mean=mean, posterior_covariance=covariance,
                    approximation='randomized_qmc_importance', seed=self.seed)


def replica_report(results):
    """Empirical independent-scramble uncertainty, not an accuracy guarantee."""
    if len(results) < 2:
        raise ValueError('at least two independent scrambles required')
    if len({r['seed'] for r in results}) != len(results):
        raise ValueError('independent scramble seeds required')
    log_integrals = -.5*np.array([r['ofv'] for r in results])
    scaled = np.exp(log_integrals-log_integrals.max())
    return dict(ofv=float(-2*(logsumexp(log_integrals)-np.log(len(results)))),
                empirical_ofv_se=float(2*np.std(scaled,ddof=1)/np.sqrt(len(scaled))/scaled.mean()),
                ofv_range=float(np.ptp(-2*log_integrals)),
                minimum_ess=float(min(r['ess'] for r in results)),
                maximum_weight=float(max(r['max_weight'] for r in results)))


class StudyImportance:
    """Fixed per-subject proposals for an entire packed study.

    Typical values are supplied by the caller, so covariate transformations and
    clinical constraints are independent of integration. Population priors must
    be added once by the caller. Preparation never occurs inside evaluate().
    This experimental evaluator is deliberately not an automatic fit fallback.
    """
    def __init__(self, study, eta_indices):
        from .packed_solver import PackedEvaluator
        # Reuse the established input/model contract, without individual solves.
        with PackedEvaluator(study, eta_indices) as boundary:
            self.study, self.code, self.indices = study, boundary.code, boundary.indices
        self.proposals = None

    def _arguments(self, typical, omega, sp, sa):
        p=np.asarray(typical,dtype=float)
        om=np.asarray(omega,dtype=float)
        columns=2 if self.code==0 else 6 if self.code==3 else 4
        if p.shape!=(len(self.study.subject_ids),columns) or not np.isfinite(p).all():
            raise ValueError('invalid typical parameter matrix')
        if np.any((p[:,:-1] if self.code in (1,3) else p)<=0) or (self.code in (1,3) and np.any(p[:,-1]<0)):
            raise ValueError('invalid structural parameters')
        if om.shape!=(len(self.indices),) or not np.isfinite(om).all() or np.any(om<=0):
            raise ValueError('positive ETA variances required')
        if not np.isfinite([sp,sa]).all() or min(sp,sa)<0 or sp+sa<=0:
            raise ValueError('invalid residual standard deviations')
        s=self.study
        for i in range(len(p)):
            lo,hi=s.observation_offsets[i:i+2];dl,dh=s.dose_offsets[i:i+2]
            yield (self.code,p[i],self.indices,om,s.time[lo:hi],s.observation[lo:hi],
                   s.dose_time[dl:dh],s.dose_amount[dl:dh],float(sp),float(sa))

    def prepare(self, typical, omega, sp, sa, *, power=14, seed=0, pilot_power=14,sample_powers=None,posterior_pilot=None,work=None):
        powers=[power]*len(self.study.subject_ids) if sample_powers is None else list(sample_powers)
        if len(powers)!=len(self.study.subject_ids) or any(not isinstance(p,(int,np.integer)) or not 2<=p<=24 for p in powers):
            raise ValueError('one valid sampling power per subject required')
        if posterior_pilot is not None and len(posterior_pilot)!=len(powers):
            raise ValueError('one physical posterior pilot per subject required')
        def prepare_subject(item):
            i,args=item
            individual_power=int(powers[i])
            pilot_options={}
            if posterior_pilot is not None:
                mean,covariance=posterior_pilot[i]
                pilot_options=dict(center=np.asarray(mean)-np.log(args[1][self.indices]),
                                   covariance=2*np.asarray(covariance)+.01*np.diag(omega))
            # Transport physical posterior moments, then independently reweight
            # a defensive mixture at the new target before drawing final samples.
            pilot=FixedImportance.prepare(omega,power=max(pilot_power,individual_power),seed=2*(seed+i),**pilot_options)
            result=pilot.evaluate(*args)
            covariance=2*result['posterior_covariance']+.01*np.diag(omega)
            center=result['posterior_mean']
            # A prior-only pilot with very few effective particles does not
            # locate a concentrated conditional distribution reliably. Refine
            # its location and curvature before constructing the defensive
            # proposal; the target likelihood and final audits are unchanged.
            if result['ess'] < 50 or result['max_weight'] > .1:
                from .packed_derivatives import refine_mode
                best=None
                for initial in (center,np.zeros(len(omega))):
                    try:
                        mode,hessian,status,_,__,___=refine_mode(initial,*args)
                        if status!=0 or np.linalg.eigvalsh(hessian).min()<=0:continue
                        value=float(joint_nll(mode,*args))
                        if np.isfinite(value) and (best is None or value<best[0]):
                            best=(value,mode,hessian)
                    except (ValueError,RuntimeError,ArithmeticError):
                        continue
                if best is not None:
                    center=best[1]
                    covariance=2*np.linalg.solve(best[2],np.eye(len(omega)))+.01*np.diag(omega)
            return FixedImportance.prepare(omega,power=individual_power,seed=2*(seed+i)+1,
                center=center,covariance=covariance)
        self.proposals=tuple(ordered_map(work,prepare_subject,
            enumerate(self._arguments(typical,omega,sp,sa))))
        return self

    def evaluate(self, typical, omega, sp, sa):
        if self.proposals is None:
            raise RuntimeError('prepare fixed proposals before evaluation')
        rows=[p.evaluate(*a) for p,a in zip(self.proposals,self._arguments(typical,omega,sp,sa))]
        return float(sum(r['ofv'] for r in rows)),rows

    def prepare_audited(self, typical, omega, sp, sa, *, seed=0,
                        powers=(12,14,16,18), pilot_power=16,
                        subject_ofv_se=.005, minimum_ess=100, maximum_weight=.01):
        """Allocate integration samples by measured error, not subject identity.

        Four independent pilot/sample streams are compared at each budget.
        Exhaustion preserves the estimate with `quality_passed=False`. This
        audit applies only at the supplied point, never certifies an entire fit.
        After optimization, audit again with new seeds at the candidate point.
        """
        if (not powers or any(type(p) is not int or not 2<=p<=24 for p in powers)
                or any(b<=a for a,b in zip(powers,powers[1:]))):
            raise ValueError('increasing sample powers required')
        if (not np.isfinite([subject_ofv_se,minimum_ess,maximum_weight]).all()
                or subject_ofv_se<=0 or minimum_ess<=0 or not 0<maximum_weight<=1):
            raise ValueError('positive finite quality tolerances required')
        proposals=[];reports=[]
        for i,args in enumerate(self._arguments(typical,omega,sp,sa)):
            pilots=[]
            for replica in range(4):
                stream_seed=int(np.random.SeedSequence([seed,i,replica,0]).generate_state(1)[0])
                pilot=FixedImportance.prepare(omega,power=pilot_power,seed=stream_seed).evaluate(*args)
                pilots.append((pilot['posterior_mean'],2*pilot['posterior_covariance']+.01*np.diag(omega)))
            history=[]
            for power in powers:
                bank=[];rows=[]
                for replica,(mean,covariance) in enumerate(pilots):
                    stream_seed=int(np.random.SeedSequence([seed,i,replica,1]).generate_state(1)[0])
                    proposal=FixedImportance.prepare(omega,power=power,seed=stream_seed,
                        center=mean,covariance=covariance)
                    bank.append(proposal);rows.append(proposal.evaluate(*args))
                report=replica_report(rows)
                # With no ETAs there is no sampling uncertainty.
                passed=(report['empirical_ofv_se']<=subject_ofv_se and
                        (len(self.indices)==0 or (report['minimum_ess']>=minimum_ess
                          and report['maximum_weight']<=maximum_weight)))
                history.append(dict(power=power,quality_passed=passed,**report))
                if passed:break
            # Retain a predetermined stream, never select the best likelihood.
            proposals.append(bank[0])
            reports.append(dict(subject_id=self.study.subject_ids[i],history=history,
                                quality_passed=passed))
        self.proposals=tuple(proposals)
        return dict(quality_passed=all(r['quality_passed'] for r in reports),
                    scope='supplied_population_point_only',subjects=reports)
