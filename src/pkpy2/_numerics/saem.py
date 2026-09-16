"""Experimental generalized SAEM for packed Gaussian mixed-effects models.

Persistent physical log-parameter chains; stochastic approximation of complete
data Q; bounded generalized M steps. Gaussian diagonal ETA distributions and
the existing heteroscedastic combined error are preserved. No Laplace Hessian
or individual MAP search is performed. This is not the Monolix implementation.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.special import ndtri
from .packed_kernels import joint_nll, event_predictions
from .packed_sensitivities import event_predictions_gradient_into
from .packed_importance import FixedImportance, replica_report, StudyImportance


def mstep_scale(x,decode,indices,bounds,observations):
    """Diagonal sensitivity preconditioner; does not alter Q or bounds.

    Mean/variance terms use complete-data Gaussian information. Residual and
    non-ETA terms supply approximate scales, not an exact Fisher information.
    """
    p,om,sp,sa=decode(x);n=len(p);indices=np.asarray(indices,dtype=int)
    other=[j for j in range(p.shape[1]) if j not in indices]
    scale=np.ones(len(x))
    for j in range(len(x)):
        h=1e-5*max(1.,abs(x[j]));a=x.copy();b=x.copy();a[j]+=h;b[j]-=h
        if bounds is not None:
            lo,hi=bounds[j]
            if lo is not None:b[j]=max(b[j],lo)
            if hi is not None:a[j]=min(a[j],hi)
        span=a[j]-b[j]
        if span==0:continue
        p1,o1,s1,a1=decode(a);p0,o0,s0,a0=decode(b)
        dm=(np.log(p1[:,indices])-np.log(p0[:,indices]))/span
        do=(o1-o0)/span
        information=2*np.sum(dm*dm/om)+n*np.sum((do/om)**2)
        information+=2*observations*(((s1-s0)/span/max(sp,1e-6))**2+
                                      ((a1-a0)/span/max(sa,1e-6))**2)
        if other:
            information+=observations*np.mean(((p1[:,other]-p0[:,other])/span/
                                               np.maximum(np.abs(p[:,other]),1e-6))**2)
        scale[j]=1/np.sqrt(max(information,1e-12))
    return np.clip(scale,1e-6,1e6)


@njit(cache=True,nogil=True)
def transition(phi, typical, omega, indices, code, oo, times, y, do, dt, amounts,
               sp, sa, normals, uniforms, scale):
    """One independence, one block and q coordinate MH proposals per chain.

    Independence proposal equals the current population prior: include its
    reverse/forward ratio. Random walks are symmetric. States are log physical
    parameters so they remain meaningful when population means change.
    """
    state=phi.copy(); n,c,q=state.shape
    accepted=np.zeros(3,dtype=np.int64)
    for i in range(n):
        lo,hi=oo[i:i+2];dl,dh=do[i:i+2]
        mu=np.log(typical[i,indices]);root=np.sqrt(omega)
        for k in range(c):
            eta=state[i,k]-mu
            value=joint_nll(eta,code,typical[i],indices,omega,times[lo:hi],y[lo:hi],dt[dl:dh],amounts[dl:dh],sp,sa)
            for j in range(q+2):
                candidate=eta.copy();correction=0.
                if j==0:
                    candidate=root*normals[i,k,j]
                    correction=.5*np.sum((candidate*candidate-eta*eta)/omega)
                elif j==1:
                    candidate+=scale*root*normals[i,k,j]/np.sqrt(max(q,1))
                else:
                    a=j-2;candidate[a]+=scale*root[a]*normals[i,k,j,a]
                proposed=joint_nll(candidate,code,typical[i],indices,omega,times[lo:hi],y[lo:hi],dt[dl:dh],amounts[dl:dh],sp,sa)
                if np.isfinite(proposed) and np.log(uniforms[i,k,j])<value-proposed+correction:
                    eta=candidate;value=proposed;accepted[min(j,2)]+=1
            state[i,k]=eta+mu
    return state,accepted


@njit(cache=True,nogil=True)
def predict_samples(phi,typical,indices,code,oo,times,do,dt,amounts):
    n,c,q=phi.shape
    result=np.empty((c,len(times)))
    for i in range(n):
        lo,hi=oo[i:i+2];dl,dh=do[i:i+2]
        for k in range(c):
            p=typical[i].copy()
            for j in range(q):p[indices[j]]=np.exp(phi[i,k,j])
            result[k,lo:hi]=event_predictions(code,p,times[lo:hi],dt[dl:dh],amounts[dl:dh])
    return result


@njit(cache=True,nogil=True)
def physical_prediction_score(phi,typical,indices,code,oo,times,y,do,dt,amounts,sp,sa):
    """Complete-data residual derivative with physical ETA samples fixed.

    Returns a subject-by-structural-parameter derivative averaged over chains.
    Only non-ETA columns are used by the population decoder chain rule.
    """
    n,chains,q=phi.shape;d=typical.shape[1]
    result=np.zeros((n,d));coefficients=np.empty((4,d))
    for i in range(n):
        lo,hi=oo[i:i+2];dl,dh=do[i:i+2]
        prediction=np.empty(hi-lo);jacobian=np.empty((hi-lo,d))
        for k in range(chains):
            individual=typical[i].copy()
            for j in range(q):individual[indices[j]]=np.exp(phi[i,k,j])
            event_predictions_gradient_into(code,individual,times[lo:hi],
                dt[dl:dh],amounts[dl:dh],prediction,jacobian,coefficients)
            for r in range(hi-lo):
                if prediction[r]<=1e-10:continue
                f=prediction[r];residual=f-y[lo+r]
                raw=(sp*f)**2+sa*sa;v=max(raw,1e-12)
                df=2*residual/v
                if raw>1e-12:df+=2*sp*sp*f*(1/v-residual*residual/(v*v))
                for j in range(d):result[i,j]+=df*jacobian[r,j]/chains
    return result


@njit(cache=True,nogil=True)
def residual_q(predictions,weights,y,sp,sa):
    total=0.
    for h in range(len(weights)):
        subtotal=0.
        for c in range(predictions.shape[1]):
            for j in range(len(y)):
                f=max(predictions[h,c,j],1e-10)
                variance=max((sp*f)**2+sa*sa,1e-12)
                subtotal+=np.log(2*np.pi*variance)+(y[j]-f)**2/variance
        total+=weights[h]*subtotal/predictions.shape[1]
    return total


@njit(cache=True,nogil=True)
def residual_value_gradient(predictions,weights,y,sp,sa):
    value=0.;gp=0.;ga=0.
    for h in range(len(weights)):
        weight=weights[h]/predictions.shape[1]
        for c in range(predictions.shape[1]):
            for j in range(len(y)):
                f=max(predictions[h,c,j],1e-10);r2=(y[j]-f)**2
                raw=(sp*f)**2+sa*sa;v=max(raw,1e-12)
                value+=weight*(np.log(2*np.pi*v)+r2/v)
                if raw>1e-12:
                    factor=weight*(1/v-r2/(v*v))
                    gp+=factor*2*sp*f*f;ga+=factor*2*sa
    return value,gp,ga


class WorkLimit(RuntimeError):
    pass


def chain_diagnostics(draws):
    """Rank/folded split R-hat and initial-positive paired bulk ESS diagnostics.

    Input shape: draw, subject, chain, ETA. These are finite-chain diagnostics,
    not proof that every posterior mode has been visited.
    """
    if len(draws)<40:
        return dict(max_rhat=None,min_bulk_ess=None,passed=False)
    a=np.asarray(draws).transpose(2,0,1,3)
    c,d,n,q=a.shape;half=d//2
    z=np.concatenate([a[:,:half],a[:,-half:]],axis=0).reshape(2*c,half,n*q)
    rhats=[];ess=[]
    def components(v):
        means=v.mean(axis=1);center=v-means[:,None]
        w=np.var(v,axis=1,ddof=1).mean();b=half*np.var(means,ddof=1)
        vp=(half-1)/half*w+b/half
        return center,w,vp
    for j in range(n*q):
        v=z[:,:,j]
        ranked=ndtri((rankdata(v).reshape(v.shape)-.375)/(v.size+.25))
        folded=np.abs(v-np.median(v))
        folded=ndtri((rankdata(folded).reshape(v.shape)-.375)/(v.size+.25))
        for values in (ranked,folded):
            _,w,vp=components(values)
            rhats.append(float(np.sqrt(vp/w)) if w>0 else float('inf'))
        centered,w,vp=components(ranked)
        if vp<=0:ess.append(0.);continue
        cumulative=0.;previous=np.inf
        for lag in range(1,half-1,2):
            pair=0.
            for k in (lag,lag+1):
                ac=np.sum(centered[:,:half-k]*centered[:,k:],axis=1).mean()/half
                pair+=1-(w-ac)/vp
            if pair<0:break
            pair=min(previous,pair);cumulative+=pair;previous=pair
        ess.append(float(min(v.size,v.size/max(1.,1+2*cumulative))))
    return dict(max_rhat=max(rhats),min_bulk_ess=min(ess),
                passed=max(rhats)<1.05 and min(ess)>=100)


@dataclass
class SAEMResult:
    x: np.ndarray
    status: str
    message: str
    trace: list
    diagnostics: dict
    audit: dict
    posterior_pilot: object = None


def fit_saem(study,eta_indices,x0,decode,*,penalty=None,bounds=None,seed=0,
             chains=4,exploration=60,smoothing=120,mstep_iterations=6,
             cpu_budget_seconds=60.,audit_reserve_seconds=10.,
             stability_tolerance=.01,diagnostic_draws=400,diagnostic_burnin=50,callback=None,
             audit_likelihood=True,scaled_mstep=False,mu_regression=False,
             adaptive_phases=False,phase_window=25,block_mstep=False,handoff_policy='legacy',analytic_non_eta=False,conditional_mstep=True):
    """Decode free coordinates to (typical matrix, diagonal omega, sp, sa).

    Q history is retained exactly during smoothing, not truncated/resampled.
    Prediction history is cached unless a structural parameter without ETA
    changes. This supports fixed/free non-ETA parameters at their true cost.
    Fixed parameters/covariates and hint constraints belong to decode/bounds;
    population priors belong to penalty and are counted once.
    """
    if any(type(v)is not int or v<1 for v in
           (chains,exploration,smoothing,mstep_iterations,diagnostic_draws,diagnostic_burnin,phase_window)):
        raise ValueError('positive integer SAEM budgets required')
    if (not np.isfinite([cpu_budget_seconds,audit_reserve_seconds,stability_tolerance]).all()
            or not 0<audit_reserve_seconds<cpu_budget_seconds or stability_tolerance<=0):
        raise ValueError('invalid time reserve or stability tolerance')
    if type(conditional_mstep) is not bool:raise ValueError('conditional_mstep must be boolean')
    if type(analytic_non_eta) is not bool:raise ValueError('analytic_non_eta must be boolean')
    if handoff_policy not in ('legacy','sustained'):raise ValueError('unknown SAEM handoff policy')
    boundary=StudyImportance(study,eta_indices)
    indices=boundary.indices;code=boundary.code
    if not len(indices):raise ValueError('SAEM requires random effects; use a deterministic fit otherwise')
    x=np.asarray(x0,dtype=float).copy()
    penalty=penalty or (lambda point:0.)
    def parameters(point):
        p,om,sp,sa=decode(point)
        p=np.asarray(p,dtype=float);om=np.asarray(om,dtype=float)
        columns=2 if code==0 else 6 if code==3 else 4
        if (p.shape!=(len(study.subject_ids),columns) or not np.isfinite(p).all() or
                np.any((p[:,:-1] if code in (1,3) else p)<=0) or
                (code in (1,3) and np.any(p[:,-1]<0))):
            raise ValueError('invalid structural parameter matrix')
        if om.shape!=(len(indices),) or not np.isfinite(om).all() or np.any(om<1e-8):
            raise ValueError('positive ETA variances at least kernel floor required')
        if not np.isfinite([sp,sa]).all() or min(sp,sa)<0 or sp+sa<=0:
            raise ValueError('invalid residual standard deviations')
        return p,om,float(sp),float(sa)
    for attribute in ('variance_coordinates','variance_eta_indices'):
        if hasattr(decode,attribute):setattr(parameters,attribute,getattr(decode,attribute))
    p,om,sp,sa=parameters(x)
    rng=np.random.default_rng(seed);n=len(study.subject_ids);q=len(indices)
    phi=np.log(p[:,indices])[:,None,:]+rng.normal(size=(n,chains,q))*np.sqrt(om)
    common=(indices,code,study.observation_offsets,study.time,study.observation,
            study.dose_offsets,study.dose_time,study.dose_amount)
    # Warm compiled specializations separately; expose this cost in diagnostics.
    warm=time.perf_counter()
    transition(phi,p,om,*common,sp,sa,np.zeros((n,chains,q+2,q)),
               np.full((n,chains,q+2),.5),1.)
    predict_samples(phi,p,indices,code,study.observation_offsets,study.time,
                    study.dose_offsets,study.dose_time,study.dose_amount)
    residual_q(np.zeros((1,chains,len(study.time))),np.ones(1),study.observation,sp,sa)
    residual_value_gradient(np.zeros((1,chains,len(study.time))),np.ones(1),study.observation,sp,sa)
    if analytic_non_eta and len(indices)<p.shape[1]:
        physical_prediction_score(phi,p,indices,code,study.observation_offsets,study.time,
            study.observation,study.dose_offsets,study.dose_time,study.dose_amount,sp,sa)
    warm=time.perf_counter()-warm
    cpu_start=time.process_time();wall_start=time.perf_counter()
    trace=[];history=[];weights=np.empty(0);cached=None;cached_key=None
    mean=None;second=None;scale=1.;work=dict(prediction_observations=0,residual_terms=0,mstep_calls=0,residual_cache_hits=0,analytic_prediction_score_observations=0,conditional_variance_updates=0)
    from .saem_acceleration import mean_regression_candidate,phase_stable,parameter_windows_stable
    stopped='iteration budget exhausted';completed_iterations=0
    exploration_end=exploration;phase_decisions=[];regression_accepted=0
    sustained=adaptive_phases and handoff_policy=='sustained'
    smoothing_limit=smoothing*(4 if sustained else 1)
    stable_checks=0;handoff_frozen=None
    def check(reserve=True):
        if time.process_time()-cpu_start>=cpu_budget_seconds-(audit_reserve_seconds if reserve else 0):
            raise WorkLimit('CPU budget reached')
    try:
        for iteration in range(exploration+smoothing_limit):
            check()
            p,om,sp,sa=parameters(x)
            phi,accept=transition(phi,p,om,*common,sp,sa,
                rng.normal(size=(n,chains,q+2,q)),
                np.maximum(rng.random((n,chains,q+2)),np.finfo(float).tiny),scale)
            work['prediction_observations']+=chains*(q+3)*len(study.time)
            gamma=1. if iteration<exploration_end else (iteration-exploration_end+2.)**(-.7)
            m=phi.mean(axis=1);s=(phi*phi).mean(axis=1)
            mean=m if mean is None else (1-gamma)*mean+gamma*m
            second=s if second is None else (1-gamma)*second+gamma*s
            if gamma==1.:
                history=[phi.copy()];weights=np.ones(1);cached=None
            else:
                history.append(phi.copy());weights=np.r_[weights*(1-gamma),gamma]
            # Reuse predictions from previous iterations when non-ETA values
            # are unchanged, and append only the newly sampled predictions.
            non_eta=[j for j in range(p.shape[1]) if j not in indices]
            active_gradients=np.arange(len(x))
            residual_cache=None;structural_score_cache=None
            def qfunction(point,with_gradient=False):
                nonlocal cached,cached_key,residual_cache,structural_score_cache
                check();pp,oo,ss,aa=parameters(point)
                key=pp[:,non_eta].tobytes()
                if cached is None or key!=cached_key:
                    records=history
                    prefix=[]
                else:
                    records=history[len(cached):]
                    prefix=list(cached)
                for sample in records:
                    check()
                    prefix.append(predict_samples(sample,pp,indices,code,
                        study.observation_offsets,study.time,study.dose_offsets,
                        study.dose_time,study.dose_amount))
                    work['prediction_observations']+=chains*len(study.time)
                if records:cached=np.ascontiguousarray(prefix)
                cached_key=key
                mu=np.log(pp[:,indices])
                prior=np.sum(np.log(2*np.pi*oo)+(second-2*mu*mean+mu*mu)/oo)
                work['mstep_calls']+=1
                residual_key=(key,float(ss),float(aa))
                if residual_cache is None or residual_cache[0]!=residual_key:
                    residual_cache=(residual_key,residual_value_gradient(cached,weights,study.observation,ss,aa))
                    work['residual_terms']+=len(weights)*chains*len(study.time)
                else:work['residual_cache_hits']+=1
                rv,gp,ga=residual_cache[1]
                value=float(prior+rv+penalty(point))
                if not with_gradient:return value
                dmu=2*(mu-mean)/oo
                dom=n/oo-np.sum(second-2*mu*mean+mu*mu,axis=0)/(oo*oo)
                gradient=np.zeros(len(point))
                for j in active_gradients:
                    h=1e-5*max(1.,abs(point[j]))
                    plus=point.copy();minus=point.copy()
                    plus[j]+=h;minus[j]-=h
                    if bounds is not None:
                        lower,upper=bounds[j]
                        if lower is not None:minus[j]=max(minus[j],lower)
                        if upper is not None:plus[j]=min(plus[j],upper)
                    span=plus[j]-minus[j]
                    if span==0:gradient[j]=0.;continue
                    p1,o1,s1,a1=parameters(plus);p0,o0,s0,a0=parameters(minus)
                    changes_prediction=(not np.array_equal(p1[:,non_eta],pp[:,non_eta]) or
                                        not np.array_equal(p0[:,non_eta],pp[:,non_eta]))
                    primitive_score=None
                    if analytic_non_eta and changes_prediction:
                        if structural_score_cache is None or structural_score_cache[0]!=residual_key:
                            primitive_score=np.zeros_like(pp)
                            for weight,sample in zip(weights,history):
                                check()
                                primitive_score+=weight*physical_prediction_score(sample,pp,indices,code,
                                    study.observation_offsets,study.time,study.observation,
                                    study.dose_offsets,study.dose_time,study.dose_amount,ss,aa)
                                work['prediction_observations']+=chains*len(study.time)
                                work['analytic_prediction_score_observations']+=chains*len(study.time)
                            structural_score_cache=(residual_key,primitive_score)
                        else:primitive_score=structural_score_cache[1]
                    if not analytic_non_eta and changes_prediction:
                        # Free non-ETA parameters affect structural predictions;
                        # retain the exact model and charge their actual work.
                        gradient[j]=(qfunction(plus)-qfunction(minus))/span
                    else:
                        gradient[j]=(np.sum(dmu*(np.log(p1[:,indices])-np.log(p0[:,indices])))+
                            np.dot(dom,o1-o0)+gp*(s1-s0)+ga*(a1-a0)+penalty(plus)-penalty(minus))/span
                        if primitive_score is not None:
                            gradient[j]+=np.sum(primitive_score[:,non_eta]*
                                (p1[:,non_eta]-p0[:,non_eta]))/span
                return value,gradient
            before=qfunction(x)
            origin=x.copy()
            mu_plan=getattr(decode,'mu_reference',None)
            if mu_regression:
                # A regression trial can leave the decoder's valid domain.
                # Reject that trial and retain the ordinary M-step at x.
                try:
                    if mu_plan is not None:
                        candidate,solved=mu_plan.solve(x,parameters,mean,bounds)
                        width=len(mu_plan.coordinates) if solved else 0
                    else:
                        candidate,width=mean_regression_candidate(x,parameters,indices,mean,bounds)
                    if width and np.isfinite(candidate).all():
                        candidate_q=qfunction(candidate)
                        if candidate_q<=before+1e-10:
                            origin=candidate;regression_accepted+=1
                            if mu_plan is not None:
                                active_gradients=np.array([j for j in range(len(x)) if j not in mu_plan.coordinates],dtype=int)
                except (ValueError,ArithmeticError):
                    pass
            variance_coordinates=np.array([],dtype=int)
            if conditional_mstep:
                from .saem_acceleration import conditional_variance_candidate
                candidate,coordinates=conditional_variance_candidate(origin,parameters,mean,second,bounds)
                if len(coordinates) and qfunction(candidate)<=qfunction(origin)+1e-10:
                    origin=candidate;variance_coordinates=coordinates
                    work['conditional_variance_updates']+=1
                    active_gradients=np.array([j for j in active_gradients if j not in coordinates],dtype=int)
            coordinate_scale=mstep_scale(x,parameters,indices,bounds,len(study.time)) if scaled_mstep else np.ones(len(x))
            def scaled_q(z):
                point=origin.copy();point[active_gradients]+=coordinate_scale[active_gradients]*z
                value,g=qfunction(point,True)
                return value,g[active_gradients]*coordinate_scale[active_gradients]
            local_bounds=None if bounds is None else [
                (None if lo is None else (lo-origin[j])/coordinate_scale[j],
                 None if hi is None else (hi-origin[j])/coordinate_scale[j]) for j in active_gradients for lo,hi in [bounds[j]]]
            if len(active_gradients):
                if block_mstep:
                    from .block_optimization import population_coordinates,minimize_blocks
                    pop,obs=population_coordinates(origin,parameters,indices,bounds)
                    blocks=[[i for i,j in enumerate(active_gradients) if j in group] for group in (pop,obs)]
                    optimum=minimize_blocks(scaled_q,np.zeros(len(active_gradients)),
                        local_bounds or [(None,None)]*len(active_gradients),blocks,maxiter=mstep_iterations,ftol=1e-8)
                else:
                    optimum=minimize(scaled_q,np.zeros(len(active_gradients)),method='L-BFGS-B',jac=True,bounds=local_bounds,
                        options=dict(maxiter=mstep_iterations,ftol=1e-8,gtol=1e-5,maxls=15))
            else:
                from scipy.optimize import OptimizeResult
                optimum=OptimizeResult(x=np.empty(0),fun=qfunction(origin),jac=np.empty(0),nit=0,success=True)
            # A generalized M step need only improve the current Q. This is not
            # a claim that the population likelihood has converged.
            old=x.copy()
            update_accepted=bool(np.isfinite(optimum.fun) and optimum.fun<=before+1e-9)
            if update_accepted:
                x=origin.copy();x[active_gradients]+=coordinate_scale[active_gradients]*optimum.x
            if iteration<exploration_end:
                rw=accept[1:].sum()/(n*chains*(q+1))
                scale=float(np.clip(scale*np.exp((rw-.3)/np.sqrt(iteration+1)),.05,4.))
            completed_iterations=iteration+1
            record=dict(iteration=iteration+1,phase='exploration' if gamma==1 else 'smoothing',
                        step_size=gamma,complete_q=float(optimum.fun if update_accepted else before),
                        mstep_success=bool(optimum.success),x=x.tolist(),
                        mstep_iterations=int(optimum.nit),mstep_accepted=update_accepted,
                        mstep_q_improvement=float(before-optimum.fun),
                        mstep_scaled_score_max=float(np.max(np.abs(optimum.jac),initial=0.)),
                        regression_coordinates=(mu_plan.coordinates.tolist() if mu_plan is not None and len(active_gradients)<len(x) else []),
                        conditional_variance_coordinates=variance_coordinates.tolist(),
                        nonlinear_coordinates=active_gradients.tolist(),
                        parameter_step=float(np.max(np.abs(x-old))),
                        acceptance=(accept/np.array([n*chains,n*chains,n*chains*q])).tolist(),
                        cpu_seconds=time.process_time()-cpu_start)
            trace.append(record)
            if callback is not None and ((iteration+1)%10==0):callback(record)
            phase_records=[r for r in trace if r['phase']==record['phase']]
            if adaptive_phases and len(phase_records)%phase_window==0:
                scheduling_stable=phase_stable(phase_records,phase_window,stability_tolerance)
                if record['phase']=='exploration':
                    if scheduling_stable:
                        exploration_end=iteration+1
                        phase_decisions.append(dict(iteration=iteration+1,event='begin_smoothing'))
                elif not sustained:
                    if scheduling_stable:
                        stopped='phase stability reached; independent marginal refinement required'
                        phase_decisions.append(dict(iteration=iteration+1,event='handoff_to_refinement'))
                        break
                else:
                    eligible=scheduling_stable and parameter_windows_stable(phase_records,stability_tolerance)
                    stable_checks=stable_checks+1 if eligible else 0
                    if stable_checks>=2:
                        # A frozen-target check is valid for this exact x only.
                        fp,fo,fs,fa=parameters(x);probe_phi=phi.copy();draws=[]
                        for draw in range(diagnostic_burnin+diagnostic_draws):
                            check()
                            probe_phi,_=transition(probe_phi,fp,fo,*common,fs,fa,
                                rng.normal(size=(n,chains,q+2,q)),
                                np.maximum(rng.random((n,chains,q+2)),np.finfo(float).tiny),scale)
                            work['prediction_observations']+=chains*(q+3)*len(study.time)
                            if draw>=diagnostic_burnin:draws.append(probe_phi.copy())
                        mixing_check=chain_diagnostics(draws)
                        phase_decisions.append(dict(iteration=iteration+1,event='sustained_handoff_check',
                            parameter_stable=True,chain_diagnostics=mixing_check))
                        phi=probe_phi
                        if mixing_check['passed']:
                            handoff_frozen=draws
                            stopped='sustained parameter stability and fixed-target mixing checks passed'
                            phase_decisions.append(dict(iteration=iteration+1,event='handoff_to_refinement'))
                            break
                        stable_checks=0
            if iteration>=exploration_end and iteration-exploration_end+1>=smoothing_limit:break
    except WorkLimit as error:
        stopped=str(error)
    except (ValueError,ArithmeticError,RuntimeError) as error:
        stopped=f'{type(error).__name__}: {error}'
    # Sampling uncertainty is audited separately from the SAEM Q trace.
    audit=dict(completed=False,subjects=[],quality_passed=False)
    p,om,sp,sa=parameters(x)
    # R-hat/ESS require a fixed target. Draw an explicitly separate conditional
    # chain segment at the final population parameters, within the same budget.
    frozen=handoff_frozen if handoff_frozen is not None else []
    try:
        for iteration in range(0 if handoff_frozen is not None else diagnostic_burnin+diagnostic_draws):
            check(False)
            phi,_=transition(phi,p,om,*common,sp,sa,
                rng.normal(size=(n,chains,q+2,q)),
                np.maximum(rng.random((n,chains,q+2)),np.finfo(float).tiny),scale)
            work['prediction_observations']+=chains*(q+3)*len(study.time)
            if iteration>=diagnostic_burnin:frozen.append(phi.copy())
    except WorkLimit:
        pass
    try:
        for i,args in enumerate(boundary._arguments(p,om,sp,sa) if audit_likelihood else ()):
            check(False)
            # Conditional states provide a pilot only. A prior component and
            # independent scrambles still guard/audit finite-sample integration.
            pilot=np.array(frozen)[:,i].reshape(-1,q) if frozen else phi[i]
            center=pilot.mean(axis=0)-np.log(p[i,indices])
            covariance=np.atleast_2d(np.cov(pilot.T,bias=True)) if len(pilot)>1 else np.zeros((q,q))
            covariance=2*covariance+.1*np.diag(om)
            for power in (10,12,14,16):
                rows=[]
                for replica in range(4):
                    check(False)
                    proposal=FixedImportance.prepare(om,power=power,
                        seed=int(np.random.SeedSequence([seed,i,replica,991]).generate_state(1)[0]),
                        center=center,covariance=covariance)
                    work['prediction_observations']+=len(proposal.samples)*len(args[4])
                    rows.append(proposal.evaluate(*args))
                r=replica_report(rows)
                passed=r['empirical_ofv_se']<=.005 and r['minimum_ess']>=100 and r['maximum_weight']<=.01
                if passed:break
            audit['subjects'].append(dict(subject_id=study.subject_ids[i],power=power,quality_passed=passed,**r))
        if audit_likelihood:
            audit.update(completed=True,quality_passed=all(r['quality_passed'] for r in audit['subjects']),
                         empirical_ofv_se=float(np.sqrt(sum(r['empirical_ofv_se']**2 for r in audit['subjects']))),
                         ofv=sum(r['ofv'] for r in audit['subjects'])+float(penalty(x)))
        else:
            audit['message']='intermediate likelihood audit omitted; final marginal audit required'
    except (WorkLimit,ValueError,ArithmeticError,RuntimeError) as error:
        audit['message']=str(error)
    smooth_records=[r for r in trace if r['phase']=='smoothing']
    stable=parameter_windows_stable(smooth_records,stability_tolerance)
    mixing=chain_diagnostics(frozen)
    criteria=stable and mixing['passed'] and audit['completed'] and audit['quality_passed']
    diagnostics=dict(**work,cpu_seconds=time.process_time()-cpu_start,
        wall_seconds=time.perf_counter()-wall_start,compilation_warmup_seconds=warm,
        iterations=completed_iterations,parameter_windows_stable=stable,
        diagnostic_draws=len(frozen),diagnostic_burnin=diagnostic_burnin,
        chain_mixing_certified=False,chain_diagnostics=mixing,
        empirical_stability_criteria_met=criteria,marginal_stationarity_checked=False,
        convergence_criteria_met=False,estimator='generalized_saem',seed=seed,
        scaled_mstep=scaled_mstep,mu_regression=mu_regression,block_mstep=block_mstep,analytic_non_eta=analytic_non_eta,conditional_mstep=conditional_mstep,
        mu_reference_coordinates=(getattr(decode,'mu_reference').coordinates.tolist()
            if getattr(decode,'mu_reference',None) is not None and mu_regression else []),
        regression_accepted=regression_accepted,adaptive_phases=adaptive_phases,
        phase_decisions=phase_decisions,handoff_policy=handoff_policy,
        smoothing_iteration_cap=smoothing_limit,handoff_sampling_reused=handoff_frozen is not None)
    # Small SAEM updates can hide EM's slow directions. These empirical checks
    # establish stability, not marginal-likelihood stationarity.
    pilot_states=np.asarray(frozen) if frozen else phi[None,...]
    posterior_pilot=[]
    for i in range(n):
        samples=pilot_states[:,i].reshape(-1,q)
        centered=samples-samples.mean(axis=0)
        posterior_pilot.append((samples.mean(axis=0),centered.T@centered/len(samples)))
    return SAEMResult(x,'stabilized' if criteria else 'partial',
        'empirical criteria met; marginal stationarity still requires verification' if criteria else stopped,
        trace,diagnostics,audit,posterior_pilot)

