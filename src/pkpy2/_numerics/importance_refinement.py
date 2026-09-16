"""Adaptive support-limited marginal likelihood refinement after SAEM.

Proposal samples stay fixed within a stage. Independent proposals validate
steps, adjust the trust radius, and audit projected marginal scores. No drug
rules and no relabeling of a local-bound or iteration-limit stop as convergence.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize
from .importance_objective import PhysicalImportance
from .standardized_importance import StandardizedImportance
from .response_importance import ResponseImportance,TransportSupportError
from .subject_work import SubjectWork


def damped_bfgs(hessian,step,gradient_change):
    """Positive-definite Powell-damped observed-objective secant update."""
    bs=hessian@step;sb=float(step@bs);sy=float(step@gradient_change)
    if not np.isfinite([sb,sy]).all() or sb<=1e-16:return hessian
    if sy<.2*sb:
        weight=.8*sb/(sb-sy)
        gradient_change=weight*gradient_change+(1-weight)*bs
        sy=float(step@gradient_change)
    updated=hessian-np.outer(bs,bs)/sb+np.outer(gradient_change,gradient_change)/sy
    return (updated+updated.T)*.5


def projected_score(x,g,bounds):
    out=np.array(g,dtype=float,copy=True)
    if bounds is not None:
        for j,(lower,upper) in enumerate(bounds):
            if lower is not None and x[j]<=lower+1e-7 and out[j]>0:out[j]=0.
            if upper is not None and x[j]>=upper-1e-7 and out[j]<0:out[j]=0.
    return out


def support_bounds(x,decode,indices,bounds,radius,standardized=False):
    p,om,sp,sa=decode(x);p=np.asarray(p);om=np.asarray(om)
    indices=np.asarray(indices,dtype=int);other=[j for j in range(p.shape[1]) if j not in indices]
    result=[]
    for j in range(len(x)):
        if bounds is not None and bounds[j][0] is not None and bounds[j][0]==bounds[j][1]:
            result.append(bounds[j]);continue
        shifted=x.copy();h=1e-5*max(1.,abs(x[j]));shifted[j]+=h
        if bounds is not None and bounds[j][1] is not None and shifted[j]>bounds[j][1]:
            shifted[j]=x[j]-h
        h=shifted[j]-x[j]
        pp,oo,ss,aa=decode(shifted);pp=np.asarray(pp)
        derivative=(np.log(pp[:,indices])-np.log(p[:,indices]))/h
        metric=float(np.max(np.sum(derivative**2/(1. if standardized else om),axis=1)))
        metric+=float(np.sum(((np.log(oo)-np.log(om))/h)**2))
        metric+=((ss-sp)/h/max(sp,1e-6))**2+((aa-sa)/h/max(sa,1e-6))**2
        if other:metric+=float(np.max(np.sum(((pp[:,other]-p[:,other])/h/np.maximum(1.,np.abs(p[:,other])))**2,axis=1)))
        width=radius/np.sqrt(max(metric,1.))
        lower,upper=(None,None) if bounds is None else bounds[j]
        result.append((max(x[j]-width,lower) if lower is not None else x[j]-width,
                       min(x[j]+width,upper) if upper is not None else x[j]+width))
    return result


@dataclass
class RefinementResult:
    x: np.ndarray
    ofv: float
    status: str
    message: str
    stages: list
    audit: dict
    cpu_seconds: float
    posterior: dict | None = None
    candidate_workspace: object | None = None
    sample_reuse: dict | None = None
    initial_audit: dict | None = None


def refine_importance(study,eta_indices,x0,decode,*,penalty=None,bounds=None,
                      seed=0,cpu_budget_seconds=300.,max_stages=60,
                      stage_iterations=40,score_tolerance=.2,ofv_tolerance=.05,
                      minimum_ess=100,max_radius=4.,stage_ftol=1e-12,callback=None,posterior_pilot=None,
                      transport_posterior=False,integration_coordinates='physical',step_method='lbfgs',
                      validation_reference='candidate',workers=1,reuse_workspace=None,audit_first=False,local_proposal_recovery=False,endpoint_backtracking=False,bridge_validation=False,short_circuit_audit=False,analytic_non_eta=False,curvature_refresh_limit=0):
    if type(curvature_refresh_limit) is not int or curvature_refresh_limit < 0:
        raise ValueError('curvature_refresh_limit must be a nonnegative integer')
    if type(analytic_non_eta) is not bool:raise ValueError('analytic_non_eta must be boolean')
    if type(short_circuit_audit) is not bool:raise ValueError('short_circuit_audit must be boolean')
    if type(bridge_validation) is not bool:raise ValueError('bridge_validation must be boolean')
    if type(endpoint_backtracking) is not bool:raise ValueError('endpoint_backtracking must be boolean')
    if type(local_proposal_recovery) is not bool:raise ValueError('local_proposal_recovery must be boolean')
    if validation_reference not in ('candidate','midpoint'):raise ValueError('unknown validation reference')
    if step_method not in ('lbfgs','block_lbfgs','trust_bfgs'):raise ValueError('unknown marginal step method')
    if integration_coordinates not in ('physical','standardized','response','response_shape'):
        raise ValueError('unknown importance integration coordinates')
    if reuse_workspace is not None:
        import hashlib
        lineage=repr((seed,tuple(reuse_workspace.source_seeds))).encode()
        seed=(1<<160)+int.from_bytes(hashlib.sha256(lineage).digest()[:16],'big')
    if (not np.isfinite([cpu_budget_seconds,score_tolerance,ofv_tolerance,minimum_ess,max_radius,stage_ftol]).all()
            or cpu_budget_seconds<=0 or min(score_tolerance,ofv_tolerance,minimum_ess)<=0
            or max_radius<.25
            or stage_ftol<=0
            or type(max_stages)is not int or max_stages<1
            or type(stage_iterations)is not int or stage_iterations<1):
        raise ValueError('positive finite refinement budgets/tolerances required')
    start=time.process_time();x=np.asarray(x0,dtype=float).copy()
    work=SubjectWork(workers,len(study.subject_ids))
    penalty=penalty or (lambda point:0.)
    stages=[];radius=.25;power=12;obj=None;audit={};value=float('nan');hessian=None
    curvature_initialization=None
    failed_audits_since_refresh=0;curvature_refreshes=0
    posterior=None
    candidate_workspace=None
    initial_audit=None
    precision_retries=0
    reuse_active=False;shared_working=None;shared_validation=None
    reuse_report=dict(requested=reuse_workspace is not None,used=False,stages=0,
                      refresh_reason=None,final_audits_reused=False,effective_seed=str(seed))
    powers=np.full(len(study.subject_ids),12,dtype=int)
    message='stage budget exhausted';status='partial'
    def check():
        if time.process_time()-start>=cpu_budget_seconds:raise TimeoutError('CPU budget exhausted')
    def prepare(point,power,stream,uniform=False,pilot=None):
        check()
        if integration_coordinates in ('response','response_shape') and not uniform:
            return ResponseImportance(study,eta_indices,*decode(point),reference_x=point,
                decode=decode,bounds=bounds,power=power,seed=stream,pilot_power=14,
                sample_powers=powers,posterior_pilot=pilot,transport_shape=integration_coordinates=='response_shape',work=work)
        # Final audits retain a separate coordinate system as well as new samples.
        evaluator=StandardizedImportance if integration_coordinates=='standardized' and not uniform else PhysicalImportance
        return evaluator(study,eta_indices,*decode(point),power=power,
                                  seed=stream,pilot_power=14,
                                  sample_powers=None if uniform else powers,posterior_pilot=pilot,work=work)
    evaluation_cache=None
    def evaluate(evaluator,point):
        nonlocal evaluation_cache
        check()
        # Repeated requests at the same point on the same frozen bank have
        # identical values, gradients and diagnostic state. Never cross banks.
        if (evaluation_cache is not None and evaluation_cache[0] is evaluator
                and np.array_equal(evaluation_cache[1],point)):
            return evaluation_cache[2],evaluation_cache[3].copy()
        evaluation_cache=None
        if analytic_non_eta and type(evaluator) is PhysicalImportance:
            v,g=evaluator.value_gradient(point,decode,penalty,bounds,analytic_non_eta=True)
        else:
            v,g=evaluator.value_gradient(point,decode,penalty,bounds)
        evaluation_cache=(evaluator,point.copy(),v,g.copy())
        return v,g
    def run_audit(point,stage,row=None):
        audits=[];moments_banks=[];candidate_banks=[]
        from .candidate_workspace import CandidateBank,CandidateWorkspace
        for replica in range(2):
            stream=seed+9000001+2*stage+replica
            external=prepare(point,16,stream,uniform=True)
            av,ag=evaluate(external,point);apg=projected_score(point,ag,bounds)
            audits.append(dict(seed=str(stream),ofv=av,score=ag.tolist(),projected_score=apg.tolist(),
                projected_score_max=float(np.max(np.abs(apg))),minimum_ess=external.last_min_ess))
            # Either failed replica is sufficient to reject this point. A
            # successful audit still requires two independent replicas below.
            replica_passed=(np.max(np.abs(apg))<=score_tolerance and
                            (not len(eta_indices) or external.last_min_ess>=minimum_ess))
            if short_circuit_audit and replica == 0 and not replica_passed:
                result=dict(x=point.tolist(),replicas=audits,completed=False,
                            decision_complete=True,passed=False,ofv_range=None,
                            stopped_early=True,reason='first_replica_failed')
                if row is not None:
                    row['stationarity_audit']=result
                    if callback is not None:callback(row)
                return result,None,None,external
            typical,variance,_,__=decode(point);moments=external.posterior_pilot()
            moments_banks.append(dict(
                eta_mean=[(m-np.log(np.asarray(typical)[i,eta_indices])).tolist() for i,(m,c) in enumerate(moments)],
                eta_variance=[np.diag(c).tolist() for m,c in moments]))
            if np.max(np.abs(apg))<=score_tolerance and external.last_min_ess>=minimum_ess:
                candidate_banks.append(CandidateBank.capture(external,*decode(point)))
            if row is not None:
                row['stationarity_audit']=dict(x=point.tolist(),replicas=audits,completed=False)
                if callback is not None:callback(row)
        result=dict(x=point.tolist(),replicas=audits,completed=True,ofv_range=abs(audits[0]['ofv']-audits[1]['ofv']))
        result['passed']=bool(result['ofv_range']<=ofv_tolerance and all(
            r['projected_score_max']<=score_tolerance and (not len(eta_indices) or r['minimum_ess']>=minimum_ess) for r in audits))
        post=None;workspace=None
        if result['passed']:
            post=dict(subject_ids=list(study.subject_ids),eta_indices=list(eta_indices),omega=np.asarray(variance).tolist(),
                replicas=moments_banks,source='passed_independent_physical_audits',x=point.tolist())
            if len(candidate_banks)==2:
                workspace=CandidateWorkspace(tuple(study.subject_ids),tuple(candidate_banks),
                    tuple(int(r['seed']) for r in audits))
        if row is not None:
            row['stationarity_audit']=result
            if callback is not None:callback(row)
        return result,post,workspace,external

    try:
        if audit_first:
            audit,posterior,candidate_workspace,obj=run_audit(x,-1)
            initial_audit=audit
            if audit['passed']:
                value=float(np.mean([r['ofv'] for r in audit['replicas']]))
                status='converged';message='SAEM point passed independent marginal stationarity; correction unnecessary'
                max_stages=0
            else:
                # Reuse the failed audit's integration to correct the retained
                # point. No new preliminary optimizer or SAEM restart.
                powers[:]=16;power=16
        if reuse_workspace is not None:
            snapshots=[getattr(b,'physical',None) for b in reuse_workspace.banks]
            if obj is None and len(snapshots)==2 and all(s is not None and s.compatible(study,eta_indices) for s in snapshots):
                shared_working,shared_validation=[PhysicalImportance.from_snapshot(s,study,eta_indices,work) for s in snapshots]
                evaluate(shared_working,x);evaluate(shared_validation,x)
                if min(shared_working.last_min_ess,shared_validation.last_min_ess)>=minimum_ess:
                    obj=shared_working;reuse_active=True;reuse_report['used']=True
                    powers=np.array([max(int(np.log2(len(s.phi[i]))) for s in snapshots)
                                     for i in range(len(study.subject_ids))])
                    power=int(powers.max())
                else:reuse_report['refresh_reason']='initial_support_inadequate'
            else:reuse_report['refresh_reason']=('independent_audit_precedence' if obj is not None
                                               else 'model_data_or_eta_coordinates_incompatible')
        if obj is None:obj=prepare(x,power,seed,pilot=posterior_pilot)
        if status!='converged':value,gradient=evaluate(obj,x)
        for stage in range(max_stages):
            check();before,gradient=evaluate(obj,x)
            value=before
            working_before_rows=obj.last_subject_results
            working_before_ess=obj.last_min_ess
            local_bounds=support_bounds(x,decode,eta_indices,bounds,radius,
                                        standardized=isinstance(obj,(StandardizedImportance,ResponseImportance)))
            scale=np.array([max((hi-lo)*.5,1e-12) for lo,hi in local_bounds])
            origin=x.copy()
            def scaled_objective(z):
                v,g=evaluate(obj,origin+scale*z)
                return v,g*scale
            z_bounds=[((lo-origin[j])/scale[j],(hi-origin[j])/scale[j])
                      for j,(lo,hi) in enumerate(local_bounds)]
            if step_method=='trust_bfgs':
                if hessian is None:
                    # Initialize from the marginal objective's curvature, not
                    # complete-data information or an arbitrary unit matrix.
                    curvature_bank=obj
                    if isinstance(obj,(ResponseImportance,StandardizedImportance)):
                        # The true marginal Hessian is coordinate independent.
                        # A separate physical bank caches PK predictions across
                        # ETA-population perturbations, avoiding path derivatives
                        # in every Hessian column. Use higher integration precision.
                        check()
                        curvature_bank=PhysicalImportance(study,eta_indices,*decode(x),
                            power=max(14,power),sample_powers=np.maximum(14,powers),
                            seed=seed+15485863,pilot_power=14,work=work)
                    raw=np.zeros((len(x),len(x)))
                    for j in range(len(x)):
                        h=1e-3*max(1.,abs(x[j]));a=x.copy();b=x.copy();a[j]+=h;b[j]-=h
                        if bounds is not None:
                            lo,hi=bounds[j]
                            if lo is not None:b[j]=max(b[j],lo)
                            if hi is not None:a[j]=min(a[j],hi)
                        if a[j]!=b[j]:raw[:,j]=(evaluate(curvature_bank,a)[1]-evaluate(curvature_bank,b)[1])/(a[j]-b[j])
                    units=np.maximum(scale/radius,1e-6)
                    scaled=(raw+raw.T)*.5*units[:,None]*units[None,:]
                    eigenvalues,vectors=np.linalg.eigh(scaled)
                    curvature_initialization=dict(scaled_eigenvalues=eigenvalues.tolist(),
                        regularized_directions=int(np.sum(eigenvalues<1e-4)),
                        method=('independent_physical_bank_marginal_gradient_differences'
                                if curvature_bank is not obj else 'fixed_bank_marginal_gradient_differences'))
                    # Trust bounds and actual descent guard the weak/negative
                    # directions; regularization only makes the subproblem SPD.
                    hessian=((vectors*np.maximum(eigenvalues,1e-4))@vectors.T)/units[:,None]/units[None,:]
                    evaluate(obj,x)
                hz=hessian*scale[:,None]*scale[None,:];gz=gradient*scale
                def quadratic(z):return float(gz@z+.5*z@hz@z),gz+hz@z
                # Solve a cheap bounded local quadratic; no repeated PK solves
                # inside this subproblem. Actual objective is evaluated below.
                fit=minimize(quadratic,np.zeros_like(x),jac=True,method='L-BFGS-B',bounds=z_bounds,
                    options=dict(maxiter=100,ftol=1e-12,gtol=1e-8,maxls=30))
                step_backtracks=0
                # A preliminary Hessian can be inaccurate. Enforce descent on
                # the working bank before spending an independent validation.
                for step_backtracks in range(20):
                    try:
                        fit.fun=scaled_objective(fit.x)[0]
                    except TransportSupportError:
                        fit.fun=float('inf')
                    if fit.fun<=before+1e-4*float(gz@fit.x)+1e-10:break
                    fit.x=fit.x*.5
                model_improvement=-quadratic(fit.x)[0]
                fit.fun=scaled_objective(fit.x)[0]
            else:
                step_backtracks=0
                model_improvement=None
                if step_method=='block_lbfgs':
                    from .block_optimization import population_coordinates,minimize_blocks
                    blocks=population_coordinates(origin,decode,eta_indices,local_bounds)
                    fit=minimize_blocks(scaled_objective,np.zeros_like(x),z_bounds,blocks,
                                        maxiter=stage_iterations,ftol=stage_ftol)
                else:
                    fit=minimize(scaled_objective,np.zeros_like(x),jac=True,method='L-BFGS-B',
                        bounds=z_bounds,options=dict(maxiter=stage_iterations,maxls=20,ftol=stage_ftol,gtol=1e-4,
                                     maxcor=min(30,max(10,len(x)))))
            candidate=origin+scale*fit.x
            # Capture identical candidate evaluations on both independent banks.
            _,working_score=evaluate(obj,candidate)
            working_rows=obj.last_subject_results
            transported=obj.posterior_pilot() if transport_posterior and len(eta_indices) else None
            validation_point=(x+candidate)*.5 if validation_reference=='midpoint' else candidate
            fresh=(shared_validation if reuse_active else
                   prepare(validation_point,power,seed+100003*(stage+1),pilot=transported))
            # Compare both endpoints on the SAME independent bank. Comparing
            # old and new estimates from different banks confounds integration
            # noise with actual improvement and corrupts trust-radius updates.
            validation_before,validation_gradient_before=evaluate(fresh,x)
            before_ess=fresh.last_min_ess
            validation_before_rows=fresh.last_subject_results
            independent,score=evaluate(fresh,candidate)
            endpoint_backtracks=0
            # Keep both banks fixed: no new random draws and no optimizer rerun.
            # Only localize a candidate if the retained endpoint already agrees.
            # Remaining uncertainty still fails the normal acceptance gate.
            if (endpoint_backtracking and not reuse_active and abs(validation_before-before)<=ofv_tolerance
                    and (not len(eta_indices) or min(working_before_ess,before_ess)>=minimum_ess)):
                for _ in range(4):
                    candidate_ok=(not len(eta_indices) or
                        min(min(r['ess'] for r in working_rows),fresh.last_min_ess)>=minimum_ess)
                    tolerance=max(ofv_tolerance,.1*max(before-float(fit.fun),0.))
                    if candidate_ok and abs(independent-float(fit.fun))<=tolerance:break
                    candidate=(x+candidate)*.5
                    fit.fun,working_score=evaluate(obj,candidate)
                    working_rows=obj.last_subject_results
                    independent,score=evaluate(fresh,candidate)
                    endpoint_backtracks+=1
                    model_improvement=None
            bridge_used=False
            bridge_gate=max(ofv_tolerance,.1*max(before-float(fit.fun),0.))
            bridge_adequate=(not len(eta_indices) or min(working_before_ess,before_ess,
                min(r['ess'] for r in working_rows),fresh.last_min_ess)>=minimum_ess)
            if (bridge_validation and not reuse_active and np.max(powers)>=16 and
                    (not bridge_adequate or abs(independent-float(fit.fun))>bridge_gate or
                     abs(validation_before-before)>bridge_gate)):
                # Independent balanced proposals cover both endpoints. Each bank
                # retains the same total count as one power-16 audit bank.
                from .bridge_importance import endpoint_bridge
                left,right=decode(x),decode(candidate)
                check()
                obj=endpoint_bridge(study,eta_indices,left,right,power=16,
                    seed=seed+31000001+2*stage,work=work)
                check()
                fresh=endpoint_bridge(study,eta_indices,left,right,power=16,
                    seed=seed+31000002+2*stage,work=work)
                before,_=evaluate(obj,x);working_before_ess=obj.last_min_ess
                working_before_rows=obj.last_subject_results
                fit.fun,working_score=evaluate(obj,candidate);working_rows=obj.last_subject_results
                validation_before,validation_gradient_before=evaluate(fresh,x)
                before_ess=fresh.last_min_ess;validation_before_rows=fresh.last_subject_results
                independent,score=evaluate(fresh,candidate)
                value=before;model_improvement=None;bridge_used=True
                powers[:]=16;power=16
            gap=independent-float(fit.fun);predicted=before-float(fit.fun)
            actual=validation_before-independent
            denominator=predicted if model_improvement is None else model_improvement
            ratio=actual/denominator if denominator>1e-8 else 0.
            adequate=not len(eta_indices) or min(working_before_ess,before_ess,
                min(r['ess'] for r in working_rows),fresh.last_min_ess)>=minimum_ess
            # Both endpoints participate in the decision. An inaccurate old
            # endpoint can invalidate a step even when the candidate is precise.
            difficult=[i for i,(a0,b0,a1,b1) in enumerate(zip(working_before_rows,
                validation_before_rows,working_rows,fresh.last_subject_results))
                if any((len(eta_indices) and min(a['ess'],b['ess'])<minimum_ess)
                    or abs(a['ofv']-b['ofv'])>.005
                    or np.max(np.abs(a['scaled_score']-b['scaled_score']))>.05
                    for a,b in ((a0,b0),(a1,b1)))]
            agreement_tolerance=max(ofv_tolerance,.1*max(predicted,0.))
            support_agreement=(abs(gap)<=agreement_tolerance and
                               abs(validation_before-before)<=agreement_tolerance)
            accepted=adequate and support_agreement and actual>=-ofv_tolerance
            decision=('integration_unresolved' if not adequate or not support_agreement
                      else 'accepted' if accepted else 'objective_worsened')
            pg=projected_score(candidate,score,bounds)
            working_pg=projected_score(candidate,working_score,bounds)
            row=dict(stage=stage,radius=radius,power=power,accepted=accepted,
                shared_physical_samples=reuse_active,bridge_validation_used=bridge_used,
                workers=work.workers,
                integration_coordinates=(('response_shape' if obj.transport_shape else 'response') if isinstance(obj,ResponseImportance) else
                    'standardized' if isinstance(obj,StandardizedImportance) else 'physical'),
                decision=decision,
                step_method=step_method,quadratic_predicted_improvement=model_improvement,
                validation_reference=validation_reference,
                step_backtracks=step_backtracks,endpoint_backtracks=endpoint_backtracks,
                curvature_initialization=curvature_initialization if stage==0 else None,
                optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
                block_updates=getattr(fit,'block_records',None),
                iterations=int(fit.nit),before_ofv=value,fixed_sample_ofv=float(fit.fun),
                independent_ofv=independent,independent_gap=gap,improvement_ratio=ratio,
                validation_before_ofv=validation_before,paired_improvement=actual,
                support_agreement=support_agreement,
                validation_before_gap=validation_before-before,
                projected_score_max=float(np.max(np.abs(pg))),minimum_ess=fresh.last_min_ess,
                working_candidate_minimum_ess=min(r['ess'] for r in working_rows),
                working_retained_minimum_ess=working_before_ess,validation_retained_minimum_ess=before_ess,
                projected_score=pg.tolist(),
                working_projected_score=working_pg.tolist(),
                working_projected_score_max=float(np.max(np.abs(working_pg))),
                sampling_power_counts={str(int(p)):int(np.sum(powers==p)) for p in np.unique(powers)},
                integration_disagreement_subjects=len(difficult),
                x=candidate.tolist(),cpu_seconds=time.process_time()-start)
            stages.append(row)
            if reuse_active:reuse_report['stages']+=1
            if callback is not None:callback(row)
            if not accepted:
                if decision=='objective_worsened':
                    radius=max(.025,radius*.5)
                else:
                    if reuse_active:
                        # The retained population point is unchanged. Replace
                        # both reused banks before proposing another step.
                        reuse_active=False;reuse_report['refresh_reason']='support_or_replica_disagreement'
                        obj=prepare(x,power,seed+700001+stage)
                        hessian=None
                        continue
                    # Numerical uncertainty is not evidence of a poor direction.
                    # Keep the radius and improve integration before deciding.
                    targets=difficult or list(range(len(powers)))
                    increased=powers.copy()
                    increased[targets]=np.minimum(16,increased[targets]+1)
                    if np.array_equal(increased,powers):
                        if not local_proposal_recovery:
                            message='integration precision exhausted before step decision'
                            break
                        # A full-size bank can still be a poor proposal for a
                        # distant candidate. Recenter at the retained point and
                        # test a smaller local move; never accept the failed one.
                        # A candidate-centered bank may itself extrapolate
                        # badly back to x. Diagnose x with an independently
                        # centered bank before classifying retained precision.
                        centered=prepare(x,power,seed+17000001+stage)
                        centered_value,_=evaluate(centered,x)
                        retained_agreement=(abs(centered_value-before)<=ofv_tolerance
                            and (not len(eta_indices) or min(working_before_ess,centered.last_min_ess)>=minimum_ess))
                        row['centered_retained_gap']=float(centered_value-before)
                        if precision_retries>=6 or (retained_agreement and radius<=.025):
                            message='integration unresolved after bounded local proposal recovery'
                            row['precision_recovery']='exhausted'
                            break
                        precision_retries+=1
                        if retained_agreement:
                            radius=max(.025,radius*.5)
                            recovery='shrink_candidate_step'
                        else:
                            recovery='recenter_retained_point'
                            # Repeated disagreement at the retained point is
                            # not fixed by interpreting a rejected step as bad.
                            if precision_retries>2:
                                message='retained-point integration unresolved after recentering'
                                row['precision_recovery']='retained_point_unresolved'
                                break
                        row.update(precision_recovery=recovery,
                            retained_endpoint_agreement=bool(retained_agreement),
                            precision_retries=precision_retries,next_radius=radius)
                        obj=centered
                        hessian=None
                        if callback is not None:callback(row)
                        continue
                    powers=increased
                    # The rejected candidate is not the retained target. Reweight
                    # at x before transporting moments for its replacement bank.
                    evaluate(obj,x)
                    retained=obj.posterior_pilot() if transport_posterior and len(eta_indices) else None
                    power=int(powers.max());obj=prepare(x,power,seed+700001+stage,pilot=retained)
                continue
            if step_method=='trust_bfgs':
                # Both scores belong to the same independent bank/transport.
                # Do not form secants by subtracting different sample banks.
                hessian=damped_bfgs(hessian,candidate-x,score-validation_gradient_before)
            precision_retries=0
            x=candidate;value=independent;obj=shared_working if reuse_active else fresh
            # An audit belongs only to its exact candidate, never a later step.
            audit={}
            # A bridge recomputes both objective values. Agreement of those
            # values is not evidence that the original optimizer model predicted
            # the improvement, so it cannot authorize radius expansion.
            if not bridge_used and actual>ofv_tolerance and ratio>.75 and abs(gap)<max(ofv_tolerance,.1*predicted):
                radius=min(max_radius,radius*1.5)
            elif ratio<.25 and predicted>ofv_tolerance:
                radius=max(.025,radius*.5)
            # A small score on a single bank is not stationarity evidence.
            if min(np.max(np.abs(pg)),np.max(np.abs(working_pg)))<max(.5,2*score_tolerance):
                audit,posterior,candidate_workspace,external=run_audit(x,stage,row)
                if audit['passed']:
                    value=float(np.mean([r['ofv'] for r in audit['replicas']]))
                    status='converged'
                    message='two independent integrations agree and projected marginal scores pass'
                    break
                # Optimize the precision that just revealed nonstationarity.
                # Dropping back to a coarse bank can repeatedly optimize its
                # sampling error and fail the same independent audit.
                if reuse_active:
                    reuse_active=False;reuse_report['refresh_reason']='independent_stationarity_audit_failed'
                value=audit['replicas'][-1]['ofv'];powers[:]=16;power=16;obj=external
                failed_audits_since_refresh+=1
                # A bank change does not invalidate a same-bank BFGS secant,
                # but persistent audit failures can expose a poor local metric.
                # Refresh only after repeated failures and within an explicit
                # cap; the existing CPU budget and acceptance gates still apply.
                if (step_method=='trust_bfgs' and failed_audits_since_refresh>=2
                        and curvature_refreshes<curvature_refresh_limit):
                    hessian=None
                    curvature_refreshes+=1;failed_audits_since_refresh=0
                    row['curvature_refresh']=dict(
                        reason='repeated_independent_audit_failure',
                        count=curvature_refreshes,limit=curvature_refresh_limit,
                        scheduled=True)
                    if callback is not None:callback(row)
            elif difficult and (actual<ofv_tolerance or abs(gap)>ofv_tolerance):
                if reuse_active:
                    reuse_active=False;reuse_report['refresh_reason']='local_integration_uncertainty'
                powers[difficult]=np.minimum(16,powers[difficult]+1)
                retained=obj.posterior_pilot() if transport_posterior and len(eta_indices) else None
                power=int(powers.max());obj=prepare(x,power,seed+800001+stage,pilot=retained)
    except (TimeoutError,ValueError,RuntimeError,ArithmeticError) as error:
        message=f'{type(error).__name__}: {error}'
    finally:
        work.close()
    reuse_report['prediction_observations']=sum(o.prediction_observations for o in
        (shared_working,shared_validation) if o is not None)
    reuse_report['density_observation_terms']=sum(o.density_terms for o in
        (shared_working,shared_validation) if o is not None)
    return RefinementResult(x,value,status,message,stages,audit,time.process_time()-start,posterior,candidate_workspace,reuse_report,initial_audit)
