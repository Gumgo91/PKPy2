"""Laplace exploration, audited initial refinement, optional SAEM, and marginal correction.

engine.run_fit and clinical_auto use this path by default.
Decode and penalty callbacks carry model/covariate/hint constraints consistently
through both phases. A phase limit is not reported as final convergence.
"""
from dataclasses import dataclass
import time
import numpy as np
from .saem import fit_saem,SAEMResult
from .importance_refinement import refine_importance,RefinementResult
from .laplace_exploration import explore_laplace


@dataclass
class MarginalFitResult:
    x: np.ndarray
    ofv: float
    status: str
    saem: SAEMResult
    refinement: RefinementResult
    cpu_seconds: float
    exploration: dict | None = None


def fit_marginal(study,eta_indices,x0,decode,*,penalty=None,bounds=None,seed=0,
                 saem_options=None,refinement_options=None,laplace_options=None,callback=None,workers=1,reuse_workspace=None,
                 estimation_schedule='laplace_refine_saem_refine'):
    """Explore with the Laplace objective, then use a short audited probe before SAEM.

    The default schedule first minimizes the deterministic Laplace objective
    (60 CPU seconds by default) only to relocate the starting point; the final
    estimate and status still come from importance refinement and its audit.

    Defaults allow 60 CPU seconds for SAEM and 300 for marginal refinement.
    Initial JIT compilation and orchestration are included in returned total CPU
    time even where an individual phase reports warmup separately. Final status
    comes from independent marginal-score/likelihood checks, not SAEM stability.
    The legacy refine_saem_refine schedule and explicit workspace reuse enable
    a preliminary probe within 20 CPU seconds (or 20% of the refinement budget).
    Its cost is charged to that budget; accepted progress is retained for SAEM.
    """
    start=time.process_time()
    if estimation_schedule not in ('saem_then_audit','refine_saem_refine','laplace_refine_saem_refine'):
        raise ValueError('unknown estimation schedule')
    if reuse_workspace is not None:
        import hashlib
        lineage=repr((seed,tuple(reuse_workspace.source_seeds))).encode()
        seed=(1<<128)+int.from_bytes(hashlib.sha256(lineage).digest()[:12],'big')
    initial=dict(exploration=400,smoothing=400,cpu_budget_seconds=60.,audit_reserve_seconds=15.,
                 audit_likelihood=False,mu_regression=True,adaptive_phases=True,block_mstep=False)
    initial.update(saem_options or {})
    final=dict(cpu_budget_seconds=300.)
    final.update(refinement_options or {})
    final.setdefault('workers',workers)
    final.setdefault('step_method','lbfgs')
    final.setdefault('audit_first',estimation_schedule=='saem_then_audit')
    if any(k in initial or k in final for k in ('seed','callback','penalty','bounds')):
        raise ValueError('set shared policy arguments at the fit_marginal level')
    def emit(phase,row):
        if callback is not None:callback(dict(fit_phase=phase,**row))
    exploration=None
    if estimation_schedule=='laplace_refine_saem_refine':
        exploration_options=dict(cpu_budget_seconds=120.,max_iterations=200,starts=8)
        exploration_options.update(laplace_options or {})
        if len(eta_indices):
            explored=explore_laplace(study,eta_indices,x0,decode,penalty=penalty,bounds=bounds,workers=workers,
                                     seed=seed+30000049,callback=lambda row:emit('laplace_exploration',row),
                                     **exploration_options)
            exploration=explored.record()
            x0=explored.x.copy()
        else:
            exploration=dict(status='not_applicable',message='no random effects')
        estimation_schedule='refine_saem_refine'
    # Legacy scheduling or explicit sample reuse permits a preliminary probe.
    # This is model-independent: neither a drug label nor a claimed warm start
    # is trusted. Only the normal independent stationarity gate can skip SAEM.
    probe=None
    probe_record=dict(status='not_run',reason='SAEM estimation precedes high-precision audit')
    if len(eta_indices) and (estimation_schedule=='refine_saem_refine' or reuse_workspace is not None):
        probe_options=dict(final)
        probe_options['cpu_budget_seconds']=min(20.,float(final['cpu_budget_seconds'])*.2)
        # A trust-quadratic stage is not equivalent to an L-BFGS stage. Do not
        # force a costly population restart after an arbitrary three updates.
        # Physical banks reuse predictions and avoid transport/Hessian setup.
        probe_options.update(integration_coordinates='physical',step_method='lbfgs',
                             validation_reference='candidate',audit_first=False)
        probe=refine_importance(study,eta_indices,x0,decode,penalty=penalty,bounds=bounds,
            seed=seed+20000033,callback=lambda row:emit('initial_refinement',row),reuse_workspace=reuse_workspace,**probe_options)
        probe_record=dict(status=probe.status,message=probe.message,cpu_seconds=probe.cpu_seconds,
                          stages=probe.stages,audit=probe.audit,sample_reuse=getattr(probe,'sample_reuse',None))
        if probe.status=='converged':
            first=SAEMResult(probe.x.copy(),'not_needed',
                'supplied estimate refined to independent marginal stationarity',[],
                {'iterations':0,'initial_refinement':probe_record,'estimation_schedule':estimation_schedule}, {})
            return MarginalFitResult(probe.x,probe.ofv,probe.status,first,probe,time.process_time()-start,exploration)
        # Charge the probe to the existing refinement budget, not a hidden extra.
        final['cpu_budget_seconds']=max(.001,float(final['cpu_budget_seconds'])-probe.cpu_seconds)
    if len(eta_indices):
        retained=(probe is not None and any(r.get('accepted',False) for r in probe.stages)
                  and np.isfinite(probe.x).all() and np.isfinite(probe.ofv))
        saem_start=probe.x.copy() if retained else np.asarray(x0,dtype=float).copy()
        first=fit_saem(study,eta_indices,saem_start,decode,penalty=penalty,bounds=bounds,seed=seed,
                       callback=lambda row:emit('saem',row),**initial)
        first.diagnostics['initial_refinement']=probe_record
        first.diagnostics['initialization_source']='accepted_refinement' if retained else 'original'
    else:
        first=SAEMResult(np.asarray(x0,dtype=float).copy(),'not_applicable',
                        'no random effects to integrate',[],{}, {})
    first.diagnostics['estimation_schedule']=estimation_schedule
    second=refine_importance(study,eta_indices,first.x,decode,penalty=penalty,bounds=bounds,
                   seed=seed+10000019,callback=lambda row:emit('marginal_refinement',row),
                   reuse_workspace=(reuse_workspace if probe is None or not (getattr(probe,'sample_reuse',None) or {}).get('refresh_reason') else None),
                   posterior_pilot=first.posterior_pilot if final.get('transport_posterior',False) else None,**final)
    return MarginalFitResult(second.x,second.ofv,second.status,first,second,time.process_time()-start,exploration)
