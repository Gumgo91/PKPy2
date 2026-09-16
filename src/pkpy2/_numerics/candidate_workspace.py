"""Transient, fixed-physical-sample workspace for model proposals.

The cached likelihood cancels in density ratios when only population means
change. These are proposal calculations, never independent convergence audits.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from .importance_objective import density_score, predictions,PhysicalSnapshot


def stratified_prefix(total,count,components):
    """Preserve mixture proportions and a Sobol prefix within every stratum."""
    if total%components or count%components or not 0<count<=total:
        raise ValueError('invalid stratified sample counts')
    return np.concatenate([np.arange(count//components)+j*(total//components) for j in range(components)])


def complete_information(eta, prediction, y, omega, sp, sa, weights):
    """Louis ingredients for log means, log variances and log residual SDs."""
    q=len(omega); n=len(eta)
    f=np.maximum(prediction,1e-10)
    parts=np.stack(((sp*f)**2,np.full_like(f,sa*sa)),axis=-1)
    raw=parts.sum(axis=-1);v=np.maximum(raw,1e-12)
    u=parts/v[:,:,None];u[raw<=1e-12]=0.
    ratio=(y-f)**2/v
    residual=((ratio-1)[:,:,None]*u).sum(axis=1)
    scores=np.column_stack((eta/omega,.5*(eta*eta/omega-1),residual))
    mean=weights@scores;centered=scores-mean
    covariance=(centered.T*weights)@centered
    hessian=np.zeros((2*q+2,2*q+2))
    hessian[:q,:q]=np.diag(1/omega)
    cross=weights@(eta/omega)
    hessian[:q,q:2*q]=np.diag(cross)
    hessian[q:2*q,:q]=np.diag(cross)
    hessian[q:2*q,q:2*q]=np.diag(.5*(weights@(eta*eta/omega)))
    for a in range(2):
        for b in range(2):
            h=2*(2*ratio-1)*u[:,:,a]*u[:,:,b]
            if a==b:h-=2*(ratio-1)*u[:,:,a]
            hessian[2*q+a,2*q+b]=weights@h.sum(axis=1)
    return mean,hessian,covariance


@dataclass(frozen=True)
class CandidateBank:
    eta: tuple
    weights: tuple
    omega: np.ndarray
    moments: tuple
    physical: object | None = None

    def __post_init__(self):
        for array in (*self.eta,*self.weights,self.omega,
                      *(a for moment in self.moments for a in moment)):
            array.flags.writeable=False

    @classmethod
    def capture(cls,evaluator,typical,omega,sp,sa,limit=4096):
        """Keep Sobol prefixes, expanding low-ESS subjects without PK solves."""
        etas=[];weights=[];moments=[];phis=[];logq=[];prediction=[];keys=[];components=[]
        for i,args in enumerate(evaluator.boundary._arguments(typical,omega,sp,sa)):
            code,p,indices,om,t,y,dt,a,_,__=args
            count=min(limit,len(evaluator.phi[i]))
            available=evaluator.last_weights[i]
            component_count=evaluator.boundary.proposals[i].components
            while count<len(available):
                prefix=available[stratified_prefix(len(available),count,component_count)];mass=prefix.sum()
                if mass>0 and mass*mass/(prefix@prefix)>=200:break
                count=min(2*count,len(available))
            keep=stratified_prefix(len(available),count,component_count)
            phi=evaluator.phi[i][keep]
            if evaluator.keys[i]==p[evaluator.non_eta].tobytes():
                f=evaluator.cache[i][keep]
            else:
                f=predictions(phi,p,indices,code,t,dt,a)
            _,_,_,w,_=density_score(phi,evaluator.boundary.proposals[i].log_proposal[keep],
                                   f,y,np.log(p[indices]),om,sp,sa)
            eta=phi-np.log(p[indices])
            etas.append(eta.copy());weights.append(w.copy())
            moments.append(complete_information(eta,f,y,om,sp,sa,w))
            phis.append(phi);logq.append(evaluator.boundary.proposals[i].log_proposal[keep])
            prediction.append(f.copy());keys.append(p[evaluator.non_eta].tobytes())
            components.append(component_count)
        physical=PhysicalSnapshot(evaluator.study,tuple(evaluator.indices),tuple(phis),tuple(logq),tuple(prediction),tuple(keys),tuple(components))
        return cls(tuple(etas),tuple(weights),np.array(omega,copy=True),tuple(moments),physical)

    def mean_shift(self,shift):
        """Exact fixed-bank likelihood ratio; residuals and variances fixed."""
        shift=np.asarray(shift,dtype=float)
        if shift.shape!=(len(self.eta),len(self.omega)) or not np.isfinite(shift).all():
            raise ValueError('one finite mean shift per subject and ETA required')
        value=0.;minimum_ess=float('inf')
        for eta,w,delta in zip(self.eta,self.weights,shift):
            ratio=(eta@ (delta/self.omega))-.5*np.sum(delta*delta/self.omega)
            logw=np.where(w>0,np.log(np.where(w>0,w,1.)),-np.inf)+ratio
            normalizer=logsumexp(logw)
            updated=np.exp(logw-normalizer)
            value-=2*normalizer;minimum_ess=min(minimum_ess,1/(updated@updated))
        return float(value),float(minimum_ess)


@dataclass(frozen=True)
class CandidateWorkspace:
    subject_ids: tuple
    banks: tuple
    source_seeds: tuple = ()

    def __deepcopy__(self,memo):
        # Read-only consumers; copying a FitResult must not duplicate particles.
        return self

    def __reduce__(self):
        # Checkpoints may replay without a cache; large numerical state is
        # intentionally process-local, including when passed in fit kwargs.
        return type(self),(self.subject_ids,(),self.source_seeds)
