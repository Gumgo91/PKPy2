"""Non-centered integration coordinates for the same marginal PK likelihood.

Freeze z, not physical log parameters: log(p_ind)=log(p_pop)+sqrt(omega)*z.
Analytic PK sensitivities propagate through the population decoder.
The transformed proposal includes the change-of-variables Jacobian.
"""
from dataclasses import replace
import numpy as np
from numba import njit
from .importance_objective import PhysicalImportance
from .packed_sensitivities import event_predictions_gradient_into


@njit(cache=True,nogil=True)
def path_score(z,logq,p,indices,omega,code,t,y,dt,amounts,sp,sa):
    """Path derivatives with compiled analytic PK prediction sensitivities."""
    count,q=z.shape;d=len(p)
    predictions=np.empty((count,len(t)))
    logw=np.empty(count);scores=np.zeros((count,d+q+2))
    individual=np.empty(d);prediction_jac=np.empty((len(t),d))
    coefficients=np.empty((4,d));sqrt_omega=np.sqrt(omega)
    for k in range(count):
        individual[:]=p
        for j in range(q):individual[indices[j]]*=np.exp(sqrt_omega[j]*z[k,j])
        f=predictions[k]
        event_predictions_gradient_into(code,individual,t,dt,amounts,f,prediction_jac,coefficients)
        nll=0.
        for j in range(q):nll+=.5*(z[k,j]*z[k,j]+np.log(2*np.pi))
        for r in range(len(t)):
            pred=max(f[r],1e-10);raw=(sp*pred)**2+sa*sa;v=max(raw,1e-12)
            residual=pred-y[r];factor=1/v-residual*residual/(v*v)
            nll+=.5*(np.log(2*np.pi*v)+residual*residual/v)
            df=0.
            if f[r]>1e-10:
                df=2*residual/v
                if raw>1e-12:df+=2*sp*sp*pred*factor
            for j in range(d):scores[k,j]+=df*prediction_jac[r,j]
            if raw>1e-12:
                scores[k,-2]+=2*sp*pred*pred*factor
                scores[k,-1]+=2*sa*factor
        for j in range(d):
            derivative=scores[k,j]
            for a in range(q):
                if indices[a]==j:
                    scores[k,j]*=individual[j]/p[j]
                    scores[k,d+a]=derivative*individual[j]*z[k,a]/(2*sqrt_omega[a])
        logw[k]=-nll-logq[k]
    peak=np.max(logw);w=np.exp(logw-peak);total=w.sum();w/=total
    return -2*(peak+np.log(total)-np.log(count)),w@scores,1/np.sum(w*w),w,scores,predictions


class StandardizedImportance(PhysicalImportance):
    def __init__(self,study,eta_indices,typical,omega,sp,sa,**kwargs):
        super().__init__(study,eta_indices,typical,omega,sp,sa,**kwargs)
        self.z=[p.samples/np.sqrt(omega) for p in self.boundary.proposals]
        self.logq_z=[p.log_proposal+.5*np.log(omega).sum() for p in self.boundary.proposals]
        self.coordinate_key=None

    def _synchronize(self,typical,omega):
        typical=np.asarray(typical);omega=np.asarray(omega)
        if not np.isfinite(omega).all() or np.any(omega<1e-8):
            raise ValueError('invalid ETA variances for standardized coordinates')
        key=(typical[:,self.indices].tobytes(),omega.tobytes())
        if key!=self.coordinate_key:
            self.phi=[np.log(typical[i,self.indices])+np.sqrt(omega)*z
                      for i,z in enumerate(self.z)]
            # q_phi(phi)=q_z(z)/prod(sqrt(omega)). The same Jacobian
            # appears in the Gaussian prior and cancels in their ratio.
            self.boundary.proposals=tuple(replace(p,log_proposal=lq-.5*np.log(omega).sum())
                for p,lq in zip(self.boundary.proposals,self.logq_z))
            self.keys=[None]*len(self.keys)
            self.coordinate_key=key

    def evaluate(self,typical,omega,sp,sa):
        self._synchronize(typical,omega)
        return super().evaluate(typical,omega,sp,sa)

    def value_gradient(self,x,decode,penalty=None,bounds=None):
        penalty=penalty or (lambda point:0.)
        p,om,sp,sa=decode(x);p=np.asarray(p);om=np.asarray(om)
        if not np.isfinite(om).all() or np.any(om<1e-8):
            raise ValueError('invalid ETA variances for standardized coordinates')
        # Differentiate each physical PK parameter once per particle, then
        # propagate through any decoder, including covariates and hint terms.
        value=0.;gp=np.zeros_like(p);go=np.zeros(len(om));gs=np.zeros(2)
        self.last_subject_results=[];self.last_weights=[];ess=[]
        self._synchronize(p,om)
        for i,args in enumerate(self.boundary._arguments(p,om,sp,sa)):
            code,pp,indices,oo,t,y,dt,amounts,ss,aa=args
            v,g,e,w,_,predictions=path_score(self.z[i],self.logq_z[i],pp,indices,oo,code,t,y,dt,amounts,ss,aa)
            # Publish only a completed subject calculation, with the same
            # non-ETA key used by the physical density evaluator.
            self.cache[i]=predictions;self.keys[i]=pp[self.non_eta].tobytes()
            d=len(pp);value+=v;gp[i]=g[:d];go+=g[d:d+len(om)];gs+=g[-2:]
            self.last_weights.append(w);ess.append(e)
            self.last_subject_results.append(dict(ofv=float(v),ess=float(e),
                scaled_score=np.r_[g[:d]*pp,g[d:d+len(om)]*om,g[-2:]*[sp,sa]]))
            self.prediction_observations+=len(t)*len(self.z[i])
        self.last_min_ess=float(min(ess))
        gradient=np.zeros(len(x))
        for j in range(len(x)):
            h=1e-5*max(1.,abs(x[j]));a=x.copy();b=x.copy();a[j]+=h;b[j]-=h
            if bounds is not None:
                lo,hi=bounds[j]
                if lo is not None:b[j]=max(b[j],lo)
                if hi is not None:a[j]=min(a[j],hi)
            if a[j]==b[j]:continue
            p1,o1,s1,a1=decode(a);p0,o0,s0,a0=decode(b)
            gradient[j]=(np.sum(gp*(np.asarray(p1)-np.asarray(p0)))+
                np.dot(go,np.asarray(o1)-np.asarray(o0))+np.dot(gs,[s1-s0,a1-a0])+
                penalty(a)-penalty(b))/(a[j]-b[j])
        # Integration diagnostics use the same density-score coordinates as
        # physical banks, including after a failed external audit is retained.
        # Path scores and density scores are different finite-sample estimators
        # and must not be compared componentwise as an integration disagreement.
        self.evaluate(p,om,sp,sa)
        return value+float(penalty(x)),gradient
