"""Fixed physical-parameter importance objective with cached predictions.

Population means/variances change the prior density, not the sampled physical
parameters. Analytic density scores avoid repeated finite-difference PK solves.
Proposal preparation and independent validation belong outside optimization.
"""
import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from numba import njit
from .packed_importance import StudyImportance
from .packed_kernels import event_predictions
from .subject_work import ordered_map


@dataclass(frozen=True)
class PhysicalSnapshot:
    study: object
    indices: tuple
    phi: tuple
    logq: tuple
    prediction: tuple
    keys: tuple
    components: tuple

    def __post_init__(self):
        for a in (*self.phi,*self.logq,*self.prediction):a.flags.writeable=False

    def compatible(self,study,indices):
        if tuple(indices)!=self.indices or study.model!=self.study.model or study.subject_ids!=self.study.subject_ids:
            return False
        return all(np.array_equal(getattr(study,k),getattr(self.study,k)) for k in
            ('observation_offsets','time','observation','dose_offsets','dose_time','dose_amount','dose_duration'))


@njit(cache=True,nogil=True)
def predictions(phi,p,indices,code,times,dt,amounts):
    out=np.empty((len(phi),len(times)))
    for k in range(len(phi)):
        individual=p.copy()
        for j in range(len(indices)):individual[indices[j]]=np.exp(phi[k,j])
        out[k]=event_predictions(code,individual,times,dt,amounts)
    return out


@njit(cache=True,nogil=True)
def density_score(phi,logq,f,y,mu,omega,sp,sa):
    count,q=phi.shape
    logw=np.empty(count);scores=np.zeros((count,2*q+2))
    for k in range(count):
        value=0.
        for j in range(q):
            eta=phi[k,j]-mu[j]
            value+=.5*(np.log(2*np.pi*omega[j])+eta*eta/omega[j])
            scores[k,j]=-2*eta/omega[j]
            scores[k,q+j]=1/omega[j]-eta*eta/(omega[j]*omega[j])
        for j in range(len(y)):
            pred=max(f[k,j],1e-10);raw=(sp*pred)**2+sa*sa;v=max(raw,1e-12)
            r2=(y[j]-pred)**2
            value+=.5*(np.log(2*np.pi*v)+r2/v)
            if raw>1e-12:
                factor=1/v-r2/(v*v)
                scores[k,-2]+=2*sp*pred*pred*factor
                scores[k,-1]+=2*sa*factor
        logw[k]=-value-logq[k]
    peak=np.max(logw);w=np.exp(logw-peak);total=w.sum();w/=total
    return -2*(peak+np.log(total)-np.log(count)),w@scores,1/np.sum(w*w),w,scores


@njit(cache=True,nogil=True)
def observation_density(f,y,sp,sa):
    """Log likelihood and -2-log-likelihood scores, independent of mu/omega."""
    loglike=np.zeros(len(f));score=np.zeros((len(f),2))
    for k in range(len(f)):
        for j in range(len(y)):
            pred=max(f[k,j],1e-10);raw=(sp*pred)**2+sa*sa;v=max(raw,1e-12)
            r2=(y[j]-pred)**2
            loglike[k]-=.5*(np.log(2*np.pi*v)+r2/v)
            if raw>1e-12:
                factor=1/v-r2/(v*v)
                score[k,0]+=2*sp*pred*pred*factor;score[k,1]+=2*sa*factor
    return loglike,score


@njit(cache=True,nogil=True)
def reweight_density(phi,logq,loglike,residual_score,mu,omega):
    count,q=phi.shape;logw=loglike-logq;scores=np.zeros((count,2*q+2))
    for k in range(count):
        for j in range(q):
            eta=phi[k,j]-mu[j]
            logw[k]-=.5*(np.log(2*np.pi*omega[j])+eta*eta/omega[j])
            scores[k,j]=-2*eta/omega[j]
            scores[k,q+j]=1/omega[j]-eta*eta/(omega[j]*omega[j])
        scores[k,-2:]=residual_score[k]
    peak=np.max(logw);w=np.exp(logw-peak);total=w.sum();w/=total
    return -2*(peak+np.log(total)-np.log(count)),w@scores,1/np.sum(w*w),w,scores


class PhysicalImportance:
    @classmethod
    def from_snapshot(cls,snapshot,study,indices,work=None):
        if not snapshot.compatible(study,indices):
            raise ValueError('physical snapshot model, data or ETA coordinates differ')
        obj=cls.__new__(cls)
        obj.work=work;obj.study=study;obj.boundary=StudyImportance(study,indices)
        obj.indices=obj.boundary.indices
        obj.boundary.proposals=tuple(SimpleNamespace(log_proposal=a,components=c)
                                    for a,c in zip(snapshot.logq,snapshot.components))
        obj.phi=list(snapshot.phi)
        # Parameter count follows the built-in model, including non-ETA ALAG.
        count=2 if obj.boundary.code==0 else 6 if obj.boundary.code==3 else 4
        obj.non_eta=[j for j in range(count) if j not in obj.indices]
        obj.cache=list(snapshot.prediction);obj.keys=list(snapshot.keys)
        obj.prediction_observations=0;obj.density_terms=0
        obj.last_min_ess=None;obj.last_subject_results=[];obj.last_weights=[]
        obj.observation_cache=[None]*len(obj.phi);obj.observation_keys=[None]*len(obj.phi)
        obj.observation_likelihood_terms=0;obj.prior_density_terms=0
        return obj

    def __init__(self,study,eta_indices,typical,omega,sp,sa,*,power=12,seed=0,pilot_power=14,sample_powers=None,posterior_pilot=None,work=None):
        typical=np.asarray(typical,dtype=float)
        self.work=work
        if np.any(np.asarray(omega)<1e-8):raise ValueError('ETA variances below kernel floor unsupported')
        self.boundary=StudyImportance(study,eta_indices).prepare(typical,omega,sp,sa,
            power=power,seed=seed,pilot_power=pilot_power,sample_powers=sample_powers,posterior_pilot=posterior_pilot,work=work)
        self.indices=self.boundary.indices;self.study=study
        self.phi=[p.samples+np.log(typical[i,self.indices]) for i,p in enumerate(self.boundary.proposals)]
        self.non_eta=[j for j in range(typical.shape[1]) if j not in self.indices]
        self.cache=[None]*len(self.phi);self.keys=[None]*len(self.phi)
        self.prediction_observations=0;self.density_terms=0
        self.last_min_ess=None
        self.last_subject_results=[]
        self.last_weights=[]
        self.observation_cache=[None]*len(self.phi);self.observation_keys=[None]*len(self.phi)
        self.observation_likelihood_terms=0;self.prior_density_terms=0

    def posterior_pilot(self):
        """Weighted moments in physical log coordinates, never ETA coordinates."""
        result=[]
        for phi,w in zip(self.phi,self.last_weights):
            mean=w@phi
            centered=phi-mean
            result.append((mean,(centered.T*w)@centered))
        return result

    def evaluate(self,typical,omega,sp,sa):
        if np.any(np.asarray(omega)<1e-8):raise ValueError('ETA variances below kernel floor unsupported')
        rows=list(self.boundary._arguments(typical,omega,sp,sa))
        q=len(self.indices);value=0.;gm=np.empty((len(rows),q));go=np.zeros(q);gs=np.zeros(2);ess=[]
        self.last_subject_results=[]
        self.last_weights=[]
        def evaluate_subject(item):
            i,args=item
            code,p,indices,om,t,y,dt,a,sp,sa=args
            key=p[self.non_eta].tobytes()
            prediction_count=0
            if self.cache[i] is None or self.keys[i]!=key:
                self.cache[i]=predictions(self.phi[i],p,indices,code,t,dt,a)
                self.keys[i]=key;prediction_count=len(t)*len(self.phi[i])
            # Response-coordinate evaluators may replace the predictions even
            # when non-ETA parameters have not changed.
            observation_key=(id(self.cache[i]),key,float(sp),float(sa))
            observation_count=0
            if self.observation_keys[i]!=observation_key:
                self.observation_cache[i]=observation_density(self.cache[i],y,sp,sa)
                self.observation_keys[i]=observation_key
                observation_count=len(t)*len(self.phi[i])
            v,g,e,w,_=reweight_density(self.phi[i],self.boundary.proposals[i].log_proposal,
                *self.observation_cache[i],np.log(p[indices]),om)
            return v,g,e,w,prediction_count,len(t)*len(self.phi[i]),observation_count
        for i,(v,g,e,w,prediction_count,density_count,observation_count) in enumerate(
                ordered_map(self.work,evaluate_subject,enumerate(rows))):
            self.prediction_observations+=prediction_count
            self.observation_likelihood_terms+=observation_count
            self.prior_density_terms+=len(self.phi[i])*q
            self.last_weights.append(w)
            self.density_terms+=density_count;value+=v
            gm[i]=g[:q];go+=g[q:2*q];gs+=g[-2:];ess.append(e)
            self.last_subject_results.append(dict(ofv=float(v),ess=float(e),
                scaled_score=np.r_[g[:q]*np.sqrt(omega),g[q:2*q]*omega,g[-2:]*[sp,sa]]))
        self.last_min_ess=float(min(ess))
        return value,gm,go,gs,self.last_min_ess

    def value_gradient(self,x,decode,penalty=None,bounds=None,analytic_non_eta=False):
        penalty=penalty or (lambda x:0.)
        p,om,sp,sa=decode(x)
        p=np.asarray(p)
        value,gm,go,gs,ess=self.evaluate(p,om,sp,sa)
        baseline_records=self.last_subject_results
        baseline_weights=self.last_weights
        gradient=np.empty(len(x));non_eta_score=None
        for j in range(len(x)):
            h=1e-5*max(1.,abs(x[j]));plus=x.copy();minus=x.copy();plus[j]+=h;minus[j]-=h
            if bounds is not None:
                lower,upper=bounds[j]
                if lower is not None:minus[j]=max(minus[j],lower)
                if upper is not None:plus[j]=min(plus[j],upper)
            span=plus[j]-minus[j]
            if span==0:gradient[j]=0.;continue
            p1,o1,s1,a1=decode(plus);p0,o0,s0,a0=decode(minus)
            p1=np.asarray(p1);p0=np.asarray(p0);o1=np.asarray(o1);o0=np.asarray(o0)
            if (not np.array_equal(p1[:,self.non_eta],p[:,self.non_eta]) or
                    not np.array_equal(p0[:,self.non_eta],p[:,self.non_eta])):
                if analytic_non_eta:
                    if non_eta_score is None:
                        # Fixed physical particles: only the non-ETA likelihood
                        # path contributes here; ETA means use density scores.
                        from .standardized_importance import path_score
                        def score_subject(item):
                            i,args=item
                            code,pp,indices,oo,t,y,dt,amounts,ss,aa=args
                            eta=self.phi[i]-np.log(pp[indices])
                            output=path_score(eta/np.sqrt(oo),
                                self.boundary.proposals[i].log_proposal+.5*np.log(oo).sum(),
                                pp,indices,oo,code,t,y,dt,amounts,ss,aa)
                            return baseline_weights[i]@output[4][:,self.non_eta]
                        non_eta_score=np.asarray(ordered_map(self.work,score_subject,
                            enumerate(self.boundary._arguments(p,om,sp,sa))))
                        self.prediction_observations+=sum(len(a[4])*len(self.phi[i])
                            for i,a in enumerate(self.boundary._arguments(p,om,sp,sa)))
                    gradient[j]=(np.sum(gm*(np.log(p1[:,self.indices])-np.log(p0[:,self.indices])))+
                        np.dot(go,o1-o0)+np.dot(gs,[s1-s0,a1-a0])+
                        np.sum(non_eta_score*(p1[:,self.non_eta]-p0[:,self.non_eta]))+
                        penalty(plus)-penalty(minus))/span
                else:
                    gradient[j]=(self.evaluate(p1,o1,s1,a1)[0]-self.evaluate(p0,o0,s0,a0)[0]+penalty(plus)-penalty(minus))/span
            else:
                gradient[j]=(np.sum(gm*(np.log(p1[:,self.indices])-np.log(p0[:,self.indices])))+
                    np.dot(go,o1-o0)+np.dot(gs,[s1-s0,a1-a0])+penalty(plus)-penalty(minus))/span
        self.last_subject_results=baseline_records;self.last_min_ess=ess
        self.last_weights=baseline_weights
        return value+float(penalty(x)),gradient
