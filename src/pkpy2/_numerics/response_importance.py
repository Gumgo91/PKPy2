"""Locally translate particles using posterior mean sensitivities.

The response is estimated on a separate pilot bank, frozen within a stage,
and propagated through the complete population decoder. Translation has unit
Jacobian. The actual proposal density stays attached to each translated sample.
"""
import numpy as np
from dataclasses import replace
from .importance_objective import PhysicalImportance,density_score
from .packed_importance import joint_batch
from .standardized_importance import path_score
from .subject_work import ordered_map


class TransportSupportError(ValueError):
    """A trial transport exceeds its numerically representable local region."""


def posterior_response(phi,weights,ofv_scores):
    """d E(phi)/d theta = Cov(phi, log-joint score); OFV=-2 log joint."""
    centered=phi-weights@phi
    scores=ofv_scores-weights@ofv_scores
    return -.5*(centered.T*weights)@scores


def covariance_response(phi,weights,ofv_scores):
    """Return covariance and its response in (parameter, phi, phi) order."""
    r=phi-weights@phi;s=ofv_scores-weights@ofv_scores
    covariance=(r.T*weights)@r
    derivative=-.5*np.einsum('n,na,nb,nj->jab',weights,r,r,s,optimize=True)
    return covariance,derivative


def shape_transform(chol,sensitivities,delta):
    """SPD affine transform and exact Frechet derivatives of its exponential."""
    matrix=.5*np.einsum('j,jab->ab',delta,sensitivities)
    values,vectors=np.linalg.eigh((matrix+matrix.T)*.5)
    if not np.isfinite(values).all() or np.max(np.abs(values),initial=0.)>50:
        raise TransportSupportError('shape transport outside finite local support')
    exponential=(vectors*np.exp(values))@vectors.T
    difference=values[:,None]-values[None,:]
    divided=np.ones_like(difference)
    mask=np.abs(difference)>1e-7
    divided[mask]=np.expm1(difference[mask])/difference[mask]
    divided[~mask]=1+difference[~mask]/2+difference[~mask]**2/6
    divided*=np.exp(values)[None,:]
    directions=np.swapaxes(vectors,-1,-2)@(.5*sensitivities)@vectors
    derivatives=vectors@(divided*directions)@vectors.T
    inverse=np.linalg.inv(chol)
    jacobian_gradient=.5*np.trace(sensitivities,axis1=1,axis2=2)
    return (chol@exponential@inverse,chol@derivatives@inverse,
            float(jacobian_gradient@delta),jacobian_gradient)


def decoder_differences(x,decode,bounds):
    result=[]
    for j in range(len(x)):
        h=1e-5*max(1.,abs(x[j]));a=x.copy();b=x.copy();a[j]+=h;b[j]-=h
        if bounds is not None:
            lo,hi=bounds[j]
            if lo is not None:b[j]=max(b[j],lo)
            if hi is not None:a[j]=min(a[j],hi)
        result.append((decode(a),decode(b),a[j]-b[j],a,b))
    return result


class ResponseImportance(PhysicalImportance):
    def __init__(self,study,eta_indices,typical,omega,sp,sa,*,reference_x,decode,bounds=None,
                 transport_shape=False,**kwargs):
        super().__init__(study,eta_indices,typical,omega,sp,sa,**kwargs)
        self.reference_x=np.array(reference_x,dtype=float,copy=True)
        self.reference_phi=[p.copy() for p in self.phi]
        self.response=[]
        self.transport_shape=transport_shape
        self.reference_logq=[p.log_proposal.copy() for p in self.boundary.proposals]
        self.centers=[];self.cholesky=[];self.shape_sensitivities=[]
        # Response estimation and optimization samples must be separate. Using
        # each optimization bank to fit its own transform creates dependence.
        pilot=PhysicalImportance(study,eta_indices,typical,omega,sp,sa,power=12,
            seed=int(kwargs.get('seed',0))+32452843,pilot_power=14,work=self.work)
        pilot.evaluate(typical,omega,sp,sa)
        diffs=decoder_differences(self.reference_x,decode,bounds)
        q=len(self.indices)
        for i,args in enumerate(pilot.boundary._arguments(typical,omega,sp,sa)):
            code,p,indices,om,t,y,dt,amounts,ss,aa=args
            phi=pilot.phi[i]
            _,_,_,w,scores=density_score(phi,pilot.boundary.proposals[i].log_proposal,
                pilot.cache[i],y,np.log(p[indices]),om,ss,aa)
            complete=np.zeros((len(phi),len(reference_x)))
            for j,(high,low,span,_,__) in enumerate(diffs):
                if span==0:continue
                p1,o1,s1,a1=high;p0,o0,s0,a0=low
                if (not np.array_equal(p1[i,pilot.non_eta],p[pilot.non_eta]) or
                    not np.array_equal(p0[i,pilot.non_eta],p[pilot.non_eta])):
                    v1=joint_batch(phi-np.log(p1[i,indices]),code,p1[i],indices,o1,t,y,dt,amounts,s1,a1)
                    v0=joint_batch(phi-np.log(p0[i,indices]),code,p0[i],indices,o0,t,y,dt,amounts,s0,a0)
                    complete[:,j]=2*(v1-v0)/span
                else:
                    tangent=np.r_[(np.log(p1[i,indices])-np.log(p0[i,indices])),
                                   np.asarray(o1)-np.asarray(o0),s1-s0,a1-a0]/span
                    complete[:,j]=scores@tangent
            self.response.append(posterior_response(phi,w,complete))
            if transport_shape:
                covariance,derivative=covariance_response(phi,w,complete)
                chol=np.linalg.cholesky(covariance+1e-8*np.diag(om))
                inverse=np.linalg.inv(chol)
                self.centers.append(w@phi);self.cholesky.append(chol)
                self.shape_sensitivities.append(inverse@derivative@inverse.T)
        self.prediction_observations+=pilot.prediction_observations

    def _move(self,x):
        delta=x-self.reference_x
        self.shape_derivatives=[];self.logdet_gradients=[]
        if self.transport_shape:
            self.phi=[];proposals=[]
            for i,(phi,A) in enumerate(zip(self.reference_phi,self.response)):
                b,db,logdet,dlogdet=shape_transform(self.cholesky[i],self.shape_sensitivities[i],delta)
                self.phi.append(self.centers[i]+A@delta+(phi-self.centers[i])@b.T)
                self.shape_derivatives.append(db);self.logdet_gradients.append(dlogdet)
                proposals.append(replace(self.boundary.proposals[i],log_proposal=self.reference_logq[i]-logdet))
            self.boundary.proposals=tuple(proposals)
        else:
            self.phi=[phi+A@delta for phi,A in zip(self.reference_phi,self.response)]
        self.keys=[None]*len(self.keys)

    def value_gradient(self,x,decode,penalty=None,bounds=None):
        penalty=penalty or (lambda point:0.)
        self._move(x)
        p,om,sp,sa=decode(x);p=np.asarray(p);om=np.asarray(om)
        if not np.isfinite(om).all() or np.any(om<1e-8):raise ValueError('invalid ETA variances')
        q=len(self.indices);value=0.;gradient=np.zeros(len(x))
        diffs=decoder_differences(x,decode,bounds)
        def evaluate_subject(item):
            i,args=item
            gradient=np.zeros(len(x))
            code,pp,indices,oo,t,y,dt,amounts,ss,aa=args
            eta=self.phi[i]-np.log(pp[indices])
            v,g,_,w,particle_scores,predictions=path_score(eta/np.sqrt(oo),self.boundary.proposals[i].log_proposal+
                .5*np.log(oo).sum(),pp,indices,oo,code,t,y,dt,amounts,ss,aa)
            self.cache[i]=predictions;self.keys[i]=pp[self.non_eta].tobytes()
            gm=-2*(w@eta)/oo
            go=1/oo-(w@(eta*eta))/(oo*oo)
            # g[:d] is a likelihood path score. Add the prior derivative with
            # respect to physical phi to obtain the total translation score.
            gphi=g[indices]*pp[indices]-gm
            gradient+=gphi@self.response[i]
            if self.transport_shape:
                particle_phi_score=particle_scores[:,indices]*pp[indices]+2*eta/oo
                moment=(particle_phi_score.T*w)@(self.reference_phi[i]-self.centers[i])
                gradient+=np.einsum('ab,jab->j',moment,self.shape_derivatives[i])-2*self.logdet_gradients[i]
            for j,(high,low,span,_,__) in enumerate(diffs):
                if span==0:continue
                p1,o1,s1,a1=high;p0,o0,s0,a0=low
                gradient[j]+=(np.dot(gm,np.log(p1[i,indices])-np.log(p0[i,indices]))+
                    np.dot(go,np.asarray(o1)-np.asarray(o0))+np.dot(g[-2:],[s1-s0,a1-a0])+
                    np.dot(g[self.non_eta],p1[i,self.non_eta]-p0[i,self.non_eta]))/span
            return v,gradient,len(t)*len(eta)
        for v,g,count in ordered_map(self.work,evaluate_subject,
                enumerate(self.boundary._arguments(p,om,sp,sa))):
            value+=v;gradient+=g;self.prediction_observations+=count
        for j,(_,__,span,a,b) in enumerate(diffs):
            if span:gradient[j]+=(penalty(a)-penalty(b))/span
        # Comparable density-score diagnostics and posterior moments at x.
        super().evaluate(p,om,sp,sa)
        return value+float(penalty(x)),gradient

    def value(self,x,decode,penalty=None):
        self._move(x)
        return super().evaluate(*decode(x))[0]+(0. if penalty is None else float(penalty(x)))
