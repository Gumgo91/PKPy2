"""Complex-step derivatives of the real Gaussian PK objective.

No subtraction of nearly equal likelihoods is used for the gradient. Branches
follow the real-valued prediction/variance floors. Hessians differentiate these
gradients with symmetric differences and Richardson extrapolation.
"""
import numpy as np
from numba import njit
from .packed_kernels import joint_nll, event_predictions


@njit(cache=True, nogil=True)
def _complex_nll(eta,code,typical,indices,omega,times,observed,dt,amounts,sp,sa):
    p=typical.astype(np.complex128)
    total=0.+0.j
    for j in range(len(eta)):
        p[indices[j]]*=np.exp(eta[j])
        var=max(omega[j],1e-8)
        total+=.5*(np.log(2.*np.pi*var)+eta[j]*eta[j]/var)
    k=p[0]/p[1]
    alpha=0.+0.j;beta=0.+0.j;aa=0.+0.j;bb=0.+0.j
    if code==2 or code==3:
        k12=p[2]/p[1];k21=p[2]/p[3];summ=k+k12+k21
        disc=summ*summ-4.*k*k21
        if disc.real<1e-12:disc=1e-12+0.j
        root=np.sqrt(disc);alpha=.5*(summ+root);beta=.5*(summ-root)
        aa=(alpha-k21)/(p[1]*(alpha-beta));bb=(k21-beta)/(p[1]*(alpha-beta))
    for i in range(len(times)):
        pred=0.+0.j
        for j in range(len(dt)):
            elapsed=times[i]-dt[j]
            if elapsed<0.:continue
            if code==0:
                pred+=amounts[j]/p[1]*np.exp(-k*elapsed)
            elif code==1:
                t=elapsed-p[3]
                if t.real<=0.:continue
                d=p[2]-k
                z=d*t
                if abs(z)<1e-4:
                    response=np.exp(-k*t)*t*(1.-z/2.+z*z/6.-z*z*z/24.+z*z*z*z/120.)
                elif d.real<0.:
                    response=np.exp(-p[2]*t)*np.expm1(z)/d
                else:
                    response=np.exp(-k*t)*(-np.expm1(-z)/d)
                pred+=amounts[j]*p[2]/p[1]*response
            elif code==3:
                t=elapsed-p[5]
                if t.real<=0.:continue
                for rate,weight in ((alpha,aa),(beta,bb)):
                    d=p[4]-rate;z=d*t
                    if abs(z)<1e-4:
                        response=np.exp(-rate*t)*t*(1.-z/2.+z*z/6.-z*z*z/24.+z*z*z*z/120.)
                    elif d.real<0.:
                        response=np.exp(-p[4]*t)*np.expm1(z)/d
                    else:
                        response=np.exp(-rate*t)*(-np.expm1(-z)/d)
                    pred+=amounts[j]*p[4]*weight*response
            else:
                pred+=amounts[j]*(aa*np.exp(-alpha*elapsed)+bb*np.exp(-beta*elapsed))
        if pred.real<1e-10:pred=1e-10+0.j
        var=(sp*pred)**2+sa*sa
        if var.real<1e-12:var=1e-12+0.j
        residual=observed[i]-pred
        total+=.5*(np.log(2.*np.pi*var)+residual*residual/var)
    return total


@njit(cache=True, nogil=True)
def precise_gradient(eta,code,typical,indices,omega,times,observed,dt,amounts,sp,sa):
    point=eta.astype(np.complex128)
    g=np.empty(len(eta))
    for j in range(len(eta)):
        point[j]+=1e-20j
        g[j]=_complex_nll(point,code,typical,indices,omega,times,observed,dt,amounts,sp,sa).imag/1e-20
        point[j]=eta[j]+0.j
    return g


@njit(cache=True, nogil=True)
def precise_hessian(eta,code,typical,indices,omega,times,observed,dt,amounts,sp,sa):
    result=np.empty((len(eta),len(eta)))
    point=eta.copy()
    for j in range(len(eta)):
        h=1e-3*max(1.,abs(eta[j]))
        point[j]=eta[j]+h
        plus=precise_gradient(point,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
        point[j]=eta[j]-h
        minus=precise_gradient(point,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
        coarse=(plus-minus)/(2.*h)
        point[j]=eta[j]+.5*h
        plus=precise_gradient(point,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
        point[j]=eta[j]-.5*h
        minus=precise_gradient(point,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
        result[:,j]=(4.*(plus-minus)/h-coarse)/3.
        point[j]=eta[j]
    return .5*(result+result.T)


@njit(cache=True,nogil=True)
def expected_information(eta,code,typical,indices,omega,times,observed,dt,amounts,sp,sa):
    """Expected Gaussian score information plus Gaussian ETA prior precision.

Three-point normal quadrature exactly integrates the quadratic observation
score's outer product. This is a different curvature approximation from observed
Laplace, not a floor applied to the observed eigenvalues.
"""
    p=typical.copy()
    for j in range(len(eta)):p[indices[j]]*=np.exp(eta[j])
    prediction=event_predictions(code,p,times,dt,amounts)
    matrix=np.diag(1./np.maximum(omega,1e-8))
    prior_score=eta/np.maximum(omega,1e-8)
    for i in range(len(times)):
        mean=max(prediction[i],1e-10)
        var=max((sp*mean)**2+sa*sa,1e-12)
        one_time=times[i:i+1]
        for z,weight in ((-1.7320508075688772,1./6.),(0.,2./3.),(1.7320508075688772,1./6.)):
            y=np.array([mean+np.sqrt(var)*z])
            score=precise_gradient(eta,code,typical,indices,omega,one_time,y,dt,amounts,sp,sa)-prior_score
            matrix+=weight*np.outer(score,score)
    return .5*(matrix+matrix.T)


@njit(cache=True, nogil=True)
def refine_mode(eta,code,typical,indices,omega,times,observed,dt,amounts,sp,sa,
                information_metric=False):
    """Require small Newton displacement as well as a small score.

Eigenvalue protection affects the search direction only, never the returned
Laplace determinant. Roundoff-level objective ties require a smaller score.
"""
    x=eta.copy()
    value=joint_nll(x,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
    gradient=precise_gradient(x,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
    for iteration in range(40):
        hessian=precise_hessian(x,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
        eig,vec=np.linalg.eigh(hessian)
        # Absolute scores have units and may stall above a fixed threshold even
        # when the remaining Newton decrease is below floating-point resolution.
        # Require both a localized mode and an unresolvable objective decrease.
        if information_metric and eig[0]>0.:
            observed_step=vec@((vec.T@gradient)/eig)
            decrement=np.dot(gradient,observed_step)
            if (np.max(np.abs(observed_step))<=1e-9 and decrement>=0.
                    and decrement<=np.finfo(np.float64).eps*max(1.,abs(value))):
                information=expected_information(x,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
                return x,information,0,iteration,np.max(np.abs(gradient)),np.max(np.abs(observed_step))
        if information_metric and eig[0]>0. and np.max(np.abs(gradient))<=1e-9:
            information=expected_information(x,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
            correction=np.linalg.solve(information,gradient)
            if np.max(np.abs(correction))<=1e-9:
                return x,information,0,iteration,np.max(np.abs(gradient)),np.max(np.abs(correction))
        if not information_metric and eig[0]>0.:
            correction=vec@((vec.T@gradient)/eig)
            if np.max(np.abs(gradient))<=1e-8 and np.max(np.abs(correction))<=1e-8:
                return x,hessian,0,iteration,np.max(np.abs(gradient)),np.max(np.abs(correction))
        safe=np.maximum(eig,max(1e-6,np.max(np.abs(eig))*1e-8))
        if eig[0]<0. and np.max(np.abs(gradient))<1e-6:
            # A zero score at a saddle is not a mode. Follow negative curvature
            # deterministically in either direction instead of declaring success.
            escaped=False
            for sign in (-1.,1.):
                for j in range(20):
                    trial=x+sign*(.5**(j+1))*vec[:,0]
                    candidate=joint_nll(trial,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
                    if np.isfinite(candidate) and candidate<value-32.*np.finfo(np.float64).eps*max(1.,abs(value)):
                        x=trial;value=candidate
                        gradient=precise_gradient(x,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
                        escaped=True
                        break
                if escaped:break
            if escaped:continue
        direction=-(vec@((vec.T@gradient)/safe))
        direction/=max(1.,np.max(np.abs(direction))/.5)
        accepted=False
        for j in range(30):
            trial=x+(0.5**j)*direction
            score=precise_gradient(trial,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
            candidate=joint_nll(trial,code,typical,indices,omega,times,observed,dt,amounts,sp,sa)
            slack=32.*np.finfo(np.float64).eps*max(1.,abs(value))
            if np.isfinite(candidate) and np.isfinite(score).all() and (
                candidate<value-slack or (candidate<=value+slack and np.linalg.norm(score)<np.linalg.norm(gradient))):
                x=trial;value=candidate;gradient=score;accepted=True
                break
        if not accepted:
            return x,hessian,1,iteration,np.max(np.abs(gradient)),np.inf
    return x,precise_hessian(x,code,typical,indices,omega,times,observed,dt,amounts,sp,sa),2,40,np.max(np.abs(gradient)),np.inf
