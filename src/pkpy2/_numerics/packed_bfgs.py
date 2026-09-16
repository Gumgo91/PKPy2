"""Experimental compiled inner BFGS, explicitly selectable in packed fitting.

Status 0: gradient convergence, 1: iteration limit, 2: line search failed,
3: nonfinite initial objective/gradient. Armijo backtracking and positive
curvature updates; no small-objective-change shortcut to success.
"""
import numpy as np
from numba import njit
from .packed_kernels import joint_value_gradient


@njit(cache=True, nogil=True)
def solve_bfgs(eta0, code, typical, eta_indices, omega, times, observed,
               dose_times, amounts, sp, sa, max_iter=80, gtol=1e-5):
    x=eta0.copy()
    n=len(x)
    inverse=np.eye(n)
    value,gradient=joint_value_gradient(x,code,typical,eta_indices,omega,times,
                                       observed,dose_times,amounts,sp,sa)
    calls=1
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return x,value,gradient,3,0,calls
    if n==0:
        return x,value,gradient,0,0,calls
    for iteration in range(max_iter):
        if np.max(np.abs(gradient))<=gtol:
            return x,value,gradient,0,iteration,calls
        direction=-(inverse@gradient)
        slope=np.dot(direction,gradient)
        if not np.isfinite(slope) or slope>=0.:
            inverse=np.eye(n)
            direction=-gradient
            slope=-np.dot(gradient,gradient)
        # Bounded initial displacement avoids enormous exponent trials; it
        # limits a line-search step, not the admissible ETA parameter domain.
        step=min(1.,1./max(1.,np.max(np.abs(direction))))
        accepted=False
        trial=x.copy();new_value=value;new_gradient=gradient.copy()
        for search in range(40):
            trial=x+step*direction
            if np.all(trial == x):
                # Further halving cannot produce a representable displacement.
                # Report failure for audited recovery; never label stagnation
                # as gradient convergence or spend the remaining budget on x.
                return x,value,gradient,2,iteration,calls
            new_value,new_gradient=joint_value_gradient(trial,code,typical,eta_indices,
                omega,times,observed,dose_times,amounts,sp,sa)
            calls+=1
            if (np.isfinite(new_value) and np.isfinite(new_gradient).all()
                    and new_value<=value+1e-4*step*slope):
                accepted=True
                break
            step*=.5
        if not accepted:
            return x,value,gradient,2,iteration,calls
        displacement=trial-x
        change=new_gradient-gradient
        curvature=np.dot(change,displacement)
        if curvature>1e-12*np.linalg.norm(change)*np.linalg.norm(displacement):
            rho=1./curvature
            left=np.eye(n)-rho*np.outer(displacement,change)
            inverse=left@inverse@left.T+rho*np.outer(displacement,displacement)
        else:
            inverse=np.eye(n)
        x=trial;value=new_value;gradient=new_gradient
    status=0 if np.max(np.abs(gradient))<=gtol else 1
    return x,value,gradient,status,max_iter,calls
