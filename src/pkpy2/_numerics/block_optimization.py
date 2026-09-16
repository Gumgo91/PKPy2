"""Block minimization of one unchanged fixed-bank marginal objective."""
import numpy as np
from scipy.optimize import minimize,OptimizeResult


def population_coordinates(x,decode,eta_indices,bounds=None):
    p,_,sp,sa=decode(x);p=np.asarray(p)
    other=[j for j in range(p.shape[1]) if j not in eta_indices];population=[];observation=[]
    for j in range(len(x)):
        h=1e-5*max(1.,abs(x[j]));hi=x.copy();lo=x.copy();hi[j]+=h;lo[j]-=h
        if bounds is not None:
            lower,upper=bounds[j]
            if lower is not None:lo[j]=max(lo[j],lower)
            if upper is not None:hi[j]=min(hi[j],upper)
        if hi[j]==lo[j]:continue
        a,_,s1,a1=decode(hi);b,_,s0,a0=decode(lo)
        pure=(s1==s0==sp and a1==a0==sa and
              np.array_equal(np.asarray(a)[:,other],p[:,other]) and
              np.array_equal(np.asarray(b)[:,other],p[:,other]))
        (population if pure else observation).append(j)
    return population,observation


def minimize_blocks(objective,x,bounds,blocks,*,maxiter=40,ftol=1e-12):
    """One population/residual sweep; every accepted block decreases the same QMC objective."""
    x=np.array(x,dtype=float,copy=True);value,gradient=objective(x)
    initial=value;iterations=0;evaluations=1;records=[]
    blocks=[np.asarray(b,dtype=int) for b in blocks if len(b)]
    total=sum(len(b) for b in blocks)
    for block in blocks:
        anchor=x.copy()
        def projected(y):
            point=anchor.copy();point[block]=y
            v,g=objective(point);return v,g[block]
        budget=max(1,int(maxiter*len(block)/max(1,total)))
        result=minimize(projected,x[block],jac=True,method='L-BFGS-B',
            bounds=None if bounds is None else [bounds[j] for j in block],options=dict(maxiter=budget,ftol=ftol,gtol=1e-4,maxls=20))
        candidate=x.copy();candidate[block]=result.x
        proposed,g=objective(candidate);evaluations+=result.nfev+1;iterations+=result.nit
        accepted=bool(np.isfinite(proposed) and proposed<=value+1e-10)
        if accepted:x=candidate;value=proposed;gradient=g
        records.append(dict(coordinates=block.tolist(),accepted=accepted,iterations=int(result.nit)))
    return OptimizeResult(x=x,fun=value,jac=gradient,nit=iterations,nfev=evaluations,
        success=all(r['accepted'] for r in records),message='fixed-bank block sweep',
        block_records=records,initial_value=initial)
