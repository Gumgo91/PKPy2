"""Safeguarded mean regression and phase scheduling; neither certifies convergence."""
import numpy as np


def mean_regression_candidate(x,decode,indices,target,bounds=None):
    """Propose a weighted regression in locally affine log-mean coordinates.

    Only coordinates that leave variances, residuals and non-ETA predictions
    unchanged qualify. The caller must accept using the complete penalized Q.
    """
    p,omega,sp,sa=decode(x);p=np.asarray(p);omega=np.asarray(omega)
    other=[j for j in range(p.shape[1]) if j not in indices]
    mu=np.log(p[:,indices]);columns=[];free=[]
    for j in range(len(x)):
        h=1e-4*max(1.,abs(x[j]));a=x.copy();b=x.copy();a[j]+=h;b[j]-=h
        if bounds is not None:
            lo,hi=bounds[j]
            if lo is not None:b[j]=max(b[j],lo)
            if hi is not None:a[j]=min(a[j],hi)
        if a[j]==b[j]:continue
        pa,oa,spa,saa=decode(a);pb,ob,spb,sab=decode(b)
        if not (np.array_equal(oa,omega) and np.array_equal(ob,omega)
                and (spa,saa)==(sp,sa) and (spb,sab)==(sp,sa)
                and np.array_equal(np.asarray(pa)[:,other],p[:,other])
                and np.array_equal(np.asarray(pb)[:,other],p[:,other])):continue
        derivative=(np.log(np.asarray(pa)[:,indices])-np.log(np.asarray(pb)[:,indices]))/(a[j]-b[j])
        if np.max(np.abs(derivative),initial=0.)>1e-12:
            columns.append((derivative/np.sqrt(omega)).ravel());free.append(j)
    if not free:return x.copy(),0
    design=np.column_stack(columns)
    delta=np.linalg.lstsq(design,((target-mu)/np.sqrt(omega)).ravel(),rcond=1e-10)[0]
    candidate=x.copy();candidate[free]+=delta
    if bounds is not None:
        for j in free:
            lo,hi=bounds[j]
            if lo is not None:candidate[j]=max(candidate[j],lo)
            if hi is not None:candidate[j]=min(candidate[j],hi)
    pp,oo,ss,aa=decode(candidate)
    predicted=mu+(design@(candidate[free]-x[free])).reshape(mu.shape)*np.sqrt(omega)
    if not (np.array_equal(oo,omega) and (ss,aa)==(sp,sa)
            and np.array_equal(np.asarray(pp)[:,other],p[:,other])
            and np.allclose(np.log(np.asarray(pp)[:,indices]),predicted,rtol=1e-7,atol=1e-8)):
        return x.copy(),0
    return candidate,len(free)


def phase_stable(records,window,tolerance):
    """Two-window stability for scheduling only, never marginal stationarity."""
    if len(records)<2*window:return False
    recent=records[-2*window:]
    x=np.asarray([r['x'] for r in recent])
    q=np.asarray([r['complete_q'] for r in recent])
    if not np.isfinite(x).all() or not np.isfinite(q).all():return False
    left=x[:window].mean(axis=0);right=x[window:].mean(axis=0)
    shift=np.max(np.abs(left-right)/np.maximum(1.,np.abs(right)),initial=0.)
    # Account for stochastic Q noise while requiring a small relative drift.
    qshift=abs(q[:window].mean()-q[window:].mean())/max(1.,abs(q.mean()))
    accepted=all(r['mstep_accepted'] for r in recent)
    return bool(accepted and shift<tolerance and qshift<tolerance)


def parameter_windows_stable(records,tolerance):
    """Same absolute-coordinate stability criterion for handoff and reporting."""
    if len(records)<40:return False
    values=np.asarray([r['x'] for r in records[-40:]])
    if not np.isfinite(values).all():return False
    windows=values.reshape(4,10,-1).mean(axis=1)
    return bool(np.max(np.ptp(windows,axis=0))<tolerance)


def conditional_variance_candidate(x,decode,mean,second,bounds=None):
    """Exact diagonal Gaussian variance CM step for explicitly declared coordinates.

    The producer guarantees these coordinates affect only variances and carry
    no variance-dependent penalty. Generic decoders remain on the general path.
    """
    coordinates=getattr(decode,'variance_coordinates',None)
    if coordinates is None:return x.copy(),np.array([],dtype=int)
    coordinates=np.asarray(coordinates,dtype=int)
    p,omega,sp,sa=decode(x)
    indices=np.asarray(getattr(decode,'variance_eta_indices'),dtype=int)
    mu=np.log(np.asarray(p)[:,indices])
    residual=np.mean(second-2*mu*mean+mu*mu,axis=0)
    if len(coordinates)!=len(omega) or not np.isfinite(residual).all():
        return x.copy(),np.array([],dtype=int)
    candidate=x.copy()
    for j,value in zip(coordinates,residual):
        value=.5*np.log(max(float(value),np.finfo(float).tiny))
        if bounds is not None:
            lo,hi=bounds[j]
            if lo is not None:value=max(value,lo)
            if hi is not None:value=min(value,hi)
        candidate[j]=value
    return candidate,coordinates
