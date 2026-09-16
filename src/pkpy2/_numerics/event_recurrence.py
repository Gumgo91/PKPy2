"""Exact linear event propagation for ordered bolus/first-order absorption data."""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def ordered_events(times, dose_times):
    for i in range(1,len(times)):
        if times[i]<times[i-1]:return False
    for i in range(1,len(dose_times)):
        if dose_times[i]<dose_times[i-1]:return False
    return True


@njit(cache=True, nogil=True)
def recursive_one_compartment_values(p,times,dose_times,amounts,result):
    result.fill(0.)
    if not len(times) or not len(dose_times):return
    k=p[0]/p[1];state=0.;clock=min(times[0],dose_times[0]);d=0
    last_dt=-1.;decay=1.
    for i in range(len(times)):
        target=times[i]
        while d<len(dose_times) and dose_times[d]<=target:
            dt=dose_times[d]-clock
            if dt!=last_dt:decay=math.exp(-k*dt);last_dt=dt
            state=state*decay+amounts[d];clock=dose_times[d];d+=1
        dt=target-clock
        if dt!=last_dt:decay=math.exp(-k*dt);last_dt=dt
        state*=decay;clock=target;result[i]=state/p[1]


@njit(cache=True, nogil=True)
def recursive_events_into(code,p,times,dose_times,amounts,result,jac,gradient):
    """Merge dose and observation streams; never reuse states across parameters.

    IV ties are post-dose. Oral lag ties use the inactive-side derivative,
    matching the direct kernel. Sensitivities are propagated analytically.
    """
    if code==0 and not gradient:
        recursive_one_compartment_values(p,times,dose_times,amounts,result)
        return
    result.fill(0.)
    if gradient:jac.fill(0.)
    if not len(times) or not len(dose_times):return
    k=p[0]/p[1];rate1=k;rate2=0.;a=1./p[1];b=0.
    coefficients=np.empty((4,len(p))) if code==2 and gradient else np.empty((0,0))
    if code==2:
        k12=p[2]/p[1];k21=p[2]/p[3];total=k+k12+k21
        disc=total*total-4*k*k21;root=math.sqrt(max(disc,1e-12))
        rate1=(total+root)/2.;rate2=(total-root)/2.
        a=(rate1-k21)/(p[1]*root);b=(k21-rate2)/(p[1]*root)
        if gradient:
            for j in range(len(p)):
                dk=1./p[1] if j==0 else (-k/p[1] if j==1 else 0.)
                d12=1./p[1] if j==2 else (-k12/p[1] if j==1 else 0.)
                d21=1./p[3] if j==2 else (-k21/p[3] if j==3 else 0.)
                dr=(total*(dk+d12+d21)-2*(dk*k21+k*d21))/root if disc>1e-12 else 0.
                da=(dk+d12+d21+dr)/2.;db=(dk+d12+d21-dr)/2.
                coefficients[0,j]=(da-d21)/(p[1]*root)-a*dr/root-(a/p[1] if j==1 else 0.)
                coefficients[1,j]=(d21-db)/(p[1]*root)-b*dr/root-(b/p[1] if j==1 else 0.)
                coefficients[2,j]=da;coefficients[3,j]=db
    lag=p[3] if code==1 else 0.
    ka=p[2] if code==1 else 0.
    clock=min(times[0],dose_times[0]+lag);d=0
    s1=0.;s2=0.;age1=0.;age2=0.
    gut=0.;central=0.;gut_ka=0.;central_k=0.;central_ka=0.
    previous_dt=-1.;e1=1.;e2=1.;conv=0.;conv_k=0.;conv_ka=0.
    for i in range(len(times)):
        target=times[i]
        while True:
            event=dose_times[d]+lag if d<len(dose_times) else math.inf
            is_dose=event<target if code==1 else event<=target
            next_time=event if is_dose else target
            dt=next_time-clock
            if dt>0.:
                if dt!=previous_dt:
                    e1=math.exp(-rate1*dt)
                    if code==2:e2=math.exp(-rate2*dt)
                    elif code==1:
                        e2=math.exp(-ka*dt);z=(ka-k)*dt
                        if abs(z)<1e-4:
                            poly=1-z/2+z*z/6-z*z*z/24+z*z*z*z/120
                            dp=-.5+z/3-z*z/8+z*z*z/30
                            conv=e1*dt*poly
                            conv_k=-e1*dt*dt*(poly+dp)
                            conv_ka=e1*dt*dt*dp
                        else:
                            slow=min(k,ka);gap=abs(ka-k)
                            es=math.exp(-slow*dt);eg=math.exp(-gap*dt)
                            h=-math.expm1(-gap*dt)/gap
                            conv=es*h;dg=es*(dt*eg-h)/gap
                            if ka>k:conv_k=-dt*conv-dg;conv_ka=dg
                            else:conv_k=dg;conv_ka=-dt*conv-dg
                    previous_dt=dt
                if code==1:
                    if gradient:
                        central_k=e1*(central_k-dt*central)+ka*conv_k*gut
                        central_ka=e1*central_ka+(conv+ka*conv_ka)*gut+ka*conv*gut_ka
                        gut_ka=e2*(gut_ka-dt*gut)
                    central=e1*central+ka*conv*gut;gut=e2*gut
                else:
                    if gradient:
                        age1=e1*(age1+dt*s1)
                        if code==2:age2=e2*(age2+dt*s2)
                    s1*=e1
                    if code==2:s2*=e2
            clock=next_time
            if is_dose:
                if code==1:gut+=amounts[d]
                else:
                    s1+=amounts[d]
                    if code==2:s2+=amounts[d]
                d+=1
            else:break
        if code==1:
            result[i]=central/p[1]
            if gradient:
                jac[i,0]=central_k/(p[1]*p[1])
                jac[i,1]=-(central+k*central_k)/(p[1]*p[1])
                jac[i,2]=central_ka/p[1]
                jac[i,3]=-(ka*gut-k*central)/p[1]
        elif code==0:
            result[i]=s1/p[1]
            if gradient:
                jac[i,0]=-age1/(p[1]*p[1])
                jac[i,1]=(k*age1-s1)/(p[1]*p[1])
        else:
            result[i]=a*s1+b*s2
            if gradient:
                for j in range(len(p)):
                    jac[i,j]=(coefficients[0,j]*s1+coefficients[1,j]*s2
                        -a*coefficients[2,j]*age1-b*coefficients[3,j]*age2)
