"""Analytic natural-parameter sensitivities of the packed event formulas."""
import math
import numpy as np
from numba import njit
from .packed_kernels import absorption_convolution


@njit(cache=True, nogil=True)
def event_predictions_gradient(code, p, times, dose_times, amounts):
    result = np.empty(len(times))
    jac = np.empty((len(times), len(p)))
    coefficients = np.empty((4, len(p)))
    event_predictions_gradient_into(code,p,times,dose_times,amounts,result,jac,coefficients)
    return result, jac


@njit(cache=True, nogil=True)
def direct_event_predictions_gradient_into(code,p,times,dose_times,amounts,result,jac,coefficients):
    """Overwrite caller-owned scratch buffers; no particle-local allocations."""
    n = len(p)
    result.fill(0.); jac.fill(0.)
    k = p[0] / p[1]
    alpha = beta = a = b = 0.
    if code == 2 or code == 3:
        k12, k21 = p[2] / p[1], p[2] / p[3]
        total = k + k12 + k21
        disc = total * total - 4. * k * k21
        root = math.sqrt(max(disc, 1e-12))
        alpha, beta = (total+root)/2., (total-root)/2.
        a, b = (alpha-k21)/(p[1]*root), (k21-beta)/(p[1]*root)
        for j in range(n):
            dk = 1./p[1] if j==0 else (-k/p[1] if j==1 else 0.)
            d12 = 1./p[1] if j==2 else (-k12/p[1] if j==1 else 0.)
            d21 = 1./p[3] if j==2 else (-k21/p[3] if j==3 else 0.)
            dr = 0.
            if disc > 1e-12:
                dr = (total*(dk+d12+d21)-2.*(dk*k21+k*d21))/root
            dalpha,dbeta = (dk+d12+d21+dr)/2.,(dk+d12+d21-dr)/2.
            da = (dalpha-d21)/(p[1]*root)-a*dr/root
            db = (d21-dbeta)/(p[1]*root)-b*dr/root
            if j==1:da-=a/p[1];db-=b/p[1]
            coefficients[0,j]=da;coefficients[1,j]=db
            coefficients[2,j]=dalpha;coefficients[3,j]=dbeta
    for i in range(len(times)):
        for d in range(len(dose_times)):
            t = times[i] - dose_times[d]
            if t < 0.:
                continue
            if code == 0:
                response = math.exp(-k*t)/p[1]
                for j in range(n):
                    dk = 1./p[1] if j==0 else (-k/p[1] if j==1 else 0.)
                    jac[i,j] += amounts[d]*(-t*response*dk)
                jac[i,1] -= amounts[d]*response/p[1]
            elif code == 1:
                active = t > p[3]
                t = max(t-p[3], 0.)
                ka = p[2]; z = (ka-k)*t
                if abs(z) < 1e-4:
                    poly = 1.-z/2.+z*z/6.-z*z*z/24.+z*z*z*z/120.
                    dp = -.5+z/3.-z*z/8.+z*z*z/30.
                    e = math.exp(-k*t)
                    response = ka/p[1]*t*e*poly
                    rk = ka/p[1]*t*e*(-t*poly-t*dp)
                    rka = t*e/p[1]*(poly+ka*t*dp)
                    rt = ka/p[1]*e*((1.-k*t)*poly+t*dp*(ka-k))
                else:
                    slow = min(k, ka); gap = abs(ka-k)
                    e = math.exp(-slow*t); eg = math.exp(-gap*t)
                    h = -math.expm1(-gap*t)/gap
                    dh = (t*eg-h)/gap
                    response = ka/p[1]*e*h
                    rs = -t*response
                    rg = ka/p[1]*e*dh
                    if ka > k:
                        rk, rka = rs-rg, e*h/p[1]+rg
                    else:
                        rk, rka = rg, e*h/p[1]+rs-rg
                    rt = ka/p[1]*e*(eg-slow*h)
                for j in range(n):
                    dk = 1./p[1] if j==0 else (-k/p[1] if j==1 else 0.)
                    jac[i,j] += amounts[d]*rk*dk
                jac[i,1] -= amounts[d]*response/p[1]
                jac[i,2] += amounts[d]*rka
                # At the lag kink use the inactive-side derivative. No smooth
                # derivative exists there; finite differences must avoid it.
                if active:
                    jac[i,3] -= amounts[d]*rt
            elif code == 3:
                active=t>p[5];t=max(t-p[5],0.)
                ga,gar,gak,gat=absorption_convolution(alpha,p[4],t)
                gb,gbr,gbk,gbt=absorption_convolution(beta,p[4],t)
                response=p[4]*(a*ga+b*gb)
                for j in range(n):
                    derivative=p[4]*(coefficients[0,j]*ga+a*gar*coefficients[2,j]
                                     +coefficients[1,j]*gb+b*gbr*coefficients[3,j])
                    if j==4: derivative+=a*ga+b*gb+p[4]*(a*gak+b*gbk)
                    if j==5 and active: derivative-=p[4]*(a*gat+b*gbt)
                    jac[i,j]+=amounts[d]*derivative
            else:
                ea, eb = math.exp(-alpha*t), math.exp(-beta*t)
                response = a*ea+b*eb
                for j in range(n):
                    jac[i,j] += amounts[d]*(ea*(coefficients[0,j]-a*t*coefficients[2,j])+eb*(coefficients[1,j]-b*t*coefficients[3,j]))
            result[i] += amounts[d]*response


from .event_recurrence import ordered_events, recursive_events_into


@njit(cache=True, nogil=True, inline='always')
def event_predictions_gradient_into(code,p,times,dose_times,amounts,result,jac,coefficients):
    if code==3 or len(dose_times)<=1 or len(times)<=1 or not ordered_events(times,dose_times):
        direct_event_predictions_gradient_into(code,p,times,dose_times,amounts,result,jac,coefficients)
    else:
        recursive_events_into(code,p,times,dose_times,amounts,result,jac,True)
