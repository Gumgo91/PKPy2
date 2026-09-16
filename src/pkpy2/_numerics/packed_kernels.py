"""Experimental packed event kernels; mandatory compilation, strict arithmetic.

Parameters are individual values ordered CL,V[,Ka,ALAG] or CL,V1,Q,V2.
This layer evaluates conditional observation likelihood, not the marginal
population objective. ETA priors and Laplace corrections belong to the solver.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def absorption_convolution(rate, ka, t):
    """Integral exp(-rate*(t-s))*exp(-ka*s) ds and natural derivatives."""
    z=(ka-rate)*t
    if abs(z)<1e-4:
        poly=1.-z/2.+z*z/6.-z*z*z/24.+z*z*z*z/120.
        dp=-.5+z/3.-z*z/8.+z*z*z/30.
        e=math.exp(-rate*t)
        return (t*e*poly,t*e*(-t*poly-t*dp),t*t*e*dp,
                e*((1.-rate*t)*poly+t*dp*(ka-rate)))
    slow=min(rate,ka);gap=abs(ka-rate)
    e=math.exp(-slow*t);eg=math.exp(-gap*t)
    h=-math.expm1(-gap*t)/gap;dh=(t*eg-h)/gap
    g=e*h;rs=-t*g;rg=e*dh
    gr,gka=(rs-rg,rg) if ka>rate else (rg,rs-rg)
    return g,gr,gka,e*(eg-slow*h)


@njit(cache=True, nogil=True)
def direct_event_predictions(code, p, times, dose_times, amounts):
    result = np.zeros(len(times))
    k = p[0] / p[1]
    if code == 2 or code == 3:
        k12, k21 = p[2] / p[1], p[2] / p[3]
        total = k + k12 + k21
        root = math.sqrt(max(total * total - 4. * k * k21, 1e-12))
        alpha = (total + root) / 2.
        beta = (total - root) / 2.
        a = (alpha - k21) / (p[1] * root)
        b = (k21 - beta) / (p[1] * root)
    for j in range(len(times)):
        value = 0.
        for d in range(len(dose_times)):
            t = times[j] - dose_times[d]
            if t < 0.:
                continue
            if code == 0:
                response = math.exp(-k * t) / p[1]
            elif code == 1:
                t = max(t - p[3], 0.)
                ka = p[2]
                z = (ka - k) * t
                if abs(z) < 1e-4:
                    response = ka / p[1] * t * math.exp(-k * t) * (1. - z/2. + z*z/6. - z*z*z/24. + z*z*z*z/120.)
                else:
                    # Factor by the slower exponential to avoid cancellation
                    # and overflow from expm1 of a large positive argument.
                    slow = min(k, ka)
                    gap = abs(ka - k)
                    response = ka / (p[1] * gap) * math.exp(-slow*t) * (-math.expm1(-gap*t))
            elif code == 3:
                t=max(t-p[5],0.)
                response=p[4]*(a*absorption_convolution(alpha,p[4],t)[0]
                               +b*absorption_convolution(beta,p[4],t)[0])
            else:
                response = a * math.exp(-alpha*t) + b * math.exp(-beta*t)
            value += amounts[d] * response
        result[j] = value
    return result


from .event_recurrence import ordered_events, recursive_events_into


@njit(cache=True, nogil=True, inline='always')
def event_predictions(code,p,times,dose_times,amounts):
    if code==3 or len(dose_times)<=1 or len(times)<=1 or (code==0 and len(times)<=2) or not ordered_events(times,dose_times):
        return direct_event_predictions(code,p,times,dose_times,amounts)
    result=np.empty(len(times));scratch=np.empty((0,0))
    recursive_events_into(code,p,times,dose_times,amounts,result,scratch,False)
    return result


@njit(cache=True, nogil=True)
def observation_nll(code, p, times, observed, dose_times, amounts, sp, sa):
    predictions = event_predictions(code, p, times, dose_times, amounts)
    value = 0.
    for j in range(len(observed)):
        f = max(predictions[j], 1e-10)
        variance = max((sp*f)**2 + sa**2, 1e-12)
        value += .5 * (math.log(2.*math.pi*variance) + (observed[j]-f)**2/variance)
    return value


@njit(cache=True, nogil=True)
def joint_nll(eta, code, typical, eta_indices, omega, times, observed,
              dose_times, amounts, sp, sa):
    p = typical.copy()
    prior = 0.
    for j in range(len(eta)):
        p[eta_indices[j]] *= math.exp(eta[j])
        variance = max(omega[j], 1e-8)
        prior += .5*(math.log(2.*math.pi*variance)+eta[j]**2/variance)
    return observation_nll(code,p,times,observed,dose_times,amounts,sp,sa)+prior


@njit(cache=True, nogil=True)
def joint_value_gradient(eta, code, typical, eta_indices, omega, times, observed,
                         dose_times, amounts, sp, sa):
    """Compiled forward differences; deliberately not called analytic derivatives."""
    value = joint_nll(eta,code,typical,eta_indices,omega,times,observed,dose_times,amounts,sp,sa)
    gradient = np.empty(len(eta))
    point = eta.copy()
    for j in range(len(eta)):
        step = 1.4901161193847656e-8 * max(1.,abs(eta[j]))
        if eta[j] < 0.:
            step = -step
        point[j] = eta[j]+step
        effective = point[j]-eta[j]
        gradient[j] = (joint_nll(point,code,typical,eta_indices,omega,times,observed,
                                 dose_times,amounts,sp,sa)-value)/effective
        point[j] = eta[j]
    return value, gradient


@njit(cache=True, nogil=True)
def joint_hessian(eta, code, typical, eta_indices, omega, times, observed,
                  dose_times, amounts, sp, sa):
    """Full central Hessian, including cross terms, with reference step 1e-3."""
    h = 1e-3
    matrix = np.empty((len(eta),len(eta)))
    point = eta.copy()
    base = joint_nll(eta,code,typical,eta_indices,omega,times,observed,dose_times,amounts,sp,sa)
    for i in range(len(eta)):
        point[i]=eta[i]+h
        plus=joint_nll(point,code,typical,eta_indices,omega,times,observed,dose_times,amounts,sp,sa)
        point[i]=eta[i]-h
        minus=joint_nll(point,code,typical,eta_indices,omega,times,observed,dose_times,amounts,sp,sa)
        point[i]=eta[i]
        matrix[i,i]=(plus-2.*base+minus)/(h*h)
        for j in range(i+1,len(eta)):
            cross=0.
            for si in (-1.,1.):
                for sj in (-1.,1.):
                    point[i]=eta[i]+si*h
                    point[j]=eta[j]+sj*h
                    cross += si*sj*joint_nll(point,code,typical,eta_indices,omega,times,
                                            observed,dose_times,amounts,sp,sa)
            point[i]=eta[i]
            point[j]=eta[j]
            matrix[i,j]=matrix[j,i]=cross/(4.*h*h)
    return matrix


@njit(cache=True, nogil=True)
def joint_central_gradient(eta, code, typical, eta_indices, omega, times, observed,
                           dose_times, amounts, sp, sa):
    """Symmetric differences for mode refinement, with a stable relative step."""
    gradient = np.empty(len(eta))
    point = eta.copy()
    for j in range(len(eta)):
        step = 6.055454452393343e-6 * max(1., abs(eta[j]))
        point[j] = eta[j] + step
        plus = joint_nll(point,code,typical,eta_indices,omega,times,observed,dose_times,amounts,sp,sa)
        point[j] = eta[j] - step
        minus = joint_nll(point,code,typical,eta_indices,omega,times,observed,dose_times,amounts,sp,sa)
        gradient[j] = (plus-minus)/(2.*step)
        point[j] = eta[j]
    return gradient


def predict_packed(study, individual_parameters):
    """One validated batch boundary; no Subject callbacks inside event kernels."""
    codes = {'1cmt_iv': 0, '1cmt_oral': 1, '2cmt_iv': 2, '2cmt_oral': 3}
    if study.model not in codes:
        raise ValueError('packed prediction currently supports bolus and oral models')
    p = np.ascontiguousarray(individual_parameters, dtype=np.float64)
    columns = 2 if study.model == '1cmt_iv' else 6 if study.model == '2cmt_oral' else 4
    if p.shape != (len(study.subject_ids), columns) or not np.isfinite(p).all():
        raise ValueError('finite individual parameter matrix required')
    positive = p[:, :-1] if study.model in ('1cmt_oral','2cmt_oral') else p
    if np.any(positive <= 0) or (study.model in ('1cmt_oral','2cmt_oral') and np.any(p[:, -1] < 0)):
        raise ValueError('invalid individual parameters')
    return _batch(codes[study.model], p, study.observation_offsets, study.time,
                  study.dose_offsets, study.dose_time, study.dose_amount)


@njit(cache=True, nogil=True)
def _batch(code, parameters, offsets, times, dose_offsets, dose_times, amounts):
    result = np.empty(len(times))
    for i in range(len(parameters)):
        lo, hi = offsets[i], offsets[i+1]
        dl, dh = dose_offsets[i], dose_offsets[i+1]
        result[lo:hi] = event_predictions(code, parameters[i], times[lo:hi],
                                         dose_times[dl:dh], amounts[dl:dh])
    return result
