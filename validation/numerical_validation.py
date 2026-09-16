"""Independent matrix-exponential/ODE and finite-difference numerical checks.

Reproduces the numerical validation reported in the PKPy2 manuscript:
analytical predictions and parameter sensitivities for the four supported
structural models are compared, under 12 parameter conditions each, against an
independent matrix-exponential implementation (``scipy.linalg.expm``) and, for
selected conditions, DOP853 numerical integration. Analytical gradients of the
observation-level objective are checked against central finite differences of
the independent prediction path.

Usage (from the repository root, with pkpy2 installed):

    python validation/numerical_validation.py

Writes ``output/numerical_validation.json`` and prints a summary.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from pkpy2 import Subject, ModelSpec, Parameter as P, Covariate, predict
from pkpy2.api import ORDER, CODES
from pkpy2._numerics.packed_sensitivities import event_predictions_gradient
from pkpy2._numerics.packed_derivatives import precise_gradient
from pkpy2._numerics.packed_kernels import joint_nll, direct_event_predictions

ROOT=Path(__file__).resolve().parents[1]


def matrix_reference(model,theta,times,doses,ode=False):
    cl=theta['CL'];v=theta.get('V',theta.get('V1'))
    two=model.startswith('2');oral=model.endswith('oral')
    a=np.array([[-cl/v]])
    if two:
        q,v2=theta['Q'],theta['V2']
        a=np.array([[-(cl+q)/v,q/v2],[q/v,-q/v2]])
    if oral:
        b=np.zeros((len(a)+1,len(a)+1));b[1:,1:]=a
        b[0,0]=-theta['Ka'];b[1,0]=theta['Ka'];a=b
    out=[]
    for t in times:
        amount=0.
        for start,dose in doses:
            dt=t-start-(theta.get('ALAG',0.) if oral else 0.)
            if dt<0:continue
            initial=np.zeros(len(a));initial[0]=dose
            state=(solve_ivp(lambda _,x:a@x,(0,dt),initial,rtol=1e-11,atol=1e-12,
                             method='DOP853',t_eval=[dt]).y[:,-1]
                   if ode and dt>0 else expm(a*dt)@initial)
            amount+=state[1 if oral else 0]/v
        out.append(amount)
    return np.array(out)


def main():
    start=time.perf_counter();rng=np.random.default_rng(26091411);rows=[]
    for model in ORDER:
        for case in range(12):
            th={'CL':4.,'V':40.} if model.startswith('1') else {'CL':4.,'V1':25.,'Q':8.,'V2':50.}
            if model.endswith('oral'):th.update(Ka=1.2,ALAG=.15)
            th={n:v*np.exp(rng.uniform(-1,1)) for n,v in th.items()}
            # Analytic absorption is evaluated at coincident/near-coincident rates.
            if model=='1cmt_oral' and case in (0,1):th['Ka']=th['CL']/th['V']*(1+case*1e-9)
            if model=='2cmt_oral' and case in (0,1):
                k=th['CL']/th['V1'];k12=th['Q']/th['V1'];k21=th['Q']/th['V2']
                alpha=(k+k12+k21+np.sqrt((k+k12+k21)**2-4*k*k21))/2
                th['Ka']=alpha*(1+case*1e-9)
            times=np.r_[-1.,np.linspace(.31,36,21)]
            doses=[(0.,100.),(4.,75.),(11.,125.)]
            p=np.array([th[n] for n in ORDER[model]])
            prediction=predict(model,th,times,doses)
            ref=matrix_reference(model,th,times,doses,ode=case==2)
            error=float(np.max(np.abs(prediction-ref)/np.maximum(1.,np.abs(ref))))
            y,jac=event_predictions_gradient(CODES[model],p,times,np.array(doses)[:,0],np.array(doses)[:,1])
            jac_ref=np.empty_like(jac)
            for j,n in enumerate(ORDER[model]):
                h=1e-5*max(1.,abs(p[j]));plus=dict(th);minus=dict(th)
                plus[n]+=h;minus[n]-=h
                jac_ref[:,j]=(matrix_reference(model,plus,times,doses)-matrix_reference(model,minus,times,doses))/(2*h)
            derivative_error=float(np.max(np.abs(jac-jac_ref)/np.maximum(1.,np.abs(jac_ref))))
            assert error<1e-8,(model,case,error)
            assert derivative_error<1e-5,(model,case,derivative_error)
            indices=np.array([0,1],dtype=np.int64);eta=np.array([.1,-.05]);om=np.array([.09,.04])
            obs=np.maximum(prediction,.01)*1.03;dt=np.array(doses)[:,0];amounts=np.array(doses)[:,1]
            g=precise_gradient(eta,CODES[model],p,indices,om,times,obs,dt,amounts,.15,.1)
            fd=[]
            for j in range(2):
                step=np.eye(2)[j]*1e-5
                fd.append((joint_nll(eta+step,CODES[model],p,indices,om,times,obs,dt,amounts,.15,.1)
                           -joint_nll(eta-step,CODES[model],p,indices,om,times,obs,dt,amounts,.15,.1))/2e-5)
            score_error=float(np.max(np.abs(g-fd)/np.maximum(1.,np.abs(fd))))
            assert score_error<1e-5,(model,case,score_error)
            rows.append(dict(model=model,case=case,prediction_relative_scaled_error=error,
                             sensitivity_relative_scaled_error=derivative_error,score_relative_scaled_error=score_error))
    subjects=[Subject(i,np.array([1.,4.]),np.array([2.,1.]),100.,{'WT':50.+i*10}) for i in range(3)]
    spec=ModelSpec('1cmt_oral',{'CL':P(4.),'V':P(40.),'Ka':P(1.2),'ALAG':P(0.,True)},
        {'CL':P(.09),'Ka':P(.4,True)},P(.15),P(.1,True),
        (Covariate('V','WT',70.,P(1.,True)),Covariate('CL','WT',70.,P(.75))))
    problem=spec.compile(subjects)
    for j in range(len(problem.x0)):
        x=problem.x0.copy();x[j]+=.03;th,om,sig,b=problem.unpack(x)
        assert th['ALAG']==0 and om['Ka']==.4 and sig['sigma_add']==.1 and b[0]==1.
    copied=problem.study.observation.copy();subjects[0].obs[0]=999
    assert np.array_equal(problem.study.observation,copied)
    assert not any(n=='pkagent' or n.startswith('pkagent.') for n in sys.modules)
    result=dict(status='passed',independent_numerical_cases=rows,fixed_components_invariant=True,
                input_ownership=True,standalone_import=True,seconds=time.perf_counter()-start)
    output=ROOT/'output/numerical_validation.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='independent_numerical_cases'}),flush=True)


if __name__=='__main__':main()
