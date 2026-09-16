"""Explicit model contracts, reduced coordinates, fitting and uncertainty.

Omega entries are log-normal random-effect variances, residual entries are
standard deviations. Fixed quantities are absent from optimization and Hessians.
No data-dependent model selection occurs in this interface.
"""
from dataclasses import dataclass, field, asdict
import copy
import hashlib
import json
import math
import time
import numpy as np

ORDER = {"1cmt_iv": ("CL", "V"), "1cmt_oral": ("CL", "V", "Ka", "ALAG"),
         "2cmt_iv": ("CL", "V1", "Q", "V2"),
         "2cmt_oral": ("CL", "V1", "Q", "V2", "Ka", "ALAG")}
CODES = {"1cmt_iv": 0, "1cmt_oral": 1, "2cmt_iv": 2, "2cmt_oral": 3}


@dataclass(frozen=True)
class Parameter:
    value: float
    fixed: bool = False


@dataclass(frozen=True)
class Covariate:
    parameter: str
    covariate: str
    center: float
    coefficient: Parameter


@dataclass
class Subject:
    sid: int | str
    time: np.ndarray
    obs: np.ndarray
    dose: float
    covariates: dict = field(default_factory=dict)
    dose_history: list = field(default_factory=list)


@dataclass(frozen=True)
class ModelSpec:
    model: str
    theta: dict[str, Parameter]
    omega: dict[str, Parameter]
    sigma_prop: Parameter = Parameter(.15)
    sigma_add: Parameter = Parameter(0., True)
    covariates: tuple[Covariate, ...] = ()
    omega_floor: float = 1e-8

    def compile(self, subjects):
        return CompiledModel(subjects, self)


def _value(p, name, *, zero=False):
    if not isinstance(p, Parameter) or type(p.fixed) is not bool:
        raise ValueError(f"{name} requires a Parameter with a boolean fixed flag")
    if not np.isfinite(p.value) or p.value < 0 or (p.value == 0 and not (zero and p.fixed)):
        raise ValueError(f"{name} must be positive; only explicitly fixed permitted terms may be zero")


class CompiledModel:
    def __init__(self, subjects, model_spec):
        from ._numerics.packed_data import pack_study
        from ._numerics.mu_reference import MuReference
        m = copy.deepcopy(model_spec)
        if m.model not in ORDER:
            raise ValueError("supported models: " + ", ".join(ORDER))
        self.model = m.model
        self.order = list(ORDER[m.model])
        if set(m.theta) != set(self.order):
            raise ValueError("theta must specify exactly: " + ", ".join(self.order))
        if not m.omega or set(m.omega) - (set(self.order) - {"ALAG"}):
            raise ValueError("at least one valid random effect is required; ALAG random effects are unsupported")
        if not np.isfinite(m.omega_floor) or m.omega_floor < 1e-8:
            raise ValueError("omega_floor must be finite and at least 1e-8")
        for n, p in m.theta.items(): _value(p, n, zero=n == 'ALAG')
        for n, p in m.omega.items():
            _value(p, 'omega ' + n)
            if p.value < m.omega_floor: raise ValueError('omega initial/fixed value is below omega_floor')
        for n in ('sigma_prop', 'sigma_add'): _value(getattr(m, n), n, zero=True)
        if m.sigma_prop.value == m.sigma_add.value == 0:
            raise ValueError('at least one positive residual standard deviation is required')
        self.subjects = copy.deepcopy(list(subjects))
        self.study = pack_study(self.subjects, m.model)
        self.theta = {n: p.value for n, p in m.theta.items()}
        effects = []
        seen = set()
        for e in m.covariates:
            if (not isinstance(e, Covariate) or e.parameter not in self.order or e.parameter == 'ALAG'
                    or not isinstance(e.coefficient, Parameter) or type(e.coefficient.fixed) is not bool
                    or not np.isfinite(e.coefficient.value) or not np.isfinite(e.center) or e.center <= 0):
                raise ValueError('invalid power covariate effect')
            key = e.parameter, e.covariate
            if key in seen: raise ValueError('duplicate covariate effect')
            seen.add(key)
            effects.append(dict(parameter=e.parameter,covariate=e.covariate,center=e.center,
                                beta=e.coefficient.value,fixed=e.coefficient.fixed))
        self.spec = dict(fixed_theta={n:p.value for n,p in m.theta.items() if p.fixed},
                         omega={n:p.value for n,p in m.omega.items()},
                         fixed_omega={n:p.value for n,p in m.omega.items() if p.fixed},
                         sigma={n:getattr(m,n).value for n in ('sigma_prop','sigma_add')},
                         fixed_sigma={n:getattr(m,n).value for n in ('sigma_prop','sigma_add') if getattr(m,n).fixed},
                         effects=effects,omega_floor=m.omega_floor)
        self.eta_names = [n for n in self.order if n in m.omega]
        self.indices = [self.order.index(n) for n in self.eta_names]
        self.free_theta = [n for n in self.order if not m.theta[n].fixed]
        self.free_omega = [n for n in self.eta_names if not m.omega[n].fixed]
        self.free_sigma = [n for n in ('sigma_prop','sigma_add') if not getattr(m,n).fixed]
        self.free_effects = [i for i,e in enumerate(effects) if not e['fixed']]
        self.labels = (['log_theta:'+n for n in self.free_theta]
                       + ['log_omega_sd:'+n for n in self.free_omega]
                       + ['log_'+n for n in self.free_sigma]
                       + ['covariate_coefficient:'+str(i) for i in self.free_effects])
        self.x0 = np.array([math.log(self.theta[n]) for n in self.free_theta]
                           + [.5*math.log(m.omega[n].value) for n in self.free_omega]
                           + [math.log(getattr(m,n).value) for n in self.free_sigma]
                           + [effects[i]['beta'] for i in self.free_effects])
        if not len(self.x0): raise ValueError('at least one free parameter is required for fitting')
        self.bounds = ([(None,None)]*len(self.free_theta)
                       + [(.5*math.log(m.omega_floor),None)]*len(self.free_omega)
                       + [(None,None)]*(len(self.free_sigma)+len(self.free_effects)))
        self.log_design = []
        for e in effects:
            values = np.array([s.covariates[e['covariate']] for s in self.subjects],dtype=float)
            if values.shape != (len(self.subjects),) or not np.isfinite(values).all() or np.any(values<=0):
                raise ValueError('power covariates must be positive and observed for every subject')
            self.log_design.append(np.log(values/e['center']))
        data = dict(ids=[(type(s.sid).__name__,str(s.sid)) for s in self.subjects],model=m.model,
                    arrays={n:getattr(self.study,n).tolist() for n in ('observation_offsets','time','observation',
                            'dose_offsets','dose_time','dose_amount')},design=[x.tolist() for x in self.log_design])
        self.data_sha256 = hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()

        def decode(point):
            th,om,sig,beta = self.unpack(point)
            typical = np.tile([th[n] for n in self.order],(len(self.subjects),1))
            for e,z,b in zip(effects,self.log_design,beta,strict=True):
                typical[:,self.order.index(e['parameter'])] *= np.exp(z*b)
            return typical,np.array([om[n] for n in self.eta_names]),sig['sigma_prop'],sig['sigma_add']
        columns,coords = [],[]
        for j,n in enumerate(self.free_theta):
            if n in self.eta_names:
                col=np.zeros((len(self.subjects),len(self.indices)));col[:,self.eta_names.index(n)]=1.
                columns.append(col.ravel());coords.append(j)
        start=len(self.x0)-len(self.free_effects)
        for j,i in enumerate(self.free_effects):
            n=effects[i]['parameter']
            if n in self.eta_names:
                col=np.zeros((len(self.subjects),len(self.indices)));col[:,self.eta_names.index(n)]=self.log_design[i]
                columns.append(col.ravel());coords.append(start+j)
        if columns:
            decode.mu_reference=MuReference(np.array(coords),np.column_stack(columns),
                np.zeros(len(columns)),np.zeros(len(columns)),np.array(self.indices))
        if self.free_omega == self.eta_names:
            decode.variance_coordinates=np.arange(len(self.free_theta),len(self.free_theta)+len(self.indices))
            decode.variance_eta_indices=np.array(self.indices)
        self.decode=decode

    def unpack(self,point):
        point=np.asarray(point)
        if point.shape != self.x0.shape or not np.isfinite(point).all(): raise ValueError('invalid coordinates')
        th=dict(self.theta);om=dict(self.spec['omega']);sig=dict(self.spec['sigma'])
        beta=[e['beta'] for e in self.spec['effects']];j=0
        for n in self.free_theta: th[n]=float(np.exp(point[j]));j+=1
        for n in self.free_omega: om[n]=float(np.exp(2*point[j]));j+=1
        for n in self.free_sigma: sig[n]=float(np.exp(point[j]));j+=1
        for i in self.free_effects: beta[i]=float(point[j]);j+=1
        return th,om,sig,beta


@dataclass
class FitResult:
    model: str
    theta: dict
    omega: dict
    sigma: dict
    coefficients: list
    ofv: float
    status: str
    audit: dict
    seconds: float
    x: np.ndarray
    problem: CompiledModel = field(repr=False)
    estimation: dict = field(default_factory=dict)
    uncertainty_report: dict | None = None

    @property
    def converged(self):
        return self.status == 'converged' and bool(self.audit.get('passed'))

    def uncertainty(self, **options):
        from ._numerics.uncertainty import estimate_uncertainty
        self._validate_point()
        if not self.converged or not np.array_equal(self.x,np.asarray(self.audit['x'])):
            raise ValueError('uncertainty requires the independently audited fitted coordinates')
        self.uncertainty_report=estimate_uncertainty(self.problem,self,**options)
        return self.uncertainty_report

    def _validate_point(self):
        values=self.problem.unpack(self.x)
        same=all(set(a)==set(b) and all(np.isclose(a[n],b[n],rtol=1e-12,atol=0.) for n in a)
                 for a,b in zip(values[:3],(self.theta,self.omega,self.sigma),strict=True))
        if (not same or values[3]!=self.coefficients or self.model!=self.problem.model):
            raise ValueError('reported parameters differ from the saved optimizer coordinates')

    def save(self, path):
        """Store a portable JSON result; loading requires the original data."""
        from pathlib import Path
        self._validate_point()
        def convert(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if isinstance(value,np.generic):return value.item()
            raise TypeError(type(value).__name__)
        Path(path).write_text(json.dumps(self.to_dict(),indent=2,default=convert,allow_nan=False),encoding='utf-8')

    def to_dict(self):
        return copy.deepcopy(dict(format='pkpy2-fit-v1',model=self.model,theta=self.theta,omega=self.omega,sigma=self.sigma,
            coefficients=self.coefficients,ofv=self.ofv,status=self.status,converged=self.converged,
            audit=self.audit,seconds=self.seconds,x=self.x.tolist(),n_param=len(self.x),
            aic=self.ofv+2*len(self.x),bic=self.ofv+len(self.x)*math.log(len(self.problem.study.time)),
            data_sha256=self.problem.data_sha256,specification=self.problem.spec,
            coordinates=self.problem.labels,uncertainty=self.uncertainty_report,estimation=self.estimation,
            information_criteria_audited=self.converged))


def fit(subjects, specification, *, seed=0, workers=1, saem_options=None,
        refinement_options=None, callback=None):
    from ._numerics.marginal_fit import fit_marginal
    problem=specification.compile(subjects)
    start=time.perf_counter()
    result=fit_marginal(problem.study,problem.indices,problem.x0,problem.decode,
        bounds=problem.bounds,seed=seed,workers=workers,saem_options=saem_options,
        refinement_options=refinement_options,callback=callback)
    th,om,sig,beta=problem.unpack(result.x)
    trace=dict(refinement_status=result.refinement.status,refinement_message=result.refinement.message,
               refinement_stages=result.refinement.stages,saem_status=result.saem.status,
               saem_message=result.saem.message,saem_diagnostics=result.saem.diagnostics,
               total_cpu_seconds=result.cpu_seconds,seed=seed)
    nonfinite=[]
    def clean(value,path):
        if isinstance(value,np.ndarray):value=value.tolist()
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,dict):return {k:clean(v,path+'.'+str(k)) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v,path+'.'+str(i)) for i,v in enumerate(value)]
        if isinstance(value,float) and not np.isfinite(value):
            nonfinite.append(path);return None
        return value
    trace=clean(trace,'estimation');trace['nonfinite_diagnostic_fields']=nonfinite
    return FitResult(problem.model,th,om,sig,beta,float(result.ofv),result.status,
                     result.refinement.audit,time.perf_counter()-start,result.x.copy(),problem,trace)


def load_fit(path, subjects):
    """Restore a saved result only when the supplied analysis data match."""
    from pathlib import Path
    row=json.loads(Path(path).read_text(encoding='utf-8'))
    if row.get('format')!='pkpy2-fit-v1':raise ValueError('unsupported saved fit format')
    meta=row['specification']
    spec=ModelSpec(row['model'],{n:Parameter(v,n in meta['fixed_theta']) for n,v in row['theta'].items()},
        {n:Parameter(v,n in meta['fixed_omega']) for n,v in row['omega'].items()},
        Parameter(row['sigma']['sigma_prop'],'sigma_prop' in meta['fixed_sigma']),
        Parameter(row['sigma']['sigma_add'],'sigma_add' in meta['fixed_sigma']),
        tuple(Covariate(e['parameter'],e['covariate'],e['center'],Parameter(b,e['fixed']))
              for e,b in zip(meta['effects'],row['coefficients'],strict=True)),meta['omega_floor'])
    problem=spec.compile(subjects)
    if problem.data_sha256!=row['data_sha256'] or problem.labels!=row['coordinates']:
        raise ValueError('saved fit and supplied analysis data/coordinates differ')
    problem.spec=copy.deepcopy(meta)
    result=FitResult(row['model'],row['theta'],row['omega'],row['sigma'],row['coefficients'],row['ofv'],
        row['status'],row['audit'],row['seconds'],np.array(row['x']),problem,row.get('estimation',{}),row.get('uncertainty'))
    result._validate_point()
    if result.converged and not np.array_equal(result.x,np.asarray(result.audit['x'])):
        raise ValueError('saved estimate differs from independently audited coordinates')
    return result


def predict(model,theta,times,doses):
    """Concentration at observation times for explicit bolus/oral dose events."""
    from ._numerics.packed_kernels import event_predictions
    if model not in ORDER or set(theta)!=set(ORDER[model]): raise ValueError('invalid model/theta')
    for n,v in theta.items(): _value(Parameter(v,n=='ALAG'),n,zero=n=='ALAG')
    t=np.asarray(times,dtype=float);events=np.asarray(doses,dtype=float)
    if t.ndim!=1 or not np.isfinite(t).all(): raise ValueError('invalid observation times')
    if events.ndim!=2 or events.shape[1]!=2 or not np.isfinite(events).all() or np.any(events[:,1]<0):
        raise ValueError('doses must be finite (time, nonnegative amount) events')
    return event_predictions(CODES[model],np.array([theta[n] for n in ORDER[model]],dtype=float),
                             t,events[:,0].copy(),events[:,1].copy())
