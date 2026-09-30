"""Explicit model contracts, reduced coordinates, fitting and uncertainty.

Omega entries are log-normal random-effect variances, residual entries are
standard deviations. Fixed quantities are absent from optimization and Hessians.
Optional lower/upper bounds on any estimated quantity restrict the optimization
on the reporting scale (as NONMEM $THETA bounds do). No data-dependent model
selection occurs in this interface.
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
    lower: float | None = None
    upper: float | None = None


COVARIATE_FORMS = ('power', 'exponential', 'linear', 'categorical')


@dataclass(frozen=True)
class Covariate:
    """Covariate effect on a structural parameter P with coefficient b.

    power:        P * (x/center)**b          (x > 0; the default)
    exponential:  P * exp(b*(x - center))
    linear:       P * (1 + b*(x - center))
    categorical:  P * exp(b) when x == level, P otherwise (use one effect per non-reference level)
    Power, exponential and categorical effects are log-linear in b.
    """
    parameter: str
    covariate: str
    center: float
    coefficient: Parameter
    form: str = 'power'
    level: float | None = None

    def __post_init__(self):
        if self.form not in COVARIATE_FORMS:
            raise ValueError(f'covariate form must be one of {COVARIATE_FORMS}')
        if self.form == 'categorical' and self.level is None:
            raise ValueError('a categorical covariate effect needs the level it applies to')

    def design(self, values):
        """Log-linear design z with log(P_i/P) = b*z (not defined for the linear form)."""
        values = np.asarray(values, dtype=float)
        if self.form == 'power':
            if np.any(values <= 0) or self.center <= 0:
                raise ValueError('power covariates must be positive and observed for every subject')
            return np.log(values / self.center)
        if self.form == 'exponential':
            return values - self.center
        if self.form == 'categorical':
            return (values == self.level).astype(float)
        raise ValueError('the linear covariate form is not log-linear')

    @staticmethod
    def categorical(parameter, covariate, levels, reference, coefficient=None):
        """One exp(b) effect per non-reference level of a categorical covariate."""
        coefficient = coefficient or Parameter(0.)
        return tuple(Covariate(parameter, covariate, reference, coefficient, 'categorical', level)
                     for level in levels if level != reference)


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


def _check_bounds(p, name, *, positive):
    lo, hi = p.lower, p.upper
    for v in (lo, hi):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
            raise ValueError(f"{name} bounds must be finite numbers or None")
    if positive and lo is not None and lo < 0:
        raise ValueError(f"{name} lower bound must be nonnegative")
    if positive and hi is not None and hi <= 0:
        raise ValueError(f"{name} upper bound must be positive")
    if lo is not None and hi is not None and not lo < hi:
        raise ValueError(f"{name} lower bound must be below the upper bound")
    if (lo is not None and p.value < lo) or (hi is not None and p.value > hi):
        raise ValueError(f"{name} value must lie within its bounds")


def _value(p, name, *, zero=False):
    if not isinstance(p, Parameter) or type(p.fixed) is not bool:
        raise ValueError(f"{name} requires a Parameter with a boolean fixed flag")
    if not np.isfinite(p.value) or p.value < 0 or (p.value == 0 and not (zero and p.fixed)):
        raise ValueError(f"{name} must be positive; only explicitly fixed permitted terms may be zero")
    _check_bounds(p, name, positive=True)


def _log_bounds(p, scale=1.):
    """Reporting-scale bounds of a positive quantity in log coordinates (times scale)."""
    lo = scale*math.log(p.lower) if p.lower is not None and p.lower > 0 else None
    hi = scale*math.log(p.upper) if p.upper is not None else None
    return lo, hi


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
                    or not np.isfinite(e.coefficient.value) or not np.isfinite(e.center)
                    or (e.form == 'power' and e.center <= 0)):
                raise ValueError('invalid power covariate effect')
            if e.form == 'linear':
                raise ValueError('linear covariate effects require the general engine (pkpy2.Model)')
            _check_bounds(e.coefficient, 'covariate coefficient', positive=False)
            key = e.parameter, e.covariate, e.level
            if key in seen: raise ValueError('duplicate covariate effect')
            seen.add(key)
            effects.append(dict(parameter=e.parameter,covariate=e.covariate,center=e.center,
                                beta=e.coefficient.value,fixed=e.coefficient.fixed,
                                lower=e.coefficient.lower,upper=e.coefficient.upper,
                                form=e.form,level=e.level))
        self.spec = dict(fixed_theta={n:p.value for n,p in m.theta.items() if p.fixed},
                         omega={n:p.value for n,p in m.omega.items()},
                         fixed_omega={n:p.value for n,p in m.omega.items() if p.fixed},
                         sigma={n:getattr(m,n).value for n in ('sigma_prop','sigma_add')},
                         fixed_sigma={n:getattr(m,n).value for n in ('sigma_prop','sigma_add') if getattr(m,n).fixed},
                         effects=effects,omega_floor=m.omega_floor,
                         bounds=dict(theta={n:[p.lower,p.upper] for n,p in m.theta.items() if (p.lower,p.upper)!=(None,None)},
                                     omega={n:[p.lower,p.upper] for n,p in m.omega.items() if (p.lower,p.upper)!=(None,None)},
                                     sigma={n:[getattr(m,n).lower,getattr(m,n).upper] for n in ('sigma_prop','sigma_add')
                                            if (getattr(m,n).lower,getattr(m,n).upper)!=(None,None)}))
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
        floor = .5*math.log(m.omega_floor)
        def omega_bounds(p):
            lo, hi = _log_bounds(p, .5)
            return (floor if lo is None else max(floor, lo), hi)
        self.bounds = ([_log_bounds(m.theta[n]) for n in self.free_theta]
                       + [omega_bounds(m.omega[n]) for n in self.free_omega]
                       + [_log_bounds(getattr(m,n)) for n in self.free_sigma]
                       + [(effects[i]['lower'],effects[i]['upper']) for i in self.free_effects])
        self.log_design = []
        for e, spec in zip(effects, m.covariates):
            values = np.array([s.covariates[e['covariate']] for s in self.subjects],dtype=float)
            if values.shape != (len(self.subjects),) or not np.isfinite(values).all():
                raise ValueError('covariates must be observed for every subject')
            self.log_design.append(spec.design(values))
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
        refinement_options=None, laplace_options=None, callback=None):
    from ._numerics.marginal_fit import fit_marginal
    problem=specification.compile(subjects)
    start=time.perf_counter()
    # Multi-start Laplace exploration perturbs typical values and covariate
    # coefficients only; variances and residual terms keep their starting values.
    laplace=dict(structural=[i for i,label in enumerate(problem.labels)
                             if label.startswith(('log_theta:','covariate_coefficient:'))])
    laplace.update(laplace_options or {})
    result=fit_marginal(problem.study,problem.indices,problem.x0,problem.decode,
        bounds=problem.bounds,seed=seed,workers=workers,saem_options=saem_options,
        refinement_options=refinement_options,laplace_options=laplace,callback=callback)
    th,om,sig,beta=problem.unpack(result.x)
    trace=dict(laplace_exploration=result.exploration,
               refinement_status=result.refinement.status,refinement_message=result.refinement.message,
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
    saved=meta.get('bounds',{})
    def bounded(group,name,value,fixed):
        lo,hi=saved.get(group,{}).get(name,(None,None))
        return Parameter(value,fixed,lo,hi)
    spec=ModelSpec(row['model'],{n:bounded('theta',n,v,n in meta['fixed_theta']) for n,v in row['theta'].items()},
        {n:bounded('omega',n,v,n in meta['fixed_omega']) for n,v in row['omega'].items()},
        bounded('sigma','sigma_prop',row['sigma']['sigma_prop'],'sigma_prop' in meta['fixed_sigma']),
        bounded('sigma','sigma_add',row['sigma']['sigma_add'],'sigma_add' in meta['fixed_sigma']),
        tuple(Covariate(e['parameter'],e['covariate'],e['center'],Parameter(b,e['fixed'],e.get('lower'),e.get('upper')),
                        e.get('form','power'),e.get('level'))
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
