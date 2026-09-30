"""Structural models: linear compartment systems and ordinary differential equations.

A structure names its parameters, states (compartments) and outputs. Dose records
address states by NONMEM compartment number (CMT 1 is the first state); observation
records address outputs by DVID (DVID 1 is the first output). Every structure may
declare, per dosed state, a bioavailability (F), an absorption lag, a modeled
infusion duration (used for RATE = -2 or for zero-order absorption) and a modeled
rate (RATE = -1).

Linear structures are advanced exactly (eigendecomposition or matrix exponential). ODE structures
take a right-hand side written in plain Python that Numba can compile:
``rhs(t, x, p, dxdt)``, an output function ``output(x, p, out)`` and an initial
state ``initial(p, x)``. Infusion inputs are added to dxdt by the solver.
"""
from dataclasses import dataclass, field
import numpy as np
from numba import njit

_KINDS = {'const': 0, 'p': 1, 'ratio': 2, 'inverse': 3, 'product': 4, 'product_ratio': 5, 'complement_ratio': 6}
_PD = {None: 0, 'emax': 1, 'sigmoid_emax': 2, 'imax': 3, 'linear': 4, 'sigmoid_imax': 5}
_PD_PARAMETERS = {None: (), 'emax': ('E0', 'EMAX', 'EC50'), 'sigmoid_emax': ('E0', 'EMAX', 'EC50', 'GAMMA'),
                  'imax': ('E0', 'IMAX', 'IC50'), 'linear': ('E0', 'SLOPE'),
                  'sigmoid_imax': ('E0', 'IMAX', 'IC50', 'GAMMA')}


@dataclass
class Output:
    name: str
    state: str
    volume: str | None = None
    pd: str | None = None
    pd_parameters: tuple = ()


@dataclass
class Dosing:
    bioavailability: str | None = None
    lag: str | None = None
    duration: str | None = None
    rate: str | None = None
    zero_order: bool = False


class Structure:
    """Common interface; see LinearStructure and ODEStructure."""
    kind = 'abstract'

    def __init__(self, name, parameters, states, outputs, dosing):
        self.name = name
        self.parameters = tuple(parameters)
        self.states = tuple(states)
        self.outputs = tuple(outputs)
        self.dosing = {k: v for k, v in (dosing or {}).items()}
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError('duplicate structural parameter names')
        for state, spec in self.dosing.items():
            if state not in self.states:
                raise ValueError(f'dosing refers to unknown state {state}')
            for name in (spec.bioavailability, spec.lag, spec.duration, spec.rate):
                if name is not None and name not in self.parameters:
                    raise ValueError(f'dosing parameter {name} is not a structural parameter')
        for out in self.outputs:
            if out.state not in self.states:
                raise ValueError(f'output {out.name} refers to unknown state {out.state}')
            if out.volume is not None and out.volume not in self.parameters:
                raise ValueError(f'output volume {out.volume} is not a structural parameter')
            for name in out.pd_parameters:
                if name not in self.parameters:
                    raise ValueError(f'PD parameter {name} is not a structural parameter')

    @property
    def output_names(self):
        return tuple(o.name for o in self.outputs)

    def dosing_arrays(self):
        n = len(self.states)
        index = {name: j for j, name in enumerate(self.parameters)}
        f = np.full(n, -1, dtype=np.int64); lag = np.full(n, -1, dtype=np.int64)
        dur = np.full(n, -1, dtype=np.int64); rate = np.full(n, -1, dtype=np.int64)
        auto = np.zeros(n, dtype=np.int64)
        for state, spec in self.dosing.items():
            s = self.states.index(state)
            if spec.bioavailability: f[s] = index[spec.bioavailability]
            if spec.lag: lag[s] = index[spec.lag]
            if spec.duration: dur[s] = index[spec.duration]
            if spec.rate: rate[s] = index[spec.rate]
            auto[s] = int(spec.zero_order)
        return f, lag, dur, rate, auto

    def describe(self):
        return dict(name=self.name, kind=self.kind, parameters=list(self.parameters), states=list(self.states),
                    outputs=[o.name for o in self.outputs])


class LinearStructure(Structure):
    """dx/dt = A(p) x + inputs, with A assembled from rate terms.

    Each term is (row state, column state, kind, parameter names, constant) and adds
    constant * g(p) to A[row, column], where g is 1 ('const'), p1 ('p'), p1/p2
    ('ratio'), 1/p1 ('inverse'), p1*p2 ('product'), p1*p2/p3 ('product_ratio') or
    (1-p1)*p2/p3 ('complement_ratio').
    """
    kind = 'linear'

    def __init__(self, name, parameters, states, terms, outputs, dosing=None):
        super().__init__(name, parameters, states, outputs, dosing)
        self.terms = list(terms)
        index = {n: j for j, n in enumerate(self.parameters)}
        rows = []
        consts = []
        for row, col, kind, names, const in self.terms:
            ids = [index[n] for n in names] + [-1] * (3 - len(names))
            rows.append([self.states.index(row), self.states.index(col), _KINDS[kind], *ids])
            consts.append(float(const))
        self.term_array = np.array(rows, dtype=np.int64).reshape(-1, 6)
        self.term_constants = np.array(consts, dtype=np.float64)
        outs = []
        for o in self.outputs:
            ids = [index[n] for n in o.pd_parameters] + [-1] * (4 - len(o.pd_parameters))
            outs.append([self.states.index(o.state), index[o.volume] if o.volume else -1, _PD[o.pd], *ids])
        self.output_array = np.array(outs, dtype=np.int64).reshape(-1, 7)


class ODEStructure(Structure):
    """Nonlinear dynamics solved numerically.

    solver: 'dopri5' (explicit, adaptive), 'rosenbrock' (L-stable, stiff systems) or
    'auto' (Dormand-Prince, switching to Rosenbrock when the step budget is exhausted).
    """
    kind = 'ode'

    def __init__(self, name, parameters, states, outputs, rhs, output=None, initial=None, dosing=None,
                 solver='auto', rtol=1e-8, atol=1e-10, max_steps=100000, ss_cycles=1000, ss_tolerance=1e-9):
        super().__init__(name, parameters, states, outputs, dosing)
        if solver not in ('dopri5', 'rosenbrock', 'auto'):
            raise ValueError('solver must be dopri5, rosenbrock or auto')
        self.method = {'dopri5': 0, 'rosenbrock': 1, 'auto': 2}[solver]
        self.rtol, self.atol, self.max_steps = float(rtol), float(atol), int(max_steps)
        self.ss_cycles, self.ss_tolerance = int(ss_cycles), float(ss_tolerance)
        self.rhs = rhs if hasattr(rhs, 'py_func') else njit(cache=False, error_model='numpy')(rhs)
        if output is None:
            output = _default_output(self)
        self.output = output if hasattr(output, 'py_func') else njit(cache=False, error_model='numpy')(output)
        if initial is None:
            initial = _zero_initial
        self.initial = initial if hasattr(initial, 'py_func') else njit(cache=False, error_model='numpy')(initial)


@njit(cache=True, error_model='numpy')
def _zero_initial(p, x):
    return None


def _default_output(structure):
    """out[k] = x[state_k] / p[volume_k] (or x[state_k]) followed by the declared PD transform."""
    index = {n: j for j, n in enumerate(structure.parameters)}
    rows = []
    for o in structure.outputs:
        ids = [index[n] for n in o.pd_parameters] + [-1] * (4 - len(o.pd_parameters))
        rows.append([structure.states.index(o.state), index[o.volume] if o.volume else -1, _PD[o.pd], *ids])
    table = np.array(rows, dtype=np.int64).reshape(-1, 7)
    from ._general.kernels import pd_transform

    @njit(error_model='numpy')
    def output(x, p, out):
        for k in range(table.shape[0]):
            s = table[k, 0]; v = table[k, 1]
            c = x[s] / p[v] if v >= 0 else x[s]
            out[k] = pd_transform(table[k, 2], c, p, table[k, 3], table[k, 4], table[k, 5], table[k, 6])
    return output


# ------------------------------------------------------------------ linear builders

def _disposition(cmt):
    if cmt == 1:
        return ('CL', 'V'), ('central',), [('central', 'central', 'ratio', ('CL', 'V'), -1.)], 'V'
    if cmt == 2:
        return (('CL', 'V1', 'Q', 'V2'), ('central', 'peripheral'),
                [('central', 'central', 'ratio', ('CL', 'V1'), -1.),
                 ('central', 'central', 'ratio', ('Q', 'V1'), -1.),
                 ('peripheral', 'central', 'ratio', ('Q', 'V1'), 1.),
                 ('peripheral', 'peripheral', 'ratio', ('Q', 'V2'), -1.),
                 ('central', 'peripheral', 'ratio', ('Q', 'V2'), 1.)], 'V1')
    if cmt == 3:
        return (('CL', 'V1', 'Q2', 'V2', 'Q3', 'V3'), ('central', 'peripheral1', 'peripheral2'),
                [('central', 'central', 'ratio', ('CL', 'V1'), -1.),
                 ('central', 'central', 'ratio', ('Q2', 'V1'), -1.),
                 ('peripheral1', 'central', 'ratio', ('Q2', 'V1'), 1.),
                 ('peripheral1', 'peripheral1', 'ratio', ('Q2', 'V2'), -1.),
                 ('central', 'peripheral1', 'ratio', ('Q2', 'V2'), 1.),
                 ('central', 'central', 'ratio', ('Q3', 'V1'), -1.),
                 ('peripheral2', 'central', 'ratio', ('Q3', 'V1'), 1.),
                 ('peripheral2', 'peripheral2', 'ratio', ('Q3', 'V3'), -1.),
                 ('central', 'peripheral2', 'ratio', ('Q3', 'V3'), 1.)], 'V1')
    raise ValueError('cmt must be 1, 2 or 3')


def pk(cmt=1, absorption=None, *, transit=0, lag=False, bioavailability=False, effect=None, pd=None,
       infusion_parameter=None):
    """Linear mammillary PK model.

    absorption: None (intravenous; CMT 1 is central), 'first_order' (depot then
    central, parameter Ka), 'zero_order' (doses into central over the modeled
    duration D1) or 'transit' (``transit`` transit compartments with mean transit
    time MTT, rate (n+1)/MTT, into a depot absorbed with Ka). Disposition
    parameters: CL, V (1 cmt); CL, V1, Q, V2 (2 cmt); CL, V1, Q2, V2, Q3, V3 (3 cmt).
    lag adds ALAG and bioavailability adds F on the dosed compartment. pd adds an
    output E: 'emax', 'sigmoid_emax', 'imax', 'sigmoid_imax' or 'linear', acting on
    the plasma concentration, or on an effect compartment concentration if
    effect='compartment' (parameter KE0). infusion_parameter='rate' or 'duration'
    adds R1 or D1 for RATE = -1 / -2 dose records into the central compartment.
    """
    params, states, terms, volume = _disposition(cmt)
    params, states = list(params), list(states)
    dose_state = 'central'
    if absorption in ('first_order', 'transit'):
        params.append('Ka')
        pre = []
        if absorption == 'transit':
            if type(transit) is not int or transit < 1:
                raise ValueError('transit absorption needs a positive integer number of transit compartments')
            params.append('MTT')
            pre = [f'transit{k}' for k in range(1, transit + 1)]
            chain = pre + ['depot']
            for a, b in zip(chain[:-1], chain[1:]):
                terms += [(a, a, 'inverse', ('MTT',), -(transit + 1.)), (b, a, 'inverse', ('MTT',), transit + 1.)]
        states = pre + ['depot'] + states
        terms += [('depot', 'depot', 'p', ('Ka',), -1.), ('central', 'depot', 'p', ('Ka',), 1.)]
        dose_state = states[0]
    elif absorption == 'zero_order':
        params.append('D1')
    elif absorption is not None:
        raise ValueError('absorption must be None, first_order, zero_order or transit')
    dose = Dosing()
    if lag:
        params.append('ALAG'); dose.lag = 'ALAG'
    if bioavailability:
        params.append('F'); dose.bioavailability = 'F'
    if absorption == 'zero_order':
        dose.duration = 'D1'; dose.zero_order = True
    dosing = {dose_state: dose}
    if infusion_parameter is not None:
        if infusion_parameter not in ('rate', 'duration') or absorption == 'zero_order':
            raise ValueError("infusion_parameter must be 'rate' or 'duration' (not with zero-order absorption)")
        name = 'R1' if infusion_parameter == 'rate' else 'D1'
        params.append(name)
        spec = dosing.get('central', Dosing())
        if infusion_parameter == 'rate':
            spec.rate = name
        else:
            spec.duration = name
        dosing['central'] = spec
    outputs = [Output('CP', 'central', volume)]
    name = f'{cmt}cmt_' + ('iv' if absorption is None else absorption)
    if pd is not None:
        pd_params = _PD_PARAMETERS[pd]
        params += list(pd_params)
        if effect == 'compartment':
            params.append('KE0')
            states.append('effect')
            terms += [('effect', 'central', 'ratio', ('KE0', volume), 1.), ('effect', 'effect', 'p', ('KE0',), -1.)]
            outputs.append(Output('E', 'effect', None, pd, pd_params))
        elif effect is None:
            outputs.append(Output('E', 'central', volume, pd, pd_params))
        else:
            raise ValueError("effect must be None or 'compartment'")
        name += '_' + pd
    return LinearStructure(name, params, states, terms, outputs, dosing)


def parent_metabolite(parent_cmt=1, absorption=None, *, metabolite_cmt=1, lag=False, bioavailability=False):
    """Parent drug and one metabolite. The parent's clearance CL is split: a fraction
    FM forms the metabolite and 1 - FM leaves by other routes (FM is usually fixed,
    for example to 1, because it is not identifiable from plasma data alone).
    Metabolite parameters: CLM, VM (1 cmt) or CLM, VM, QM, VM2 (2 cmt). Outputs CP, CM.
    """
    parent = pk(parent_cmt, absorption, lag=lag, bioavailability=bioavailability)
    params = list(parent.parameters) + ['FM']
    states = list(parent.states)
    volume = 'V' if parent_cmt == 1 else 'V1'
    terms = [t for t in parent.terms if not (t[0] == 'central' and t[1] == 'central' and t[3] == ('CL', volume))]
    terms += [('central', 'central', 'ratio', ('CL', volume), -1.),
              ('metabolite', 'central', 'product_ratio', ('FM', 'CL', volume), 1.)]
    states.append('metabolite')
    params += ['CLM', 'VM']
    terms.append(('metabolite', 'metabolite', 'ratio', ('CLM', 'VM'), -1.))
    if metabolite_cmt == 2:
        params += ['QM', 'VM2']
        states.append('metabolite_peripheral')
        terms += [('metabolite', 'metabolite', 'ratio', ('QM', 'VM'), -1.),
                  ('metabolite_peripheral', 'metabolite', 'ratio', ('QM', 'VM'), 1.),
                  ('metabolite_peripheral', 'metabolite_peripheral', 'ratio', ('QM', 'VM2'), -1.),
                  ('metabolite', 'metabolite_peripheral', 'ratio', ('QM', 'VM2'), 1.)]
    elif metabolite_cmt != 1:
        raise ValueError('metabolite_cmt must be 1 or 2')
    outputs = [Output('CP', 'central', volume), Output('CM', 'metabolite', 'VM')]
    return LinearStructure(parent.name + '_metabolite', params, states, terms, outputs, parent.dosing)


def linear(parameters, states, terms, outputs, dosing=None, name='custom_linear'):
    """A user-defined linear system; see LinearStructure for the term format."""
    outputs = [o if isinstance(o, Output) else Output(*o) for o in outputs]
    return LinearStructure(name, parameters, states, terms, outputs, dosing)


# ------------------------------------------------------------------ nonlinear builders

def _pk_block(cmt, absorption, lag, bioavailability):
    """Linear PK part used inside ODE models: parameter/state lists and A-matrix terms."""
    base = pk(cmt, absorption, lag=lag, bioavailability=bioavailability)
    return base


def _linear_rhs(base, n_extra, extra):
    """Compile rhs(t,x,p,dx) = A(p) x[:n_pk] for the PK states plus extra(t, x, p, dx)."""
    from ._general.kernels import linear_apply
    terms = base.term_array
    constants = base.term_constants
    n = len(base.states) + n_extra

    @njit(error_model='numpy')
    def rhs(t, x, p, dx):
        for i in range(n):
            dx[i] = 0.
        linear_apply(terms, constants, p, x, dx)
        extra(t, x, p, dx)
    return rhs


def michaelis_menten(cmt=1, absorption=None, *, lag=False, bioavailability=False, linear_clearance=False):
    """Saturable elimination VMAX*C/(KM + C) from the central compartment (amount/time).

    The linear clearance CL is removed unless linear_clearance=True (parallel
    linear and saturable elimination).
    """
    base = pk(cmt, absorption, lag=lag, bioavailability=bioavailability)
    volume = 'V' if cmt == 1 else 'V1'
    if not linear_clearance:
        keep = [q for q in base.parameters if q != 'CL']
        terms = [t for t in base.terms if t[3] != ('CL', volume)]
        base = LinearStructure(base.name, keep, base.states, terms, base.outputs, base.dosing)
    params = list(base.parameters) + ['VMAX', 'KM']
    index = {n: j for j, n in enumerate(params)}
    central = base.states.index('central')
    iv, ivmax, ikm = index[volume], index['VMAX'], index['KM']

    @njit(error_model='numpy')
    def extra(t, x, p, dx):
        c = x[central] / p[iv]
        dx[central] -= p[ivmax] * c / (p[ikm] + c)
    rhs = _linear_rhs(base, 0, extra)
    name = f'{cmt}cmt_mm' + ('' if absorption is None else '_' + absorption)
    return ODEStructure(name, params, base.states, [Output('CP', 'central', volume)], rhs, dosing=base.dosing)


def indirect_response(kind, pk_model=None, *, sigmoid=False):
    """Indirect response (turnover) models I-IV driven by the plasma concentration.

    dR/dt = KIN*(1 - I(C)) - KOUT*R          (I,  inhibition of production)
    dR/dt = KIN - KOUT*(1 - I(C))*R          (II, inhibition of loss)
    dR/dt = KIN*(1 + S(C)) - KOUT*R          (III, stimulation of production)
    dR/dt = KIN - KOUT*(1 + S(C))*R          (IV, stimulation of loss)
    with KIN = R0*KOUT and R(0) = R0; I(C) = IMAX*C^g/(IC50^g + C^g),
    S(C) = EMAX*C^g/(EC50^g + C^g), g = GAMMA if sigmoid else 1.
    Outputs: CP (plasma concentration) and R (response).
    """
    if kind not in (1, 2, 3, 4):
        raise ValueError('indirect response kind must be 1-4')
    base = pk_model if pk_model is not None else pk(1, 'first_order')
    if base.kind != 'linear':
        raise ValueError('the PK part must be a linear structure')
    volume = [o.volume for o in base.outputs if o.name == 'CP'][0]
    names = ['R0', 'KOUT'] + (['IMAX', 'IC50'] if kind in (1, 2) else ['EMAX', 'EC50']) + (['GAMMA'] if sigmoid else [])
    params = list(base.parameters) + names
    index = {n: j for j, n in enumerate(params)}
    central = base.states.index('central')
    r_state = len(base.states)
    iv, ir0, ikout = index[volume], index['R0'], index['KOUT']
    imax = index['IMAX'] if kind in (1, 2) else index['EMAX']
    i50 = index['IC50'] if kind in (1, 2) else index['EC50']
    igam = index['GAMMA'] if sigmoid else -1
    k_ = kind

    @njit(error_model='numpy')
    def extra(t, x, p, dx):
        c = max(x[central] / p[iv], 0.)
        g = p[igam] if igam >= 0 else 1.
        cg = c ** g
        effect = p[imax] * cg / (p[i50] ** g + cg)
        kin = p[ir0] * p[ikout]
        r = x[r_state]
        if k_ == 1:
            dx[r_state] = kin * (1. - effect) - p[ikout] * r
        elif k_ == 2:
            dx[r_state] = kin - p[ikout] * (1. - effect) * r
        elif k_ == 3:
            dx[r_state] = kin * (1. + effect) - p[ikout] * r
        else:
            dx[r_state] = kin - p[ikout] * (1. + effect) * r
    rhs = _linear_rhs(base, 1, extra)

    @njit(error_model='numpy')
    def initial(p, x):
        x[r_state] = p[ir0]
    states = list(base.states) + ['response']
    outputs = [Output('CP', 'central', volume), Output('R', 'response', None)]
    return ODEStructure(f'{base.name}_idr{kind}', params, states, outputs, rhs, initial=initial, dosing=base.dosing)


def tmdd(kind='full', cmt=1, absorption=None, *, lag=False, bioavailability=False):
    """Target-mediated drug disposition (Mager and Jusko 2001; QSS of Gibiansky et al. 2008).

    full: drug amount in central, free target R and complex RC concentrations with
    parameters CL, V (or CL, V1, Q, V2), KON, KOFF, KINT, KDEG and R0 (KSYN = R0*KDEG).
    qss: total drug and total target with KSS = (KOFF + KINT)/KON; parameters
    CL, V, KSS, KINT, KDEG, R0. Outputs: CP (free drug), RTOT (total target).
    """
    base = pk(cmt, absorption, lag=lag, bioavailability=bioavailability)
    volume = 'V' if cmt == 1 else 'V1'
    central = base.states.index('central')
    n_pk = len(base.states)
    base_terms = [t for t in base.terms if t[3] != ('CL', volume)]
    lin = LinearStructure(base.name, base.parameters, base.states, base_terms, base.outputs, base.dosing)
    if kind == 'full':
        extra_names = ['KON', 'KOFF', 'KINT', 'KDEG', 'R0']
    elif kind == 'qss':
        extra_names = ['KSS', 'KINT', 'KDEG', 'R0']
    else:
        raise ValueError("tmdd kind must be 'full' or 'qss'")
    params = list(base.parameters) + extra_names
    index = {n: j for j, n in enumerate(params)}
    icl, iv = index['CL'], index[volume]
    ikint, ikdeg, ir0 = index['KINT'], index['KDEG'], index['R0']
    if kind == 'full':
        ikon, ikoff = index['KON'], index['KOFF']
        s_r, s_rc = n_pk, n_pk + 1

        @njit(error_model='numpy')
        def extra(t, x, p, dx):
            v = p[iv]
            c = x[central] / v
            r = x[s_r]
            rc = x[s_rc]
            bind = p[ikon] * c * r - p[ikoff] * rc
            dx[central] += -p[icl] / v * x[central] - bind * v
            dx[s_r] = p[ir0] * p[ikdeg] - p[ikdeg] * r - bind
            dx[s_rc] = bind - p[ikint] * rc
        rhs = _linear_rhs(lin, 2, extra)

        @njit(error_model='numpy')
        def initial(p, x):
            x[s_r] = p[ir0]

        @njit(error_model='numpy')
        def output(x, p, out):
            out[0] = x[central] / p[iv]
            out[1] = x[s_r] + x[s_rc]
        states = list(base.states) + ['target', 'complex']
    else:
        ikss = index['KSS']
        s_rt = n_pk

        @njit(error_model='numpy')
        def free(ctot, rtot, kss):
            b = ctot - rtot - kss
            return .5 * (b + (b * b + 4. * kss * ctot) ** .5)

        @njit(error_model='numpy')
        def extra(t, x, p, dx):
            v = p[iv]
            ctot = x[central] / v
            rtot = x[s_rt]
            c = free(ctot, rtot, p[ikss])
            complex_ = ctot - c
            dx[central] += -p[icl] * c - p[ikint] * complex_ * v
            dx[s_rt] = p[ir0] * p[ikdeg] - p[ikdeg] * rtot - (p[ikint] - p[ikdeg]) * complex_
        rhs = _linear_rhs(lin, 1, extra)

        @njit(error_model='numpy')
        def initial(p, x):
            x[s_rt] = p[ir0]

        @njit(error_model='numpy')
        def output(x, p, out):
            ctot = x[central] / p[iv]
            out[0] = free(ctot, x[s_rt], p[ikss])
            out[1] = x[s_rt]
        states = list(base.states) + ['total_target']
    outputs = [Output('CP', 'central', volume), Output('RTOT', states[-1], None)]
    return ODEStructure(f'{base.name}_tmdd_{kind}', params, states, outputs, rhs, output=output, initial=initial,
                        dosing=base.dosing, solver='auto')


def ode(parameters, states, outputs, rhs, *, output=None, initial=None, dosing=None, name='custom_ode', **options):
    """A user-defined ODE model; rhs(t, x, p, dxdt) must be Numba-compilable Python."""
    outputs = [o if isinstance(o, Output) else Output(*o) for o in outputs]
    return ODEStructure(name, parameters, states, outputs, rhs, output=output, initial=initial, dosing=dosing, **options)


def from_classic(model):
    """Linear structure equivalent to a classic ModelSpec model name."""
    return {'1cmt_iv': lambda: pk(1), '1cmt_oral': lambda: pk(1, 'first_order', lag=True),
            '2cmt_iv': lambda: pk(2), '2cmt_oral': lambda: pk(2, 'first_order', lag=True)}[model]()
