"""General model declarations for the extended estimation engine.

A Model combines a structure (pkpy2.structures) with a statistical model:

* theta: typical values of every structural parameter. Parameters are log-normal
  by default; transforms may set 'logit' (fractions such as F) or 'identity'.
  Covariate effects and random effects act on the transformed scale.
* omega: interindividual variances. omega_blocks lists groups of parameters
  whose random effects are correlated (full covariance within the block);
  omega_covariance gives initial (or fixed) covariances within blocks.
* iov: interoccasion variances; occasions come from the data (read_nonmem
  occasion=...). An occasion effect applies to the records of that occasion.
* covariates: Covariate effects of any form (power, exponential, linear,
  categorical). A covariate that changes within a subject is used per record.
* residual: a Residual per observed output (keyed by output name), additive
  and/or proportional (combined variance prop^2 f^2 + add^2) or lognormal
  (log y ~ N(log f, sd^2), i.e. log-transform-both-sides). Observations flagged
  as censored (CENS) contribute the probability of the censoring interval (M3/M4).
"""
from dataclasses import dataclass, field
from .api import Parameter, Covariate
from .structures import Structure


@dataclass(frozen=True)
class Residual:
    additive: Parameter | None = None
    proportional: Parameter | None = None
    lognormal: Parameter | None = None

    def __post_init__(self):
        if self.lognormal is not None and (self.additive is not None or self.proportional is not None):
            raise ValueError('a lognormal residual cannot be combined with additive/proportional terms')
        if self.lognormal is None and self.additive is None and self.proportional is None:
            raise ValueError('a residual model needs at least one component')
        for name in ('additive', 'proportional', 'lognormal'):
            p = getattr(self, name)
            if p is not None and (not isinstance(p, Parameter) or p.value < 0):
                raise ValueError(f'{name} residual SD must be a nonnegative Parameter')

    @property
    def form(self):
        return 'lognormal' if self.lognormal is not None else 'combined'

    def components(self):
        """(name, Parameter) for every declared component; missing combined terms are fixed at zero."""
        if self.lognormal is not None:
            return [('lognormal', self.lognormal)]
        return [('proportional', self.proportional or Parameter(0., True)),
                ('additive', self.additive or Parameter(0., True))]


@dataclass
class Model:
    structure: Structure
    theta: dict
    omega: dict = field(default_factory=dict)
    omega_blocks: tuple = ()
    omega_covariance: dict = field(default_factory=dict)
    iov: dict = field(default_factory=dict)
    covariates: tuple = ()
    residual: object = None
    transforms: dict = field(default_factory=dict)
    omega_floor: float = 1e-8

    def __post_init__(self):
        s = self.structure
        if not isinstance(s, Structure):
            raise ValueError('structure must be a pkpy2.structures structure')
        names = set(s.parameters)
        if set(self.theta) != names:
            raise ValueError(f'theta must specify exactly the structural parameters {s.parameters}; '
                             f'missing {sorted(names - set(self.theta))}, unknown {sorted(set(self.theta) - names)}')
        for k, v in list(self.theta.items()):
            if not isinstance(v, Parameter):
                self.theta[k] = Parameter(float(v))
        for group in (self.omega, self.iov):
            for k, v in list(group.items()):
                if k not in names:
                    raise ValueError(f'random effect on unknown parameter {k}')
                if not isinstance(v, Parameter):
                    group[k] = Parameter(float(v))
                if group[k].value < self.omega_floor:
                    raise ValueError(f'variance of {k} is below the omega floor')
        seen = set()
        blocks = []
        for block in self.omega_blocks:
            block = tuple(block)
            if len(block) < 2 or any(b not in self.omega for b in block) or seen & set(block):
                raise ValueError('omega blocks need at least two distinct parameters with IIV, each in one block')
            if len({self.omega[b].fixed for b in block}) != 1:
                raise ValueError('all variances of an omega block must be fixed or all estimated')
            if any(self.omega[b].lower is not None or self.omega[b].upper is not None for b in block):
                raise ValueError('bounds are not supported on variances inside omega blocks')
            seen |= set(block)
            blocks.append(block)
        self.omega_blocks = tuple(blocks)
        for pair, value in list(self.omega_covariance.items()):
            a, b = pair
            if not any(a in blk and b in blk for blk in self.omega_blocks):
                raise ValueError(f'covariance {pair} is not inside a declared omega block')
            if not isinstance(value, Parameter):
                self.omega_covariance[pair] = Parameter(float(value))
        for c in self.covariates:
            if not isinstance(c, Covariate) or c.parameter not in names:
                raise ValueError('covariate effects must be Covariate objects on structural parameters')
        for k, t in self.transforms.items():
            if k not in names or t not in ('log', 'logit', 'identity'):
                raise ValueError(f'invalid transform {t} for {k}')
        if self.residual is None:
            self.residual = Residual(proportional=Parameter(.2))
        if isinstance(self.residual, Residual):
            self.residual = {s.outputs[0].name: self.residual}
        for name, r in self.residual.items():
            if name not in s.output_names:
                raise ValueError(f'residual for unknown output {name}; outputs are {s.output_names}')
            if not isinstance(r, Residual):
                raise ValueError('residual entries must be Residual objects')

    def transform(self, name):
        return self.transforms.get(name, 'log')
