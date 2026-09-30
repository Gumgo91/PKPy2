"""PKPy2: declared pharmacokinetic models and population inference.

Two estimation engines share one interface. ModelSpec declarations (one- and
two-compartment bolus/oral models with diagonal random effects) use the
original analytical engine. Model declarations (pkpy2.structures with
infusions, steady state, lag and bioavailability, three compartments,
nonlinear ODE and PK/PD models, correlated and interoccasion random effects,
time-varying and categorical covariates, lognormal residuals, censored and
multiple-output data) use the general engine. Both report an importance-sampled
marginal likelihood with an independent two-bank convergence audit.
"""
from .api import Subject, ModelSpec, Covariate, Parameter, FitResult, predict, load_fit
from .api import fit as _classic_fit
from .data import Individual, Dataset, read_nonmem
from .model import Model, Residual
from . import structures

__version__ = "0.2.0"


def fit(data, specification, **options):
    """Fit a ModelSpec (classic engine) or a Model (general engine).

    data: a list of Subjects (either engine) or a Dataset from read_nonmem (general engine).
    """
    if isinstance(specification, Model):
        from ._general.fit import fit_general
        if not isinstance(data, Dataset):
            items = list(data)
            data = Dataset(items) if items and isinstance(items[0], Individual) else Dataset.from_subjects(items)
        if options.get('saem_options') is not None:
            raise ValueError('the general engine does not use SAEM options')
        options.pop('saem_options', None)
        return fit_general(data, specification, **options)
    if isinstance(data, Dataset):
        raise ValueError('event-record Datasets need a pkpy2.Model; ModelSpec takes a list of Subjects')
    return _classic_fit(data, specification, **options)


def evaluate(data, model, **options):
    """OFV and diagnostics-ready result of a Model at its declared values, without estimation (MAXEVAL=0)."""
    from ._general.fit import evaluate_general
    if not isinstance(model, Model):
        raise ValueError('evaluate takes a pkpy2.Model')
    if not isinstance(data, Dataset):
        items = list(data)
        data = Dataset(items) if items and isinstance(items[0], Individual) else Dataset.from_subjects(items)
    return evaluate_general(data, model, **options)


def solve(structure, individual, parameters):
    """Predictions of a structure for one Individual at given parameter values (per record or constant)."""
    from ._general.packing import predict as _predict
    return _predict(structure, individual, parameters)


from ._diagnostics import diagnostics, individual_estimates
from ._simulation import simulate, vpc, simulate_data
from ._tools import (profile, bootstrap, sir, robust_covariance, likelihood_ratio_test, compare,
                     stepwise_covariates)


__all__ = ["Subject", "ModelSpec", "Covariate", "Parameter", "FitResult", "fit", "predict", "load_fit",
           "Individual", "Dataset", "read_nonmem", "Model", "Residual", "structures", "solve", "evaluate",
           "diagnostics", "individual_estimates", "simulate", "vpc", "simulate_data", "profile", "bootstrap", "sir",
           "robust_covariance", "likelihood_ratio_test", "compare", "stepwise_covariates"]
