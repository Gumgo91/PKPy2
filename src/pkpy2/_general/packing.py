"""Subject records packed for a structure, and direct prediction entry points."""
from dataclasses import dataclass
import numba
import numpy as np
from . import kernels as K

SERIAL_BATCH = 32


@dataclass
class SubjectArrays:
    id: object
    rec_time: np.ndarray
    rec_evid: np.ndarray
    rec_amt: np.ndarray
    rec_cmt: np.ndarray
    rec_rate: np.ndarray
    rec_ii: np.ndarray
    rec_ss: np.ndarray
    rec_obs: np.ndarray
    rec_out: np.ndarray
    rec_occ: np.ndarray
    n_obs: int
    obs_time: np.ndarray
    obs_dv: np.ndarray
    obs_out: np.ndarray
    obs_used: np.ndarray
    obs_cens: np.ndarray
    obs_limit: np.ndarray
    obs_record: np.ndarray
    occasions: tuple

    @property
    def n_records(self):
        return len(self.rec_time)


def pack_individual(individual, structure):
    n_state = len(structure.states)
    n_out = len(structure.outputs)
    evid = individual.evid
    dose = (evid == 1) | (evid == 4)
    cmt = np.where(dose, individual.cmt - 1, 0).astype(np.int64)
    if np.any(cmt[dose] >= n_state):
        raise ValueError(f'subject {individual.id}: dose CMT exceeds the {n_state} states of {structure.name}')
    obs = evid == 0
    out = np.where(obs, individual.dvid - 1, 0).astype(np.int64)
    if np.any(out[obs] >= n_out):
        raise ValueError(f'subject {individual.id}: DVID exceeds the {n_out} outputs of {structure.name}')
    rec_obs = np.full(len(evid), -1, dtype=np.int64)
    rec_obs[obs] = np.arange(int(obs.sum()))
    if individual.occasion is not None:
        labels = tuple(dict.fromkeys(int(v) for v in individual.occasion))
        lookup = {v: j for j, v in enumerate(labels)}
        rec_occ = np.array([lookup[int(v)] for v in individual.occasion], dtype=np.int64)
    else:
        labels = ()
        rec_occ = np.full(len(evid), -1, dtype=np.int64)
    used = (individual.mdv[obs] == 0)
    return SubjectArrays(
        individual.id, individual.time.astype(np.float64), evid.astype(np.int64), individual.amt.astype(np.float64),
        cmt, individual.rate.astype(np.float64), individual.ii.astype(np.float64), individual.ss.astype(np.int64),
        rec_obs, out, rec_occ, int(obs.sum()), individual.time[obs].astype(np.float64),
        np.where(np.isfinite(individual.dv[obs]), individual.dv[obs], 0.), out[obs], used,
        individual.cens[obs].astype(np.int64), individual.limit[obs].astype(np.float64),
        np.nonzero(obs)[0].astype(np.int64), labels)


def kernel_arguments(structure):
    """Structure-level arguments shared by every subject."""
    f, lag, dur, rate, auto = structure.dosing_arrays()
    if structure.kind == 'linear':
        return ('linear', structure.term_array, structure.term_constants, structure.output_array,
                len(structure.states), f, lag, dur, rate, auto)
    return ('ode', structure.rhs, structure.output, structure.initial, structure.method, len(structure.states),
            len(structure.outputs), structure.rtol, structure.atol, structure.max_steps, structure.ss_cycles,
            structure.ss_tolerance, f, lag, dur, rate, auto)


def batch_predict(structure_args, subject, base, tv, kappa, iov_param, transform):
    """Predictions (K, n_obs) and error codes (K,) for K particles of one subject."""
    s = subject
    base = np.ascontiguousarray(base, dtype=np.float64)
    tv = np.ascontiguousarray(tv, dtype=np.float64)
    kappa = np.ascontiguousarray(kappa, dtype=np.float64)
    if structure_args[0] == 'linear':
        _, terms, constants, outputs, n, f, lag, dur, rate, auto = structure_args
        # small batches (mode searches) run serially: a parallel launch costs more than they do
        args = (terms, constants, outputs, n, f, lag, dur, rate, auto, s.rec_time, s.rec_evid, s.rec_amt, s.rec_cmt,
                s.rec_rate, s.rec_ii, s.rec_ss, s.rec_obs, s.rec_out, s.n_obs, base, tv, kappa, iov_param, s.rec_occ,
                transform)
        if base.shape[0] <= SERIAL_BATCH:
            return K.linear_batch_serial(*args)
        return K.linear_batch(*args, 4 * numba.get_num_threads())
    (_, rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol,
     f, lag, dur, rate, auto) = structure_args
    args = (rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol, f, lag, dur, rate, auto,
            s.rec_time, s.rec_evid, s.rec_amt, s.rec_cmt, s.rec_rate, s.rec_ii, s.rec_ss, s.rec_obs, s.rec_out, s.n_obs,
            base, tv, kappa, iov_param, s.rec_occ, transform)
    if base.shape[0] <= 2:
        return K.ode_batch_serial(*args)
    return K.ode_batch(*args, 2 * numba.get_num_threads())


def predict(structure, individual, parameters):
    """Predictions at the observation records of one individual.

    parameters: {name: value} (constant) or {name: array over records}. Returns
    (times, outputs, predictions) for every EVID 0 record.
    """
    s = pack_individual(individual, structure)
    names = structure.parameters
    missing = [n for n in names if n not in parameters]
    if missing:
        raise ValueError(f'missing structural parameters: {missing}')
    rows = np.empty((s.n_records, len(names)))
    for j, n in enumerate(names):
        rows[:, j] = np.broadcast_to(np.asarray(parameters[n], dtype=float), (s.n_records,))
    if np.any(rows <= 0) and not np.all(np.isfinite(rows)):
        raise ValueError('parameters must be finite')
    identity = np.full(len(names), 2, dtype=np.int64)
    base = np.zeros((1, len(names)))
    kappa = np.zeros((1, 1, 0))
    pred, errors = batch_predict(kernel_arguments(structure), s, base, rows, kappa, np.zeros(0, dtype=np.int64),
                                 identity)
    if errors[0]:
        raise RuntimeError(f'prediction failed with error code {int(errors[0])}')
    return s.obs_time, s.obs_out, pred[0]
