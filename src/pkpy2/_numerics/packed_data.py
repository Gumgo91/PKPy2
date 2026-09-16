"""Owned array input boundary for the experimental numerical engine.

Single and repeated doses share one explicit event representation. Packing is
performed once per analysis; no Subject objects cross the numerical boundary.
"""
from dataclasses import dataclass
import numpy as np


def _owned(values, dtype):
    array = np.array(values, dtype=dtype, order='C', copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class PackedStudy:
    model: str
    subject_ids: tuple
    observation_offsets: np.ndarray
    time: np.ndarray
    observation: np.ndarray
    dose_offsets: np.ndarray
    dose_time: np.ndarray
    dose_amount: np.ndarray
    dose_duration: np.ndarray

    def __post_init__(self):
        if self.model not in {'1cmt_iv','1cmt_oral','2cmt_iv','2cmt_oral','2cmt_inf'}:
            raise ValueError('unsupported packed model')
        ids=tuple(self.subject_ids)
        if not ids or any(type(sid) not in (int,str) or sid=='' for sid in ids):
            raise ValueError('nonempty typed subject IDs required')
        if len({(type(sid),sid) for sid in ids})!=len(ids):
            raise ValueError('duplicate packed subject IDs')
        object.__setattr__(self,'subject_ids',ids)
        for name in ('time','observation','dose_time','dose_amount','dose_duration'):
            value=_owned(getattr(self,name),np.float64)
            if value.ndim!=1 or not np.isfinite(value).all():
                raise ValueError('packed arrays must be finite one-dimensional values: '+name)
            object.__setattr__(self,name,value)
        if len(self.observation)!=len(self.time):
            raise ValueError('observation/time lengths differ')
        if len(self.dose_time)!=len(self.dose_amount) or len(self.dose_duration)!=len(self.dose_amount):
            raise ValueError('dose event array lengths differ')
        for name,extent in [('observation_offsets',len(self.time)),('dose_offsets',len(self.dose_time))]:
            raw=np.asarray(getattr(self,name))
            if (raw.ndim!=1 or raw.dtype.kind not in 'iu' or len(raw)!=len(ids)+1
                    or raw[0]!=0 or raw[-1]!=extent or np.any(raw[1:]<=raw[:-1])):
                raise ValueError('invalid packed offsets: '+name)
            object.__setattr__(self,name,_owned(raw,np.int64))
        if np.any(self.dose_amount<0) or np.any(self.dose_duration<0):
            raise ValueError('negative packed dosing values')
        if self.model!='2cmt_inf' and np.any(self.dose_duration!=0):
            raise ValueError('infusion duration conflicts with packed model')
        if self.model=='2cmt_inf' and np.any(np.diff(self.dose_offsets)!=1):
            raise ValueError('packed infusion histories are unsupported')


def pack_study(subjects, model):
    """Snapshot finite fitting data without sorting or mutating the caller's data.

    A nonempty dose_history is authoritative, as in the existing public API.
    An absent/empty history becomes a single event at time zero. Infusion
    histories need an explicit duration-aware public contract before support.
    """
    if model not in {'1cmt_iv', '1cmt_oral', '2cmt_iv', '2cmt_oral', '2cmt_inf'}:
        raise ValueError('unsupported packed model')
    subjects = list(subjects)
    if not subjects:
        raise ValueError('at least one subject required')
    ids, tokens, offsets, dose_offsets = [], set(), [0], [0]
    times, observations, dose_times, amounts, durations = [], [], [], [], []
    for subject in subjects:
        sid = subject.sid.item() if isinstance(subject.sid, np.generic) else subject.sid
        if type(sid) not in (int, str) or (isinstance(sid, str) and not sid):
            raise ValueError('packed IDs must be integers or nonempty strings')
        token = (type(sid).__name__, sid)
        if token in tokens:
            raise ValueError('duplicate subject ID')
        tokens.add(token)
        ids.append(sid)
        t = np.asarray(subject.time, dtype=np.float64)
        y = np.asarray(subject.obs, dtype=np.float64)
        if t.ndim != 1 or not t.size or y.shape != t.shape:
            raise ValueError('nonempty matching one-dimensional observations required')
        if not np.isfinite(t).all() or not np.isfinite(y).all():
            raise ValueError('fitting times and observations must be finite')
        duration = float(getattr(subject, 'tinf', 0.))
        if not np.isfinite(duration) or duration < 0:
            raise ValueError('invalid infusion duration')
        if model != '2cmt_inf' and duration != 0:
            raise ValueError('infusion duration conflicts with model')
        history = getattr(subject, 'dose_history', None)
        events = list(history) if history is not None else []
        if events and model == '2cmt_inf':
            raise ValueError('infusion histories require duration-aware events')
        if not events:
            events = [(0., subject.dose)]
        for event in events:
            if len(event) != 2:
                raise ValueError('dose events require time and amount')
            start, amount = map(float, event)
            if not np.isfinite(start) or not np.isfinite(amount) or amount < 0:
                raise ValueError('invalid dose event')
            dose_times.append(start)
            amounts.append(amount)
            durations.append(duration)
        times.extend(t)
        observations.extend(y)
        offsets.append(len(times))
        dose_offsets.append(len(amounts))
    return PackedStudy(model, tuple(ids), offsets, times, observations,
                       dose_offsets, dose_times, amounts, durations)
