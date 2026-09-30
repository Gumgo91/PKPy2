"""Event-record analysis data in NONMEM conventions.

An Individual holds one subject's records in time order: observations (EVID 0),
doses (EVID 1), other events such as covariate changes (EVID 2), resets (EVID 3)
and reset-and-dose records (EVID 4). Dose records carry the compartment (CMT),
amount (AMT), infusion rate (RATE: 0 bolus, >0 rate, -1 modeled rate,
-2 modeled duration), steady-state flag (SS 0/1/2) and interval (II); ADDL
additional doses are expanded when the data are read. Observation records carry
DV, the output they measure (DVID, 1-based) and optional censoring: CENS=1 means
the observation lies below DV (the quantification limit), CENS=-1 above DV, and
LIMIT gives the other end of the censoring interval. MDV=1 observations are
predicted but do not enter the likelihood. Covariates are stored per record; a
covariate that changes within a subject is time varying.
"""
from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
import numpy as np

MISSING = ('.', '', 'NA', 'NaN', 'nan', 'NULL')


@dataclass
class Individual:
    id: int | str
    time: np.ndarray
    evid: np.ndarray
    amt: np.ndarray
    cmt: np.ndarray
    rate: np.ndarray
    ii: np.ndarray
    ss: np.ndarray
    dv: np.ndarray
    mdv: np.ndarray
    dvid: np.ndarray
    cens: np.ndarray
    limit: np.ndarray
    occasion: np.ndarray | None = None
    covariates: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.time)
        if not n:
            raise ValueError(f'subject {self.id}: no records')
        cast = dict(time=float, evid=int, amt=float, cmt=int, rate=float, ii=float, ss=int, dv=float,
                    mdv=int, dvid=int, cens=int, limit=float)
        for name, kind in cast.items():
            value = np.asarray(getattr(self, name), dtype=np.float64 if kind is float else np.int64).copy()
            if value.shape != (n,):
                raise ValueError(f'subject {self.id}: column {name} has the wrong length')
            setattr(self, name, value)
        if not np.isfinite(self.time).all() or np.any(np.diff(self.time) < 0):
            raise ValueError(f'subject {self.id}: times must be finite and nondecreasing')
        if not set(np.unique(self.evid)) <= {0, 1, 2, 3, 4}:
            raise ValueError(f'subject {self.id}: EVID must be 0-4')
        dose = (self.evid == 1) | (self.evid == 4)
        if np.any(~np.isfinite(self.amt[dose])) or np.any(self.amt[dose] < 0):
            raise ValueError(f'subject {self.id}: dose amounts must be finite and nonnegative')
        if np.any(self.cmt[dose] < 1):
            raise ValueError(f'subject {self.id}: dose compartments are 1-based')
        rate = self.rate[dose]
        if np.any(~np.isfinite(rate)) or np.any((rate < 0) & (rate != -1) & (rate != -2)):
            raise ValueError(f'subject {self.id}: RATE must be 0, positive, -1 or -2')
        if np.any((self.ss < 0) | (self.ss > 2)) or np.any(self.ii < 0):
            raise ValueError(f'subject {self.id}: SS must be 0-2 and II nonnegative')
        if np.any(dose & (self.ss > 0) & (self.ii == 0) & (self.rate <= 0)):
            raise ValueError(f'subject {self.id}: steady-state bolus doses require II > 0')
        obs = self.evid == 0
        if np.any(self.dvid[obs] < 1):
            raise ValueError(f'subject {self.id}: DVID is 1-based')
        used = obs & (self.mdv == 0)
        if np.any(~np.isfinite(self.dv[used])):
            raise ValueError(f'subject {self.id}: DV missing on an observation with MDV=0')
        if not set(np.unique(self.cens)) <= {-1, 0, 1}:
            raise ValueError(f'subject {self.id}: CENS must be -1, 0 or 1')
        if self.occasion is not None:
            occ = np.asarray(self.occasion, dtype=np.int64).copy()
            if occ.shape != (n,) or np.any(occ < 1):
                raise ValueError(f'subject {self.id}: occasions must be positive integers')
            self.occasion = occ
        covariates = {}
        for name, value in self.covariates.items():
            value = np.asarray(value, dtype=np.float64).copy()
            if value.shape != (n,):
                raise ValueError(f'subject {self.id}: covariate {name} has the wrong length')
            if not np.isfinite(value).all():
                raise ValueError(f'subject {self.id}: covariate {name} is missing after filling')
            covariates[name] = value
        self.covariates = covariates

    @property
    def observed(self):
        """Records that enter the likelihood."""
        return (self.evid == 0) & (self.mdv == 0)

    def time_varying(self, name):
        value = self.covariates[name]
        return bool(np.any(value != value[0]))

    @classmethod
    def from_subject(cls, subject):
        """Records equivalent to a pkpy2.Subject: bolus doses into CMT 1 and observations of output 1."""
        history = list(subject.dose_history) if subject.dose_history else [(0., subject.dose)]
        rows = [(float(t), 1, float(a), np.nan) for t, a in history]
        rows += [(float(t), 0, 0., float(y)) for t, y in zip(np.asarray(subject.time), np.asarray(subject.obs))]
        rows.sort(key=lambda r: r[0])   # stable: at equal times doses precede observations
        n = len(rows)
        time = np.array([r[0] for r in rows]); evid = np.array([r[1] for r in rows])
        cov = {k: np.full(n, float(v)) for k, v in subject.covariates.items()}
        return cls(subject.sid, time, evid, np.array([r[2] for r in rows]), np.ones(n), np.zeros(n), np.zeros(n),
                   np.zeros(n), np.array([r[3] for r in rows]), (evid != 0).astype(int), np.ones(n), np.zeros(n),
                   np.full(n, np.nan), None, cov)


class Dataset(list):
    """A list of Individuals with the covariate and occasion columns that were read."""

    def __init__(self, individuals=(), *, covariates=None, occasion=None):
        super().__init__(individuals)
        ids = [(type(i.id).__name__, i.id) for i in self]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate subject IDs')
        names = set(covariates) if covariates is not None else set(self[0].covariates) if self else set()
        for i in self:
            if set(i.covariates) != names:
                raise ValueError('every subject must carry the same covariates')
        self.covariate_names = tuple(sorted(names))
        self.occasion_column = occasion

    @property
    def time_varying(self):
        return {name: any(i.time_varying(name) for i in self) for name in self.covariate_names}

    @property
    def n_observations(self):
        return int(sum(int(np.sum(i.observed)) for i in self))

    def subset(self, ids):
        chosen = [i for i in self if i.id in set(ids)]
        return Dataset(chosen, covariates=self.covariate_names, occasion=self.occasion_column)

    def resample(self, rng):
        """Case bootstrap: sample subjects with replacement and relabel them 1..N."""
        import copy
        picks = rng.integers(0, len(self), len(self))
        out = []
        for k, j in enumerate(picks, start=1):
            clone = copy.deepcopy(self[j])
            clone.id = k
            out.append(clone)
        return Dataset(out, covariates=self.covariate_names, occasion=self.occasion_column)

    @classmethod
    def from_subjects(cls, subjects):
        return cls([Individual.from_subject(s) for s in subjects])


def _columns(source):
    """Return {column: list of raw values} from a CSV path, a mapping, or a pandas DataFrame."""
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding='utf-8-sig').splitlines()
        if text and text[0].startswith('#'):
            text[0] = text[0][1:]
        rows = list(csv.DictReader(text))
        if not rows:
            raise ValueError('no data rows')
        return {k.strip(): [r[k] for r in rows] for k in rows[0]}
    if hasattr(source, 'to_dict'):
        return {str(k): list(v) for k, v in source.to_dict('list').items()}
    return {str(k): list(v) for k, v in dict(source).items()}


def _number(value):
    if value is None:
        return math.nan
    if isinstance(value, str):
        value = value.strip()
        if value in MISSING:
            return math.nan
    return float(value)


def read_nonmem(source, *, covariates=(), occasion=None, id='ID', time='TIME', amt='AMT', dv='DV',
                evid='EVID', mdv='MDV', cmt='CMT', rate='RATE', ii='II', ss='SS', addl='ADDL', dvid='DVID',
                cens='CENS', limit='LIMIT', covariate_fill='locf'):
    """Read NONMEM-style event records into a Dataset.

    Missing optional columns take their NONMEM defaults. EVID is derived from AMT
    when absent (dose if AMT > 0); MDV is 1 for non-observation records and for
    observations without DV. Records are sorted by time within subject, keeping the
    data order at equal times. ADDL doses are expanded at TIME + k*II. Covariate
    values missing on some records are filled within subject, last observation
    carried forward and then backward (covariate_fill='locf'), or next observation
    carried backward first ('nocb').
    """
    cols = _columns(source)
    if id not in cols or time not in cols:
        raise ValueError('ID and TIME columns are required')
    n = len(cols[id])

    def column(name, default):
        if name is None or name not in cols:
            return np.full(n, default, dtype=float)
        return np.array([_number(v) for v in cols[name]], dtype=float)
    raw_id = cols[id]
    t = column(time, math.nan)
    a = column(amt, 0.)
    a[np.isnan(a)] = 0.
    y = column(dv, math.nan)
    e = column(evid, math.nan)
    if np.all(np.isnan(e)):
        e = np.where(a > 0, 1., 0.)
    elif np.any(np.isnan(e)):
        raise ValueError('EVID has missing values')
    m = column(mdv, math.nan)
    derived_mdv = ((e != 0) | np.isnan(y)).astype(float)
    m = np.where(np.isnan(m), derived_mdv, np.maximum(m, (e != 0).astype(float)))
    c = column(cmt, 1.); c[np.isnan(c)] = 1.
    r = column(rate, 0.); r[np.isnan(r)] = 0.
    i_ = column(ii, 0.); i_[np.isnan(i_)] = 0.
    s = column(ss, 0.); s[np.isnan(s)] = 0.
    ad = column(addl, 0.); ad[np.isnan(ad)] = 0.
    d_ = column(dvid, 1.); d_[np.isnan(d_)] = 1.
    ce = column(cens, 0.); ce[np.isnan(ce)] = 0.
    li = column(limit, math.nan)
    oc = column(occasion, math.nan) if occasion else None
    cov = {}
    for name in covariates:
        if name not in cols:
            raise ValueError(f'covariate column {name} not found')
        cov[name] = column(name, math.nan)
    if np.any(np.isnan(t)):
        raise ValueError('TIME has missing values')

    def key(v):
        if isinstance(v, str):
            v = v.strip()
            try:
                f = float(v)
                return int(f) if f == int(f) else v
            except ValueError:
                return v
        f = float(v)
        return int(f) if f == int(f) else f
    ids = [key(v) for v in raw_id]
    order = {}
    for j, k in enumerate(ids):
        order.setdefault(k, []).append(j)
    individuals = []
    for k, rows in order.items():
        records = []
        for j in rows:
            base = dict(time=t[j], evid=int(e[j]), amt=a[j], cmt=int(c[j]), rate=r[j], ii=i_[j], ss=int(s[j]),
                        dv=y[j], mdv=int(m[j]), dvid=int(d_[j]), cens=int(ce[j]), limit=li[j],
                        occ=(oc[j] if oc is not None else math.nan), cov={nm: v[j] for nm, v in cov.items()})
            records.append(base)
            extra = int(ad[j])
            if extra > 0 and base['evid'] in (1, 4):
                if i_[j] <= 0:
                    raise ValueError(f'subject {k}: ADDL requires II > 0')
                for q in range(1, extra + 1):
                    records.append(dict(base, time=t[j] + q * i_[j], evid=1, ss=0))
        records.sort(key=lambda q: q['time'])    # stable
        names = list(cov)
        values = {nm: np.array([q['cov'][nm] for q in records]) for nm in names}
        for nm in names:
            v = values[nm]
            if covariate_fill == 'nocb':
                v = _fill(v[::-1])[::-1]
            v = _fill(v)
            v = _fill(v[::-1])[::-1]
            values[nm] = v
        occ = None
        if oc is not None:
            ov = _fill(np.array([q['occ'] for q in records]))
            ov = _fill(ov[::-1])[::-1]
            if np.any(np.isnan(ov)):
                raise ValueError(f'subject {k}: occasion missing on every record')
            occ = ov.astype(int)
        individuals.append(Individual(
            k, np.array([q['time'] for q in records]), np.array([q['evid'] for q in records]),
            np.array([q['amt'] for q in records]), np.array([q['cmt'] for q in records]),
            np.array([q['rate'] for q in records]), np.array([q['ii'] for q in records]),
            np.array([q['ss'] for q in records]), np.array([q['dv'] for q in records]),
            np.array([q['mdv'] for q in records]), np.array([q['dvid'] for q in records]),
            np.array([q['cens'] for q in records]), np.array([q['limit'] for q in records]), occ, values))
    return Dataset(individuals, covariates=tuple(covariates), occasion=occasion)


def _fill(values):
    out = np.array(values, dtype=float, copy=True)
    last = math.nan
    for j in range(len(out)):
        if np.isnan(out[j]):
            out[j] = last
        else:
            last = out[j]
    return out
