"""Quick independent checks of the event-record interface (run in continuous integration).

1. Linear predictions with infusions, steady state, lag and bioavailability against an
   independent SciPy matrix-exponential calculation (steady state by repeating the interval).
2. The PK output of an ODE (indirect-response) model against the exact linear solution.
3. The importance-sampled marginal OFV of a model with M3 censoring against adaptive
   Gauss-Hermite quadrature over its single random effect.
4. A small population fit converges and passes the two-bank audit.
Exits with an error if any check fails.
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

import pkpy2
from pkpy2 import Model, Residual, Parameter as P, structures as S
from pkpy2.data import Individual

FAILURES = []


def check(name, value, limit):
    ok = value <= limit
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} (limit {limit:.0e})")
    if not ok:
        FAILURES.append(name)


def individual(rows, dvid=None):
    n = len(rows)
    t, evid, amt, cmt, rate, ii, ss = (np.array([r[k] for r in rows], dtype=float) for k in range(7))
    obs = evid == 0
    return Individual(1, t, evid, amt, cmt, rate, ii, ss, np.where(obs, 1., np.nan), (~obs).astype(int),
                      np.ones(n) if dvid is None else dvid, np.zeros(n), np.full(n, np.nan))


# ------------------------------------------------------------------ 1. linear predictions
def reference_two_compartment(theta, doses, times):
    """Amounts in central/peripheral by stepwise exp(A dt) with infusions via the augmented matrix."""
    cl, v1, q, v2 = theta
    a = np.array([[-(cl + q) / v1, q / v2], [q / v1, -q / v2]])

    def advance(x, dt, r):
        m = np.zeros((3, 3))
        m[:2, :2] = a * dt
        m[:2, 2] = np.array([r, 0.]) * dt
        e = expm(m)
        return e[:2, :2] @ x + e[:2, 2]

    events = sorted({0.} | {d[0] for d in doses} | {d[0] + d[1] / d[2] for d in doses} | set(times))
    x, out, t = np.zeros(2), {}, 0.
    for te in events:
        rate = sum(d[2] for d in doses if d[0] <= t < d[0] + d[1] / d[2] - 1e-12)
        x = advance(x, te - t, rate)
        t = te
        if te in times:
            out[te] = x[0] / v1
    return np.array([out[s] for s in times])


def check_linear():
    theta = (4.1, 22., 7.5, 51.)
    # steady state (SS=1) of 500 mg over 2 h every 12 h, then a 2-h infusion of 800 mg at 36 h
    rows = [(0., 1, 500., 1, 250., 12., 1), (36., 1, 800., 1, 400., 0., 0)]
    times = [.5, 1., 2., 5., 11.9, 12.5, 24.5, 36.5, 38., 48.]
    rows += [(s, 0, 0., 1, 0., 0., 0) for s in times]
    rows.sort(key=lambda r: (r[0], r[1] != 1))
    _, _, pred = pkpy2.solve(S.pk(2), individual(rows), dict(CL=theta[0], V1=theta[1], Q=theta[2], V2=theta[3]))
    # reference: the regimen given for 80 intervals up to and including time 0 (steady state), no later
    # regimen doses (SS = 1 without ADDL), then the single infusion at 36 h; times shifted by 960 h
    doses = [(-12. * k, 500., 250.) for k in range(80)] + [(36., 800., 400.)]
    ref = reference_two_compartment(theta, [(d[0] + 960., d[1], d[2]) for d in doses], [s + 960. for s in times])
    check('two-compartment steady-state infusion vs SciPy expm', float(np.max(np.abs(pred - ref) / ref)), 1e-10)

    # oral with lag and bioavailability: closed form of the one-compartment model
    cl, v, ka, lag, f = 2.7, 33., 1.4, .37, .8
    rows = [(0., 1, 100., 1, 0., 0., 0)] + [(s, 0, 0., 2, 0., 0., 0) for s in (.2, .5, 1., 3., 8., 24.)]
    _, _, pred = pkpy2.solve(S.pk(1, 'first_order', lag=True, bioavailability=True), individual(rows),
                             dict(CL=cl, V=v, Ka=ka, ALAG=lag, F=f))
    k = cl / v
    t = np.array([.2, .5, 1., 3., 8., 24.]) - lag
    ref = np.where(t > 0, f * 100. * ka / (v * (ka - k)) * (np.exp(-k * np.maximum(t, 0)) - np.exp(-ka * np.maximum(t, 0))), 0.)
    check('oral lag and bioavailability vs closed form', float(np.max(np.abs(pred - ref)) / np.max(ref)), 1e-12)


# ------------------------------------------------------------------ 2. ODE against linear
def check_ode():
    rows = [(0., 1, 150., 1, 0., 0., 0)] + [(s, 0, 0., 1, 0., 0., 0) for s in (1., 3., 8., 24., 48.)]
    ind = individual(rows)
    values = dict(CL=2., V=25., Ka=1.3, R0=80., KOUT=.15, EMAX=1.5, EC50=1.)
    _, _, ode = pkpy2.solve(S.indirect_response(3), ind, values)
    _, _, lin = pkpy2.solve(S.pk(1, 'first_order'), ind, dict(CL=2., V=25., Ka=1.3))
    check('ODE (indirect response) PK output vs exact linear solution', float(np.max(np.abs(ode - lin) / lin)), 1e-7)


# ------------------------------------------------------------------ 3. marginal likelihood with censoring
def check_likelihood():
    rng = np.random.default_rng(3)
    lines = ['ID,TIME,EVID,AMT,CMT,DV']
    for i in range(1, 9):
        lines.append(f'{i},0,1,100,1,.')
        lines += [f'{i},{s},0,0,1,.' for s in (1, 4, 12, 24)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'design.csv'
        path.write_text('\n'.join(lines))
        design = pkpy2.read_nonmem(path)
    model = Model(S.pk(1), theta=dict(CL=P(4.), V=P(40.)), omega=dict(CL=P(.09)), residual=Residual(proportional=P(.15)))
    data = pkpy2.simulate_data(model, design, seed=int(rng.integers(1, 1000)), lloq={'CP': .3})
    result = pkpy2.evaluate(data, model, power=14)
    total = 0.
    nodes, weights = hermegauss(40)
    for ind in data:
        obs = ind.evid == 0
        t, y, cens = ind.time[obs], ind.dv[obs], ind.cens[obs]

        def joint(eta):
            cl = 4. * math.exp(eta)
            f = 100. / 40. * np.exp(-cl / 40. * t)
            sd = .15 * f
            z = (y - f) / sd
            ll = np.where(cens == 1, log_ndtr(z), -.5 * z * z - np.log(sd) - .5 * math.log(2 * math.pi))
            return -(ll.sum() - .5 * eta * eta / .09 - .5 * math.log(2 * math.pi * .09))
        mode = minimize_scalar(joint, bounds=(-3, 3), method='bounded', options=dict(xatol=1e-10)).x
        h = 1e-4
        curvature = (joint(mode + h) - 2 * joint(mode) + joint(mode - h)) / h ** 2
        s = 1. / math.sqrt(curvature)
        values = np.array([-joint(mode + s * u) for u in nodes])
        top = values.max()
        integral = s * np.sum(weights * np.exp(values - top + .5 * nodes ** 2))
        total += -2. * (top + math.log(integral))
    check('marginal OFV with M3 censoring vs Gauss-Hermite quadrature', abs(result.ofv - total), .02)


# ------------------------------------------------------------------ 4. a small fit
def check_fit():
    lines = ['ID,TIME,EVID,AMT,CMT,RATE,DV']
    for i in range(1, 13):
        lines.append(f'{i},0,1,500,1,250,.')
        lines += [f'{i},{s},0,0,1,0,.' for s in (1, 2, 4, 8, 12, 24)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'design.csv'
        path.write_text('\n'.join(lines))
        design = pkpy2.read_nonmem(path)
    truth = Model(S.pk(1), theta=dict(CL=P(5.), V=P(40.)), omega=dict(CL=P(.09), V=P(.04)),
                  residual=Residual(proportional=P(.12)))
    data = pkpy2.simulate_data(truth, design, seed=7)
    start = Model(S.pk(1), theta=dict(CL=P(3.), V=P(30.)), omega=dict(CL=P(.1), V=P(.1)),
                  residual=Residual(proportional=P(.2)))
    result = pkpy2.fit(data, start, seed=1)
    print(f'      fit status {result.status}, OFV {result.ofv:.3f}, {result.seconds:.0f} s')
    check('small infusion fit converged (0 = yes)', 0. if result.converged else 1., 0.)


if __name__ == '__main__':
    check_linear()
    check_ode()
    check_likelihood()
    check_fit()
    if FAILURES:
        sys.exit(f'failed: {FAILURES}')
    print('all event-record checks passed')
