"""Compiled event-record prediction for general structural models.

Every subject is a time-ordered list of records. For each particle (one set of
individual parameters per record) the records are expanded into dynamic events:
parameter-row switches at record times (last observation carried forward),
observations, bolus doses and infusion starts at the record time plus the
compartment lag, infusion ends, resets and steady-state dosing. Bioavailability
scales the dose amount; for RATE > 0 the rate is kept and the duration follows
(NONMEM convention). RATE = -1 and -2 take the rate or duration from model
parameters. Linear systems are advanced exactly: in the eigenbasis of the system
matrix when its eigenvalues are real and distinct, otherwise with matrix
exponentials (the Van Loan block form gives the infusion integral); steady states
are solved in closed form. Nonlinear systems are integrated with an adaptive Dormand-Prince
5(4) method or, for stiff systems, a Rosenbrock (ode23s) method; their steady
states are reached by repeating the dosing cycle to convergence.
"""
import math
import numpy as np
from numba import njit, prange

# Dynamic event types
ROW, OBS, BOLUS, INF_ON, INF_OFF, RESET, SS1, SS2, CONT1, CONT2 = range(10)
_PRIORITY = np.array([0, 6, 4, 5, 3, 1, 2, 2, 2, 2], dtype=np.int64)

# Kernels use NumPy floating-point semantics (error_model="numpy"): a division by zero at an extreme
# parameter value gives inf or nan, which the likelihood treats as an invalid point. Kernels that take a
# model right-hand side as an argument are compiled per process (their argument types are not cacheable).
# Error codes returned by the drivers
OK, E_LAG_SS, E_DURATION, E_RATE, E_ODE, E_SS_ODE, E_SINGULAR = 0, 1, 2, 3, 4, 5, 6


@njit(cache=True, nogil=True, error_model='numpy')
def expm(a):
    """Matrix exponential by scaling and squaring with a (6,6) Pade approximant."""
    n = a.shape[0]
    norm = 0.
    for j in range(n):
        column = 0.
        for i in range(n):
            column += abs(a[i, j])
        if column > norm:
            norm = column
    s = 0
    if norm > .5:
        s = int(math.ceil(math.log2(norm / .5)))
    b = a / (2. ** s)
    ident = np.eye(n)
    b2 = b @ b
    b4 = b2 @ b2
    b6 = b4 @ b2
    u = b @ (.5 * ident + (1. / 66.) * b2 + (1. / 15840.) * b4)
    v = ident + (5. / 44.) * b2 + (1. / 792.) * b4 + (1. / 665280.) * b6
    e = np.ascontiguousarray(np.linalg.solve(v - u, v + u))
    for _ in range(s):
        e = e @ e
    return e


@njit(cache=True, nogil=True, error_model='numpy')
def _matvec(a, x):
    n = a.shape[0]
    out = np.zeros(n)
    for i in range(n):
        acc = 0.
        for j in range(a.shape[1]):
            acc += a[i, j] * x[j]
        out[i] = acc
    return out


@njit(cache=True, nogil=True, error_model='numpy')
def expm_input(a, b, dt):
    """exp(A dt) and the integral of exp(A s) b over [0, dt] (Van Loan block form)."""
    n = a.shape[0]
    m = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            m[i, j] = a[i, j] * dt
        m[i, n] = b[i] * dt
    e = expm(m)
    return e[:n, :n].copy(), e[:n, n].copy()


@njit(cache=True, nogil=True, error_model='numpy')
def build_events(rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, params,
                 f_idx, lag_idx, dur_idx, rate_idx, auto_dur):
    """Dynamic events for one particle, sorted by (time, record, priority).

    Returns (count, error, time, record, type, cmt, value, rate, duration, lag, id).
    `rate` holds the infusion rate of steady-state events and `id` links infusion
    starts and ends so that resets cancel pending ends.
    """
    nrec = len(rec_time)
    cap = 6 * nrec + 1
    et = np.empty(cap); er = np.empty(cap, dtype=np.int64); ep = np.empty(cap, dtype=np.int64)
    ety = np.empty(cap, dtype=np.int64); ec = np.empty(cap, dtype=np.int64); ev = np.empty(cap)
    erate = np.zeros(cap); edur = np.zeros(cap); elag = np.zeros(cap); eid = np.full(cap, -1, dtype=np.int64)
    m = 0
    next_id = 0
    error = 0
    for r in range(nrec):
        t = rec_time[r]
        evid = rec_evid[r]
        et[m] = t; er[m] = r; ety[m] = ROW; ep[m] = _PRIORITY[ROW]; ec[m] = -1; ev[m] = 0.; m += 1
        if evid == 0:
            et[m] = t; er[m] = r; ety[m] = OBS; ep[m] = _PRIORITY[OBS]; ec[m] = -1; ev[m] = 0.; m += 1
            continue
        if evid == 2:
            continue
        if evid == 3 or evid == 4:
            et[m] = t; er[m] = r; ety[m] = RESET; ep[m] = _PRIORITY[RESET]; ec[m] = -1; ev[m] = 0.; m += 1
            if evid == 3:
                continue
        c = rec_cmt[r]
        p = params[r]
        ss = rec_ss[r]
        ii = rec_ii[r]
        if ss > 0 and ii <= 0. and (rec_rate[r] > 0. or rec_rate[r] == -1.):
            # Steady state of a constant infusion at RATE (or the modeled rate); as in
            # NONMEM, the system is set to that steady state and the infusion stops.
            cont_rate = rec_rate[r]
            if cont_rate == -1.:
                if rate_idx[c] < 0:
                    error = E_RATE
                    continue
                cont_rate = p[rate_idx[c]]
            et[m] = t; er[m] = r; ety[m] = CONT1 if ss == 1 else CONT2; ep[m] = _PRIORITY[CONT1]
            ec[m] = c; ev[m] = 0.; erate[m] = cont_rate; m += 1
            continue
        f = p[f_idx[c]] if f_idx[c] >= 0 else 1.
        lag = p[lag_idx[c]] if lag_idx[c] >= 0 else 0.
        amount = f * rec_amt[r]
        rate = rec_rate[r]
        if rate == 0. and auto_dur[c] == 1:
            rate = -2.
        duration = 0.
        if rate == -1.:
            if rate_idx[c] < 0:
                error = E_RATE; rate = 0.
            else:
                rate = p[rate_idx[c]]
                duration = amount / rate if rate > 0. else 0.
        elif rate == -2.:
            if dur_idx[c] < 0:
                error = E_DURATION; rate = 0.
            else:
                duration = p[dur_idx[c]]
                rate = amount / duration if duration > 0. else 0.
        elif rate > 0.:
            duration = amount / rate
        if duration <= 0.:
            rate = 0.
            duration = 0.
        if ss > 0:
            if lag >= ii:
                error = E_LAG_SS
            if duration > ii:
                error = E_DURATION
            et[m] = t; er[m] = r; ety[m] = SS1 if ss == 1 else SS2; ep[m] = _PRIORITY[SS1]; ec[m] = c
            ev[m] = amount; erate[m] = rate; edur[m] = duration; elag[m] = lag
            if rate > 0. and lag + duration > ii:
                # The previous cycle's infusion is still running at the record time.
                eid[m] = next_id
                m += 1
                et[m] = t - ii + lag + duration; er[m] = r; ety[m] = INF_OFF; ep[m] = _PRIORITY[INF_OFF]
                ec[m] = c; ev[m] = rate; eid[m] = next_id; m += 1
                next_id += 1
            else:
                m += 1
        if rate > 0.:
            et[m] = t + lag; er[m] = r; ety[m] = INF_ON; ep[m] = _PRIORITY[INF_ON]; ec[m] = c; ev[m] = rate
            eid[m] = next_id; m += 1
            et[m] = t + lag + duration; er[m] = r; ety[m] = INF_OFF; ep[m] = _PRIORITY[INF_OFF]; ec[m] = c
            ev[m] = rate; eid[m] = next_id; m += 1
            next_id += 1
        else:
            et[m] = t + lag; er[m] = r; ety[m] = BOLUS; ep[m] = _PRIORITY[BOLUS]; ec[m] = c; ev[m] = amount; m += 1
    key = er[:m] * 16 + ep[:m]
    first = np.argsort(key, kind='mergesort')
    order = first[np.argsort(et[:m][first], kind='mergesort')]
    return (m, error, et[order], er[order], ety[order], ec[order], ev[order], erate[order], edur[order],
            elag[order], eid[order], next_id)


# ---------------------------------------------------------------- linear systems

@njit(cache=True, nogil=True, error_model='numpy')
def term_value(kind, p, i1, i2, i3):
    if kind == 0:
        return 1.
    if kind == 1:
        return p[i1]
    if kind == 2:
        return p[i1] / p[i2]
    if kind == 3:
        return 1. / p[i1]
    if kind == 4:
        return p[i1] * p[i2]
    if kind == 5:
        return p[i1] * p[i2] / p[i3]
    return (1. - p[i1]) * p[i2] / p[i3]


@njit(cache=True, nogil=True, error_model='numpy')
def linear_matrix(terms, constants, p, n):
    """A[row, col] += const * g(p) for each term (row, col, kind, i1, i2, i3)."""
    a = np.zeros((n, n))
    for k in range(terms.shape[0]):
        a[terms[k, 0], terms[k, 1]] += constants[k] * term_value(terms[k, 2], p, terms[k, 3], terms[k, 4], terms[k, 5])
    return a


@njit(cache=True, nogil=True, error_model='numpy')
def linear_apply(terms, constants, p, x, dx):
    """dx += A(p) x without forming A."""
    for k in range(terms.shape[0]):
        g = term_value(terms[k, 2], p, terms[k, 3], terms[k, 4], terms[k, 5])
        dx[terms[k, 0]] += constants[k] * g * x[terms[k, 1]]


@njit(cache=True, nogil=True, error_model='numpy')
def pd_transform(kind, c, p, i0, i1, i2, i3):
    """Algebraic pharmacodynamic outputs of a concentration c."""
    if kind == 0:
        return c
    if kind == 1:       # E0 + EMAX*C/(EC50 + C)
        return p[i0] + p[i1] * c / (p[i2] + c)
    if kind == 2:       # E0 + EMAX*C^g/(EC50^g + C^g)
        cg = max(c, 0.) ** p[i3]
        return p[i0] + p[i1] * cg / (p[i2] ** p[i3] + cg)
    if kind == 3:       # E0*(1 - IMAX*C/(IC50 + C))
        return p[i0] * (1. - p[i1] * c / (p[i2] + c))
    if kind == 4:       # E0 + SLOPE*C
        return p[i0] + p[i1] * c
    cg = max(c, 0.) ** p[i3]   # E0*(1 - IMAX*C^g/(IC50^g + C^g))
    return p[i0] * (1. - p[i1] * cg / (p[i2] ** p[i3] + cg))


@njit(cache=True, nogil=True, error_model='numpy')
def linear_outputs(outputs, x, p):
    """outputs rows: (state, volume index or -1, pd kind, i0, i1, i2, i3)."""
    out = np.empty(outputs.shape[0])
    for k in range(outputs.shape[0]):
        s = outputs[k, 0]; v = outputs[k, 1]
        c = x[s] / p[v] if v >= 0 else x[s]
        out[k] = pd_transform(outputs[k, 2], c, p, outputs[k, 3], outputs[k, 4], outputs[k, 5], outputs[k, 6])
    return out


@njit(cache=True, nogil=True, error_model='numpy')
def linear_ss(a, cmt, amount, rate, duration, lag, ii):
    """State at the record time under steady state, excluding the dose at record time + lag."""
    n = a.shape[0]
    unit = np.zeros(n); unit[cmt] = 1.
    period = expm(a * ii)
    lhs = np.eye(n) - period
    if rate <= 0.:
        rhs = _matvec(expm(a * (ii - lag)), unit * amount)
        return np.linalg.solve(lhs, rhs)
    e_d, j_d = expm_input(a, unit * rate, duration)
    z = np.linalg.solve(lhs, _matvec(expm(a * (ii - duration)), j_d))   # state just before an infusion start
    u = ii - lag
    if u <= duration:
        e_u, j_u = expm_input(a, unit * rate, u)
        return _matvec(e_u, z) + j_u
    return _matvec(expm(a * (u - duration)), _matvec(e_d, z) + j_d)


@njit(cache=True, nogil=True, error_model='numpy')
def linear_subject(terms, constants, outputs, n, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                   rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out,
                   params, pred):
    """Predictions of one particle for a linear system; returns an error code."""
    m, error, et, er, ety, ec, ev, erate, edur, elag, eid, nid = build_events(
        rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, params,
        f_idx, lag_idx, dur_idx, rate_idx, auto_dur)
    if error:
        return error
    nrow = params.shape[0]
    # Distinct system matrices across records (constant for time-invariant parameters).
    group = np.empty(nrow, dtype=np.int64)
    mats = np.empty((nrow, n, n))
    ng = 0
    for r in range(nrow):
        a = linear_matrix(terms, constants, params[r], n)
        found = -1
        for g in range(ng):
            same = True
            for i in range(n):
                for j in range(n):
                    if mats[g, i, j] != a[i, j]:
                        same = False
                        break
                if not same:
                    break
            if same:
                found = g
                break
        if found < 0:
            mats[ng] = a
            found = ng
            ng += 1
        group[r] = found
    cache_n = 16
    cache_g = np.full(cache_n, -1, dtype=np.int64)
    cache_dt = np.zeros(cache_n)
    cache_e = np.empty((cache_n, n, n))
    cache_next = 0
    # infusion segments: (group, dt, input rates) -> exp(A dt) and the input integral
    icache_g = np.full(cache_n, -1, dtype=np.int64)
    icache_dt = np.zeros(cache_n)
    icache_r = np.zeros((cache_n, n))
    icache_e = np.empty((cache_n, n, n))
    icache_j = np.empty((cache_n, n))
    icache_next = 0
    x = np.zeros(n)
    rate_in = np.zeros(n)
    active = np.zeros(max(nid, 1), dtype=np.int64)
    row = 0
    t = et[0]
    for k in range(m):
        te = et[k]
        if te > t:
            dt = te - t
            g = group[row]
            any_rate = False
            for i in range(n):
                if rate_in[i] != 0.:
                    any_rate = True
                    break
            if any_rate:
                hit = -1
                for q in range(cache_n):
                    if icache_g[q] == g and icache_dt[q] == dt:
                        same = True
                        for i in range(n):
                            if icache_r[q, i] != rate_in[i]:
                                same = False
                                break
                        if same:
                            hit = q
                            break
                if hit < 0:
                    hit = icache_next
                    e, j = expm_input(np.ascontiguousarray(mats[g]), rate_in, dt)
                    icache_g[hit] = g
                    icache_dt[hit] = dt
                    icache_r[hit] = rate_in
                    icache_e[hit] = e
                    icache_j[hit] = j
                    icache_next = (icache_next + 1) % cache_n
                x = _matvec(icache_e[hit], x) + icache_j[hit]
            else:
                hit = -1
                for q in range(cache_n):
                    if cache_g[q] == g and cache_dt[q] == dt:
                        hit = q
                        break
                if hit < 0:
                    hit = cache_next
                    cache_g[hit] = g
                    cache_dt[hit] = dt
                    cache_e[hit] = expm(np.ascontiguousarray(mats[g]) * dt)
                    cache_next = (cache_next + 1) % cache_n
                x = _matvec(cache_e[hit], x)
            t = te
        typ = ety[k]
        if typ == ROW:
            row = er[k]
        elif typ == OBS:
            j = rec_obs[er[k]]
            if j >= 0:
                out = linear_outputs(outputs, x, params[row])
                pred[j] = out[rec_out[er[k]]]
        elif typ == BOLUS:
            x[ec[k]] += ev[k]
        elif typ == INF_ON:
            rate_in[ec[k]] += ev[k]
            active[eid[k]] = 1
        elif typ == INF_OFF:
            if active[eid[k]] == 1:
                rate_in[ec[k]] -= ev[k]
                if abs(rate_in[ec[k]]) < 1e-12 * ev[k]:
                    rate_in[ec[k]] = 0.
                active[eid[k]] = 0
        elif typ == RESET:
            x[:] = 0.
            rate_in[:] = 0.
            active[:] = 0
        elif typ == SS1 or typ == SS2:
            xs = linear_ss(np.ascontiguousarray(mats[group[row]]), ec[k], ev[k], erate[k], edur[k], elag[k], rec_ii[er[k]])
            if typ == SS1:
                x = xs
                rate_in[:] = 0.
                active[:] = 0
            else:
                x = x + xs
            if eid[k] >= 0:
                rate_in[ec[k]] += erate[k]
                active[eid[k]] = 1
        else:   # CONT1 / CONT2: steady state of a constant infusion (the infusion then stops)
            a = np.ascontiguousarray(mats[group[row]])
            unit = np.zeros(n); unit[ec[k]] = erate[k]
            xs = np.linalg.solve(a, -unit)
            if typ == CONT1:
                x = xs
                rate_in[:] = 0.
                active[:] = 0
            else:
                x = x + xs
    return OK


@njit(cache=True, nogil=True, error_model='numpy')
def individual_rows(base, tv, kappa, iov_param, rec_occ, transform, out):
    """Natural-scale parameters per record: transform^-1(base + tv[r] + kappa[occ(r)])."""
    nrow, npar = tv.shape
    for r in range(nrow):
        for q in range(npar):
            psi = base[q] + tv[r, q]
            o = rec_occ[r]
            if o >= 0:
                for j in range(len(iov_param)):
                    if iov_param[j] == q:
                        psi += kappa[o, j]
            kind = transform[q]
            if kind == 0:
                out[r, q] = math.exp(psi)
            elif kind == 1:
                out[r, q] = 1. / (1. + math.exp(-psi))
            else:
                out[r, q] = psi


# ---------------------------------------------------------------- fast linear path (eigendecomposition)

@njit(cache=True, nogil=True, error_model='numpy')
def eigen_system(a):
    """(ok, lambda, V, V^-1) when A has real, distinct eigenvalues and a well-conditioned basis."""
    n = a.shape[0]
    lam = np.zeros(n)
    V = np.eye(n)
    Vi = np.eye(n)
    try:
        w, v = np.linalg.eig(a)
    except Exception:
        return False, lam, V, Vi
    for i in range(n):
        for j in range(i + 1, n):
            if abs(w[i] - w[j]) <= 1e-6 * max(1., abs(w[i]), abs(w[j])):
                return False, lam, V, Vi
    try:
        vi = np.linalg.inv(v)
    except Exception:
        return False, lam, V, Vi
    n1 = 0.
    n2 = 0.
    for j in range(n):
        s1 = 0.
        s2 = 0.
        for i in range(n):
            s1 += abs(v[i, j])
            s2 += abs(vi[i, j])
        n1 = max(n1, s1)
        n2 = max(n2, s2)
    if not (n1 * n2 < 1e6):
        return False, lam, V, Vi
    for i in range(n):
        lam[i] = w[i]
        for j in range(n):
            V[i, j] = v[i, j]
            Vi[i, j] = vi[i, j]
    return True, lam, V, Vi


@njit(cache=True, nogil=True, error_model='numpy')
def _phi(l, t):
    """Integral of exp(l s) over [0, t]."""
    z = l * t
    if abs(z) < 1e-10:
        return t * (1. + .5 * z)
    return math.expm1(z) / l


@njit(cache=True, nogil=True, error_model='numpy')
def eig_advance(lam, V, Vi, x, dt, rate_in, any_rate, c):
    n = len(x)
    for k in range(n):
        s = 0.
        for j in range(n):
            s += Vi[k, j] * x[j]
        s *= math.exp(lam[k] * dt)
        if any_rate:
            r = 0.
            for j in range(n):
                r += Vi[k, j] * rate_in[j]
            s += _phi(lam[k], dt) * r
        c[k] = s
    for i in range(n):
        s = 0.
        for k in range(n):
            s += V[i, k] * c[k]
        x[i] = s


@njit(cache=True, nogil=True, error_model='numpy')
def eig_ss(lam, V, Vi, cmt, amount, rate, duration, lag, ii, cont, out, c):
    """Steady state at the record time (see linear_ss); cont: constant-infusion steady state."""
    n = len(lam)
    for k in range(n):
        l = lam[k]
        g = Vi[k, cmt]
        if cont:
            if l == 0.:
                return False
            c[k] = -g * rate / l
            continue
        denom = -math.expm1(l * ii)
        if denom == 0.:
            return False
        if rate <= 0.:
            c[k] = math.exp(l * (ii - lag)) / denom * g * amount
        else:
            phid = _phi(l, duration)
            z = math.exp(l * (ii - duration)) * phid / denom * g * rate
            u = ii - lag
            if u <= duration:
                c[k] = math.exp(l * u) * z + _phi(l, u) * g * rate
            else:
                c[k] = math.exp(l * (u - duration)) * (math.exp(l * duration) * z + phid * g * rate)
    for i in range(n):
        s = 0.
        for k in range(n):
            s += V[i, k] * c[k]
        out[i] = s
    return True


@njit(cache=True, nogil=True, error_model='numpy')
def linear_output_one(outputs, k, x, p):
    s = outputs[k, 0]
    v = outputs[k, 1]
    conc = x[s] / p[v] if v >= 0 else x[s]
    return pd_transform(outputs[k, 2], conc, p, outputs[k, 3], outputs[k, 4], outputs[k, 5], outputs[k, 6])


@njit(cache=True, nogil=True, error_model='numpy')
def linear_subject_eig(terms, constants, outputs, n, events, rec_ii, rec_obs, rec_out, params, single_group, pred):
    """Eigen-coordinate propagation; returns -1 when a system matrix is not diagonalizable."""
    m, error, et, er, ety, ec, ev, erate, edur, elag, eid, nid = events
    if error:
        return error
    nrow = params.shape[0]
    ngmax = 1 if single_group else nrow
    lams = np.empty((ngmax, n))
    Vs = np.empty((ngmax, n, n))
    Vis = np.empty((ngmax, n, n))
    group = np.zeros(nrow, dtype=np.int64)
    ng = 0
    if single_group:
        ok, l, V, Vi = eigen_system(linear_matrix(terms, constants, params[0], n))
        if not ok:
            return -1
        lams[0] = l
        Vs[0] = V
        Vis[0] = Vi
        ng = 1
    else:
        mats = np.empty((nrow, n, n))
        for r in range(nrow):
            a = linear_matrix(terms, constants, params[r], n)
            found = -1
            for g in range(ng):
                same = True
                for i in range(n):
                    for j in range(n):
                        if mats[g, i, j] != a[i, j]:
                            same = False
                            break
                    if not same:
                        break
                if same:
                    found = g
                    break
            if found < 0:
                ok, l, V, Vi = eigen_system(a)
                if not ok:
                    return -1
                mats[ng] = a
                lams[ng] = l
                Vs[ng] = V
                Vis[ng] = Vi
                found = ng
                ng += 1
            group[r] = found
    x = np.zeros(n)
    xs = np.zeros(n)
    c = np.zeros(n)
    rate_in = np.zeros(n)
    active = np.zeros(max(nid, 1), dtype=np.int64)
    any_rate = False
    row = 0
    t = et[0]
    for k in range(m):
        te = et[k]
        if te > t:
            g = group[row]
            eig_advance(lams[g], Vs[g], Vis[g], x, te - t, rate_in, any_rate, c)
            t = te
        typ = ety[k]
        if typ == ROW:
            row = er[k]
        elif typ == OBS:
            j = rec_obs[er[k]]
            if j >= 0:
                pred[j] = linear_output_one(outputs, rec_out[er[k]], x, params[row])
        elif typ == BOLUS:
            x[ec[k]] += ev[k]
        elif typ == INF_ON:
            rate_in[ec[k]] += ev[k]
            active[eid[k]] = 1
            any_rate = True
        elif typ == INF_OFF:
            if active[eid[k]] == 1:
                rate_in[ec[k]] -= ev[k]
                if abs(rate_in[ec[k]]) < 1e-12 * ev[k]:
                    rate_in[ec[k]] = 0.
                active[eid[k]] = 0
                any_rate = False
                for i in range(n):
                    if rate_in[i] != 0.:
                        any_rate = True
                        break
        elif typ == RESET:
            x[:] = 0.
            rate_in[:] = 0.
            active[:] = 0
            any_rate = False
        else:
            g = group[row]
            cont = typ == CONT1 or typ == CONT2
            if not eig_ss(lams[g], Vs[g], Vis[g], ec[k], ev[k], erate[k], edur[k], elag[k], rec_ii[er[k]], cont, xs, c):
                return E_SINGULAR
            if typ == SS1 or typ == CONT1:
                x[:] = xs
                rate_in[:] = 0.
                active[:] = 0
                any_rate = False
            else:
                for i in range(n):
                    x[i] += xs[i]
            if (typ == SS1 or typ == SS2) and eid[k] >= 0:
                rate_in[ec[k]] += erate[k]
                active[eid[k]] = 1
                any_rate = True
    return OK


@njit(cache=True, nogil=True, error_model='numpy')
def _linear_particle(terms, constants, outputs, n, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                     rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out,
                     rows, shared, shared_events, single_group, pred):
    if shared:
        events = shared_events
    else:
        events = build_events(rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rows,
                              f_idx, lag_idx, dur_idx, rate_idx, auto_dur)
    code = linear_subject_eig(terms, constants, outputs, n, events, rec_ii, rec_obs, rec_out, rows, single_group, pred)
    if code == -1:
        code = linear_subject(terms, constants, outputs, n, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                              rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out,
                              rows, pred)
    return code


@njit(cache=True, nogil=True, error_model='numpy')
def _shared_events(f_idx, lag_idx, dur_idx, rate_idx, rec_evid, rec_cmt, base, iov_param):
    """True when no dosing parameter (F, lag, duration, rate) differs between the particles of a batch."""
    npar = base.shape[1]
    varies = np.zeros(npar, dtype=np.bool_)
    for q in range(npar):
        for k in range(1, base.shape[0]):
            if base[k, q] != base[0, q]:
                varies[q] = True
                break
    for j in range(len(iov_param)):
        varies[iov_param[j]] = True
    for r in range(len(rec_evid)):
        if rec_evid[r] == 1 or rec_evid[r] == 4:
            c = rec_cmt[r]
            for idx in (f_idx[c], lag_idx[c], dur_idx[c], rate_idx[c]):
                if idx >= 0 and varies[idx]:
                    return False
    return True


@njit(cache=True, nogil=True, error_model='numpy')
def _batch_flags(terms, f_idx, lag_idx, dur_idx, rate_idx, auto_dur, rec_evid, rec_cmt, rec_rate, base, tv, iov_param):
    """(shared events, single system matrix) for a batch."""
    npar = base.shape[1]
    by_record = np.zeros(npar, dtype=np.bool_)
    for q in range(npar):
        for r in range(1, tv.shape[0]):
            if tv[r, q] != tv[0, q]:
                by_record[q] = True
                break
    for j in range(len(iov_param)):
        by_record[iov_param[j]] = True
    shared = _shared_events(f_idx, lag_idx, dur_idx, rate_idx, rec_evid, rec_cmt, base, iov_param)
    single = True
    for t in range(terms.shape[0]):
        for s in range(3, 6):
            idx = terms[t, s]
            if idx >= 0 and by_record[idx]:
                single = False
    return shared, single


@njit(cache=True, nogil=True, error_model='numpy', parallel=True)
def linear_batch(terms, constants, outputs, n, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                 rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out, n_obs,
                 base, tv, kappa, iov_param, rec_occ, transform, chunks):
    """Predictions for K particles: base (K,P) transformed values, kappa (K,O,Q), in `chunks` work blocks."""
    count = base.shape[0]
    pred = np.full((count, n_obs), np.nan)
    errors = np.zeros(count, dtype=np.int64)
    shared, single = _batch_flags(terms, f_idx, lag_idx, dur_idx, rate_idx, auto_dur, rec_evid, rec_cmt, rec_rate,
                                  base, tv, iov_param)
    if count == 0:
        return pred, errors
    first = np.empty(tv.shape)
    individual_rows(base[0], tv, kappa[0], iov_param, rec_occ, transform, first)
    events = build_events(rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, first,
                          f_idx, lag_idx, dur_idx, rate_idx, auto_dur)
    chunks = max(1, min(count, chunks))
    for chunk in prange(chunks):
        rows = np.empty(tv.shape)
        lo = chunk * count // chunks
        hi = (chunk + 1) * count // chunks
        for k in range(lo, hi):
            individual_rows(base[k], tv, kappa[k], iov_param, rec_occ, transform, rows)
            errors[k] = _linear_particle(terms, constants, outputs, n, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                                         rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs,
                                         rec_out, rows, shared, events, single, pred[k])
    return pred, errors


@njit(cache=True, nogil=True, error_model='numpy')
def linear_batch_serial(terms, constants, outputs, n, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                        rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out, n_obs,
                        base, tv, kappa, iov_param, rec_occ, transform):
    count = base.shape[0]
    pred = np.full((count, n_obs), np.nan)
    errors = np.zeros(count, dtype=np.int64)
    shared, single = _batch_flags(terms, f_idx, lag_idx, dur_idx, rate_idx, auto_dur, rec_evid, rec_cmt, rec_rate,
                                  base, tv, iov_param)
    if count == 0:
        return pred, errors
    rows = np.empty(tv.shape)
    individual_rows(base[0], tv, kappa[0], iov_param, rec_occ, transform, rows)
    events = build_events(rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rows,
                          f_idx, lag_idx, dur_idx, rate_idx, auto_dur)
    for k in range(count):
        individual_rows(base[k], tv, kappa[k], iov_param, rec_occ, transform, rows)
        errors[k] = _linear_particle(terms, constants, outputs, n, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                                     rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs,
                                     rec_out, rows, shared, events, single, pred[k])
    return pred, errors


# ---------------------------------------------------------------- nonlinear systems

@njit(cache=False, nogil=True, error_model='numpy')
def _derivative(rhs, t, x, p, rate_in, dx):
    rhs(t, x, p, dx)
    for i in range(len(x)):
        dx[i] += rate_in[i]


@njit(cache=True, nogil=True, error_model='numpy')
def _error_norm(err, x0, x1, rtol, atol):
    total = 0.
    for i in range(len(err)):
        scale = atol + rtol * max(abs(x0[i]), abs(x1[i]))
        total += (err[i] / scale) ** 2
    return math.sqrt(total / len(err))


@njit(cache=False, nogil=True, error_model='numpy')
def dopri5(rhs, t0, t1, x, p, rate_in, rtol, atol, h, max_steps, ws):
    """Dormand-Prince 5(4) with FSAL, in place on x; ws is a (10, n) workspace.

    Returns (x, proposed next step, steps or -1 on failure).
    """
    n = len(x)
    k1 = ws[0]; k2 = ws[1]; k3 = ws[2]; k4 = ws[3]; k5 = ws[4]
    k6 = ws[5]; k7 = ws[6]; y = ws[7]; xn = ws[8]; err = ws[9]
    t = t0
    span = t1 - t0
    if h <= 0.:
        h = 1e-2 * span
    _derivative(rhs, t, x, p, rate_in, k1)
    steps = 0
    while t < t1:
        if steps >= max_steps:
            return x, h, -1
        last = t + h >= t1
        step = t1 - t if last else h
        for i in range(n):
            y[i] = x[i] + step * (0.2 * k1[i])
        _derivative(rhs, t + 0.2 * step, y, p, rate_in, k2)
        for i in range(n):
            y[i] = x[i] + step * (3. / 40. * k1[i] + 9. / 40. * k2[i])
        _derivative(rhs, t + 0.3 * step, y, p, rate_in, k3)
        for i in range(n):
            y[i] = x[i] + step * (44. / 45. * k1[i] - 56. / 15. * k2[i] + 32. / 9. * k3[i])
        _derivative(rhs, t + 0.8 * step, y, p, rate_in, k4)
        for i in range(n):
            y[i] = x[i] + step * (19372. / 6561. * k1[i] - 25360. / 2187. * k2[i] + 64448. / 6561. * k3[i]
                                  - 212. / 729. * k4[i])
        _derivative(rhs, t + 8. / 9. * step, y, p, rate_in, k5)
        for i in range(n):
            y[i] = x[i] + step * (9017. / 3168. * k1[i] - 355. / 33. * k2[i] + 46732. / 5247. * k3[i]
                                  + 49. / 176. * k4[i] - 5103. / 18656. * k5[i])
        _derivative(rhs, t + step, y, p, rate_in, k6)
        for i in range(n):
            xn[i] = x[i] + step * (35. / 384. * k1[i] + 500. / 1113. * k3[i] + 125. / 192. * k4[i]
                                   - 2187. / 6784. * k5[i] + 11. / 84. * k6[i])
        _derivative(rhs, t + step, xn, p, rate_in, k7)
        for i in range(n):
            err[i] = step * (71. / 57600. * k1[i] - 71. / 16695. * k3[i] + 71. / 1920. * k4[i]
                             - 17253. / 339200. * k5[i] + 22. / 525. * k6[i] - 1. / 40. * k7[i])
        steps += 1
        norm = _error_norm(err, x, xn, rtol, atol)
        if not np.isfinite(norm):
            return x, h, -1
        factor = 5. if norm == 0. else min(5., max(.2, .9 * norm ** -.2))
        if norm <= 1. or step < 1e-14 * max(1., abs(t)):
            t = t1 if last else t + step
            for i in range(n):
                x[i] = xn[i]
                k1[i] = k7[i]
            if not last:
                h = step * factor
            elif factor < 1.:
                h = min(h, step * factor)
        else:
            h = step * factor
    return x, h, steps


@njit(cache=False, nogil=True, error_model='numpy')
def rosenbrock23(rhs, t0, t1, x, p, rate_in, rtol, atol, h, max_steps):
    """L-stable Rosenbrock (ode23s) with a finite-difference Jacobian; for stiff systems."""
    n = len(x)
    x = x.copy()
    d = 1. / (2. + math.sqrt(2.))
    e32 = 6. + math.sqrt(2.)
    f0 = np.empty(n); f1 = np.empty(n); f2 = np.empty(n); fp = np.empty(n); ft = np.empty(n)
    jac = np.empty((n, n)); y = np.empty(n)
    t = t0
    span = t1 - t0
    if h <= 0.:
        h = 1e-3 * span
    steps = 0
    while t < t1:
        if steps >= max_steps:
            return x, h, -1
        last = t + h >= t1
        step = t1 - t if last else h
        _derivative(rhs, t, x, p, rate_in, f0)
        for j in range(n):
            delta = 1e-7 * max(1., abs(x[j]))
            y[:] = x
            y[j] += delta
            _derivative(rhs, t, y, p, rate_in, fp)
            for i in range(n):
                jac[i, j] = (fp[i] - f0[i]) / delta
        dt_ = 1e-7 * max(1., abs(t))
        _derivative(rhs, t + dt_, x, p, rate_in, ft)
        tder = (ft - f0) / dt_
        w = np.eye(n) - step * d * jac
        try:
            k1 = np.linalg.solve(w, f0 + step * d * tder)
            y[:] = x + .5 * step * k1
            _derivative(rhs, t + .5 * step, y, p, rate_in, f1)
            k2 = np.linalg.solve(w, f1 - k1) + k1
            xn = x + step * k2
            _derivative(rhs, t + step, xn, p, rate_in, f2)
            k3 = np.linalg.solve(w, f2 - e32 * (k2 - f1) - 2. * (k1 - f0) + step * d * tder)
        except Exception:           # singular or non-finite iteration matrix: the integration fails
            return x, h, -1
        err = step / 6. * (k1 - 2. * k2 + k3)
        steps += 1
        norm = _error_norm(err, x, xn, rtol, atol)
        if not np.isfinite(norm):
            return x, h, -1
        factor = 5. if norm == 0. else min(5., max(.2, .8 * norm ** (-1. / 3.)))
        if norm <= 1. or step < 1e-14 * max(1., abs(t)):
            t = t1 if last else t + step
            x[:] = xn
            if not last:
                h = step * factor
            elif factor < 1.:
                h = min(h, step * factor)
        else:
            h = step * factor
    return x, h, steps


@njit(cache=False, nogil=True, error_model='numpy')
def integrate(rhs, method, t0, t1, x, p, rate_in, rtol, atol, h, max_steps, ws):
    """method 0: Dormand-Prince, 1: Rosenbrock, 2: Dormand-Prince with a stiff fallback.

    ws: (11, n) workspace; its last row keeps the starting state for the fallback.
    """
    if method == 1:
        return rosenbrock23(rhs, t0, t1, x, p, rate_in, rtol, atol, h, max_steps)
    if method == 2:
        for i in range(len(x)):
            ws[10, i] = x[i]
    out, h_new, steps = dopri5(rhs, t0, t1, x, p, rate_in, rtol, atol, h, max_steps, ws)
    if steps < 0 and method == 2:
        for i in range(len(x)):
            x[i] = ws[10, i]
        return rosenbrock23(rhs, t0, t1, x, p, rate_in, rtol, atol, h, max_steps)
    return out, h_new, steps


@njit(cache=False, nogil=True, error_model='numpy')
def _advance(rhs, method, t0, t1, x, p, rate_in, rtol, atol, h, max_steps, ws):
    if t1 <= t0:
        return x, h, 0
    return integrate(rhs, method, t0, t1, x, p, rate_in, rtol, atol, h, max_steps, ws)


@njit(cache=False, nogil=True, error_model='numpy')
def ode_ss(rhs, method, x_start, p, n, cmt, amount, rate, duration, lag, ii, rtol, atol, max_steps,
           cycles, tolerance, ws):
    """Periodic steady state at the record time by repeating the dosing cycle.

    Cycle k spans [k*ii, (k+1)*ii); the dose starts at lag within the cycle and an
    infusion running past the cycle end continues into the next cycle. The state
    is recorded at the cycle end, i.e. at the next record time before its dose.
    Returns (state, converged).
    """
    x = x_start.copy()
    rate_in = np.zeros(n)
    h = 0.
    end = lag + duration
    carry = False
    for k in range(cycles):
        start = x.copy()
        t = 0.
        if carry:
            rate_in[cmt] = rate
            x, h, s = _advance(rhs, method, 0., end - ii, x, p, rate_in, rtol, atol, h, max_steps, ws)
            if s < 0:
                return x, False
            rate_in[cmt] = 0.
            t = end - ii
        x, h, s = _advance(rhs, method, t, lag, x, p, rate_in, rtol, atol, h, max_steps, ws)
        if s < 0:
            return x, False
        t = max(t, lag)
        if rate <= 0.:
            x[cmt] += amount
            x, h, s = _advance(rhs, method, t, ii, x, p, rate_in, rtol, atol, h, max_steps, ws)
            if s < 0:
                return x, False
            carry = False
        else:
            rate_in[cmt] = rate
            stop = min(end, ii)
            x, h, s = _advance(rhs, method, t, stop, x, p, rate_in, rtol, atol, h, max_steps, ws)
            if s < 0:
                return x, False
            rate_in[cmt] = 0.
            if end < ii:
                x, h, s = _advance(rhs, method, stop, ii, x, p, rate_in, rtol, atol, h, max_steps, ws)
                if s < 0:
                    return x, False
            carry = end > ii
        change = 0.
        size = 0.
        for i in range(n):
            change = max(change, abs(x[i] - start[i]))
            size = max(size, abs(x[i]))
        if k > 0 and change <= tolerance * max(size, 1e-12) + 1e-14:
            return x, True
    return x, False


@njit(cache=False, nogil=True, error_model='numpy')
def ode_subject(rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol,
                f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out, params, pred,
                events, ws):
    m, error, et, er, ety, ec, ev, erate, edur, elag, eid, nid = events
    if error:
        return error
    x = np.zeros(n)
    initial(params[0], x)
    rate_in = np.zeros(n)
    active = np.zeros(max(nid, 1), dtype=np.int64)
    out = np.empty(n_out)
    row = 0
    t = et[0]
    h = 0.
    for k in range(m):
        te = et[k]
        if te > t:
            x, h, steps = _advance(rhs, method, t, te, x, params[row], rate_in, rtol, atol, h, max_steps, ws)
            if steps < 0:
                return E_ODE
            t = te
        typ = ety[k]
        if typ == ROW:
            row = er[k]
        elif typ == OBS:
            j = rec_obs[er[k]]
            if j >= 0:
                output(x, params[row], out)
                pred[j] = out[rec_out[er[k]]]
        elif typ == BOLUS:
            x[ec[k]] += ev[k]
        elif typ == INF_ON:
            rate_in[ec[k]] += ev[k]
            active[eid[k]] = 1
        elif typ == INF_OFF:
            if active[eid[k]] == 1:
                rate_in[ec[k]] -= ev[k]
                if abs(rate_in[ec[k]]) < 1e-12 * ev[k]:
                    rate_in[ec[k]] = 0.
                active[eid[k]] = 0
        elif typ == RESET:
            x[:] = 0.
            initial(params[row], x)
            rate_in[:] = 0.
            active[:] = 0
        elif typ == SS1 or typ == SS2:
            start = np.zeros(n)
            initial(params[row], start)
            xs, converged = ode_ss(rhs, method, start, params[row], n, ec[k], ev[k], erate[k], edur[k],
                                   elag[k], rec_ii[er[k]], rtol, atol, max_steps, ss_cycles, ss_tol, ws)
            if not converged:
                return E_SS_ODE
            if typ == SS1:
                x = xs
                rate_in[:] = 0.
                active[:] = 0
            else:
                x = x + xs - start
            if eid[k] >= 0:
                rate_in[ec[k]] += erate[k]
                active[eid[k]] = 1
        else:   # constant-infusion steady state by long integration
            start = np.zeros(n)
            initial(params[row], start)
            xs = start.copy()
            infusion = np.zeros(n)
            infusion[ec[k]] = erate[k]
            converged = False
            chunk = 24.
            for _ in range(ss_cycles):
                previous = xs.copy()
                xs, h, steps = integrate(rhs, method, 0., chunk, xs, params[row], infusion, rtol, atol, h, max_steps, ws)
                if steps < 0:
                    return E_ODE
                change = 0.
                size = 0.
                for i in range(n):
                    change = max(change, abs(xs[i] - previous[i]))
                    size = max(size, abs(xs[i]))
                if change <= ss_tol * max(size, 1e-12) + 1e-14:
                    converged = True
                    break
                chunk *= 2.
            if not converged:
                return E_SS_ODE
            if typ == CONT1:
                x = xs
                rate_in[:] = 0.
                active[:] = 0
            else:
                x = x + xs - start
    return OK


@njit(cache=False, nogil=True, error_model='numpy')
def _ode_particle(rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol,
                  f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                  rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out,
                  rows, shared, shared_events, pred, ws):
    if shared:
        events = shared_events
    else:
        events = build_events(rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rows,
                              f_idx, lag_idx, dur_idx, rate_idx, auto_dur)
    return ode_subject(rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol,
                       f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                       rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out,
                       rows, pred, events, ws)


@njit(cache=False, nogil=True, error_model='numpy', parallel=True)
def ode_batch(rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol,
              f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
              rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out, n_obs,
              base, tv, kappa, iov_param, rec_occ, transform, chunks):
    """Predictions for K particles of a nonlinear system, in `chunks` work blocks."""
    count = base.shape[0]
    pred = np.full((count, n_obs), np.nan)
    errors = np.zeros(count, dtype=np.int64)
    if count == 0:
        return pred, errors
    shared = _shared_events(f_idx, lag_idx, dur_idx, rate_idx, rec_evid, rec_cmt, base, iov_param)
    first = np.empty(tv.shape)
    individual_rows(base[0], tv, kappa[0], iov_param, rec_occ, transform, first)
    events = build_events(rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, first,
                          f_idx, lag_idx, dur_idx, rate_idx, auto_dur)
    chunks = max(1, min(count, chunks))
    for chunk in prange(chunks):
        rows = np.empty(tv.shape)
        ws = np.empty((11, n))
        lo = chunk * count // chunks
        hi = (chunk + 1) * count // chunks
        for k in range(lo, hi):
            individual_rows(base[k], tv, kappa[k], iov_param, rec_occ, transform, rows)
            errors[k] = _ode_particle(rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles,
                                      ss_tol, f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                                      rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs,
                                      rec_out, rows, shared, events, pred[k], ws)
    return pred, errors


@njit(cache=False, nogil=True, error_model='numpy')
def ode_batch_serial(rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol,
                     f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                     rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out, n_obs,
                     base, tv, kappa, iov_param, rec_occ, transform):
    count = base.shape[0]
    pred = np.full((count, n_obs), np.nan)
    errors = np.zeros(count, dtype=np.int64)
    if count == 0:
        return pred, errors
    shared = _shared_events(f_idx, lag_idx, dur_idx, rate_idx, rec_evid, rec_cmt, base, iov_param)
    rows = np.empty(tv.shape)
    ws = np.empty((11, n))
    individual_rows(base[0], tv, kappa[0], iov_param, rec_occ, transform, rows)
    events = build_events(rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rows,
                          f_idx, lag_idx, dur_idx, rate_idx, auto_dur)
    for k in range(count):
        individual_rows(base[k], tv, kappa[k], iov_param, rec_occ, transform, rows)
        errors[k] = _ode_particle(rhs, output, initial, method, n, n_out, rtol, atol, max_steps, ss_cycles, ss_tol,
                                  f_idx, lag_idx, dur_idx, rate_idx, auto_dur,
                                  rec_time, rec_evid, rec_amt, rec_cmt, rec_rate, rec_ii, rec_ss, rec_obs, rec_out,
                                  rows, shared, events, pred[k], ws)
    return pred, errors
