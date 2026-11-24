"""Logit SUE solver and exact adjoint gradient for MaTE's city configuration with a polynomial travel-time function."""
import copy

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, gmres


def _gmres(op, b, tol):
    try:
        return gmres(op, b, rtol=tol, atol=0.0, restart=100, maxiter=50)
    except TypeError:  # scipy < 1.12
        return gmres(op, b, tol=tol, atol=0.0, restart=100, maxiter=50)


class Static:
    """Network, demand structure and baseline observations exported by prepare.py."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.D = sps.csr_matrix((z["D_data"], z["D_indices"], z["D_indptr"]), shape=tuple(z["D_shape"]))
        self.DT = self.D.T.tocsr()
        self.od = z["od"].astype(np.int64); self.n_od = int(z["n_od"])
        self.orig = z["orig"].astype(np.int64); self.n_nodes = int(z["n_nodes"])
        self.refmask = z["refmask"].astype(np.float64)
        self.tf_ = z["tf_"]; self.cap = z["cap"]
        self.theta = z["theta"].astype(np.float64)
        self.L = z["links_len"]
        self.obs_flow = z["obs_flow"]
        self.obs_speed = self.L[None, :] / z["obs_tt"]
        self.Nc = int(np.isfinite(self.obs_flow).sum()); self.vc = float(np.nanstd(self.obs_flow) ** 2)
        self.Ns = int(np.isfinite(self.obs_speed).sum()); self.vs = float(np.nanstd(self.obs_speed) ** 2)
        self.n_links, self.n_paths = self.D.shape

    def with_cap(self, cap):
        S2 = copy.copy(self); S2.cap = np.asarray(cap, np.float64)
        return S2


def demand(S, h, G_h, U_h):
    """Generation x distribution: q_j = relu(G[o_j]) * softmax over the destinations of o_j of U."""
    gin = G_h * S.refmask[h]
    g = np.maximum(gin, 0.0)
    umax = np.full(S.n_nodes, -np.inf); np.maximum.at(umax, S.orig, U_h)
    e = np.exp(U_h - umax[S.orig]); den = np.bincount(S.orig, weights=e, minlength=S.n_nodes)
    phi = e / den[S.orig]
    return g, gin, phi, g[S.orig] * phi


def tt(S, x, w, c):
    """t = t0 (1 + w sum_k c_k u^k), u = x / cap; terms with c_k = 0 are skipped."""
    u = x / S.cap
    poly = np.zeros_like(u)
    for k, ck in enumerate(c):
        if ck != 0.0:
            poly = poly + ck * u ** (k + 1)
    return S.tf_ * (1.0 + w * poly), u, poly


def dtt_dx(S, x, w, c):
    u = x / S.cap
    d = np.zeros_like(u)
    for k, ck in enumerate(c):
        if ck != 0.0:
            d = d + (k + 1) * ck * u ** k
    return S.tf_ * w * d / S.cap


def shares(S, V):
    vmax = np.full(S.n_od, -np.inf); np.maximum.at(vmax, S.od, V)
    e = np.exp(V - vmax[S.od])
    den = np.bincount(S.od, weights=e, minlength=S.n_od)
    return e / den[S.od]


class Hour:
    """One period: warm-started SUE solve, loss and its exact gradient through the equilibrium."""

    def __init__(self, S, h):
        self.S, self.h = S, h
        self.th = float(S.theta[h])
        self.cl = np.where(np.isfinite(S.obs_flow[h]))[0]; self.cv = S.obs_flow[h, self.cl]
        self.sl = np.where(np.isfinite(S.obs_speed[h]))[0]; self.sv = S.obs_speed[h, self.sl]
        self.P = None
        self.lin_extra = self.path_pen = None

    def set_intervention(self, lin_extra=None, closed_paths=None):
        self.lin_extra = None if lin_extra is None else np.asarray(lin_extra, np.float64)
        self.path_pen = None if closed_paths is None else np.where(closed_paths, -np.inf, 0.0)

    def set_params(self, fe_raw, G_h, U_h, logc, w_raw):
        S = self.S
        self.c = np.exp(np.asarray(logc, np.float64).reshape(-1)); self.w = float(np.exp(w_raw))
        fe_raw = np.asarray(fe_raw, np.float64)
        self.mfe = fe_raw <= 0  # link fixed effects can only lower utility
        self.lin = fe_raw * self.mfe
        if self.lin_extra is not None:
            self.lin = self.lin + self.lin_extra
        self.g, self.gin, self.phi, self.q = demand(S, self.h, np.asarray(G_h, np.float64), np.asarray(U_h, np.float64))
        self.qp = self.q[S.od]

    def pathflows(self, x):
        t = tt(self.S, x, self.w, self.c)[0]
        V = self.S.DT @ (self.th * t + self.lin)
        if self.path_pen is not None:
            V = V + self.path_pen
        P = shares(self.S, V)
        return self.qp * P, P

    def fisk_deriv(self, f, d, lam):
        fl = f + lam * d
        x = self.S.D @ fl; dx = self.S.D @ d
        g = -(self.th * tt(self.S, x, self.w, self.c)[0] + self.lin) @ dx
        pos = fl > 0
        return g + np.sum(d[pos] * np.log(fl[pos]))

    def solve(self, f0, tol=1e-6, max_iter=3000, bis=40):
        S = self.S; f = f0.copy(); reason = "max_iter"; r = np.inf; k = 0
        for k in range(max_iter + 1):
            x = S.D @ f
            fy, _ = self.pathflows(x)
            y = S.D @ fy
            r = float(np.abs(y - x).sum() / max(np.abs(x).sum(), 1e-300))
            if r <= tol:
                reason = "tolerance"; break
            if k == max_iter:
                break
            d = fy - f
            if self.fisk_deriv(f, d, 1.0) <= 0:
                lam = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(bis):
                    mid = 0.5 * (lo + hi)
                    if self.fisk_deriv(f, d, mid) > 0: hi = mid
                    else: lo = mid
                lam = 0.5 * (lo + hi)
            f = f + lam * d
        return f, dict(iters=k, resid=r, reason=reason)

    def polish(self, x, tol=1e-13, max_newton=30, gmres_tol=1e-12):
        """Newton-GMRES on x = F(x); used for the tight solves of the finite-difference check."""
        S = self.S; od = S.od; hist = []

        def resid(x):
            fy, P = self.pathflows(x); y = S.D @ fy
            return float(np.abs(y - x).sum() / max(np.abs(x).sum(), 1e-300)), y, fy, P
        r, y, fy, P = resid(x)
        while True:
            hist.append(r)
            if r <= tol or len(hist) > max_newton:
                break
            k = self.th * dtt_dx(S, x, self.w, self.c)

            def Jv(v, fy=fy, P=P, k=k):
                u = S.DT @ (k * v)
                return S.D @ (fy * (u - np.bincount(od, weights=P * u, minlength=S.n_od)[od]))
            op = LinearOperator((S.n_links, S.n_links), matvec=lambda v: v - Jv(v), dtype=np.float64)
            dx, _ = _gmres(op, y - x, gmres_tol)
            step, ok = 1.0, False
            for _ in range(30):
                rn, yn, fyn, Pn = resid(x + step * dx)
                if rn < r:
                    ok = True; break
                step *= 0.5
            if not ok:
                break
            x, r, y, fy, P = x + step * dx, rn, yn, fyn, Pn
        return x, dict(newton_iters=len(hist) - 1, resid=r, reason="tolerance" if r <= tol else "stalled")

    def equilibrate(self, x_init=None, tol=1e-6, polish_tol=None, **kw):
        f0 = self.qp * self.P if self.P is not None else self.pathflows(x_init)[0]
        f, info = self.solve(f0, tol=tol, **kw)
        x = self.S.D @ f
        if polish_tol is not None:
            x, pinfo = self.polish(x, tol=polish_tol)
            info = dict(info, resid=pinfo["resid"], reason=pinfo["reason"])
        _, P = self.pathflows(x)
        self.P = P; self.f = f if polish_tol is None else self.qp * P; self.x = x
        return x, info

    def loss_terms(self, x):
        """(count + speed) / 3 with MaTE's z2score normalisation, pooled over periods."""
        S = self.S
        t = tt(S, x, self.w, self.c)[0]
        rc = x[self.cl] - self.cv
        sp = S.L[self.sl] / t[self.sl]; rs = sp - self.sv
        lc = float(rc @ rc) / (S.Nc * S.vc); ls = float(rs @ rs) / (S.Ns * S.vs)
        return (lc + ls) / 3.0, lc, ls, t, rc, rs

    def value_and_grad(self, x_init=None, tol=1e-6, gmres_tol=1e-10, **kw):
        S = self.S; od = S.od
        x, info = self.equilibrate(x_init=x_init, tol=tol, **kw)
        f, P = self.f, self.P
        loss, lc, ls, t, rc, rs = self.loss_terms(x)
        tp = dtt_dx(S, x, self.w, self.c)
        dldt = np.zeros(S.n_links)
        dldt[self.sl] = (2.0 / 3.0) * rs / (S.Ns * S.vs) * (-S.L[self.sl] / t[self.sl] ** 2)
        b = dldt * tp
        b[self.cl] += (2.0 / 3.0) * rc / (S.Nc * S.vc)

        def Sop(u):
            return f * (u - np.bincount(od, weights=P * u, minlength=S.n_od)[od])

        # adjoint system (I - dF/dx)^T lam = dl/dx
        k = self.th * tp
        op = LinearOperator((S.n_links, S.n_links), matvec=lambda v: v - k * (S.D @ Sop(S.DT @ v)), dtype=np.float64)
        lam, _ = _gmres(op, b, gmres_tol)
        nu = S.DT @ lam
        glin = S.D @ Sop(nu)
        gt = self.th * glin + dldt
        _, u, poly = tt(S, x, self.w, self.c)
        g_wraw = float(np.sum(gt * self.w * S.tf_ * poly))
        g_logc = np.array([float(np.sum(gt * self.w * self.c[i] * S.tf_ * u ** (i + 1))) for i in range(len(self.c))])
        gq = np.bincount(od, weights=nu * P, minlength=S.n_od)
        gg = np.bincount(S.orig, weights=gq * self.phi, minlength=S.n_nodes)
        grads = dict(fe_raw=glin * self.mfe, G_h=gg * S.refmask[self.h] * (self.gin > 0),
                     U_h=self.g[S.orig] * self.phi * (gq - gg[S.orig]), logc=g_logc, w_raw=g_wraw)
        diag = dict(loss_count=lc, loss_speed=ls, solve=info, demand=float(self.q.sum()),
                    ape_speed_sum=float(np.sum(np.abs(rs) / self.sv)), n_speed=len(rs))
        return loss, grads, diag


class Adam:
    """Keras legacy Adam update; parameters fixed at -inf (zero polynomial terms) stay there."""

    def __init__(self, b1=0.9, b2=0.999, eps=1e-7):
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0
        self.m, self.v = {}, {}

    def step(self, params, grads, lrs):
        self.t += 1
        for k, g in grads.items():
            g = np.asarray(g, np.float64)
            if k not in self.m:
                self.m[k] = np.zeros_like(g); self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            lr_t = lrs[k] * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
            upd = lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)
            p = np.asarray(params[k], np.float64)
            params[k] = np.where(np.isfinite(p), p - upd, p)
        return params
