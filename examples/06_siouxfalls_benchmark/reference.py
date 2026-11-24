"""TNTP Sioux Falls reader, route catalogue loader and the independent reference logit SUE (numpy/scipy only)."""
import hashlib
import json
import os
import re

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
TNTP_DIR = os.path.join(HERE, "downloads", "SiouxFalls")
RUNS = os.path.join(HERE, "runs")
CATALOGUE = os.path.join(RUNS, "route_catalogue.json")
LEVELS = (1.0, 0.8)
VOT, TOLL = 0.20, 2.00  # $ per time unit and $; the toll adds 10 time units to the link
CAP_FACTOR = {"cap+25%": 1.25, "cap-50%": 0.50}


def read_net(folder=TNTP_DIR):
    rows, started = [], False
    with open(os.path.join(folder, "SiouxFalls_net.tntp")) as fh:
        for line in fh:
            s = line.strip()
            if not started:
                started = s.startswith("~")
                continue
            if s and not s.startswith("~"):
                rows.append([float(p) for p in s.replace(";", "").split()[:10]])
    a = np.array(rows)
    return dict(init=a[:, 0].astype(int), term=a[:, 1].astype(int), cap=a[:, 2], length=a[:, 3], fft=a[:, 4],
                alpha=a[:, 5], beta=a[:, 6])


def read_trips(folder=TNTP_DIR):
    txt = open(os.path.join(folder, "SiouxFalls_trips.tntp")).read()
    total = float(re.search(r"<TOTAL OD FLOW>\s*([0-9.]+)", txt).group(1))
    Q, orig = {}, None
    for line in txt.split("<END OF METADATA>")[1].splitlines():
        s = line.strip()
        if s.startswith("Origin"):
            orig = int(s.split()[1])
        elif s:
            for m in re.finditer(r"(\d+)\s*:\s*([0-9.]+)", s):
                Q[(orig, int(m.group(1)))] = float(m.group(2))
    return Q, total


def load_network(folder=TNTP_DIR):
    """Links in file order (1-based nodes) and O-D pairs with positive demand, sorted by origin then destination."""
    net = read_net(folder)
    Q, total = read_trips(folder)
    ods = sorted([od for od, v in Q.items() if v > 0 and od[0] != od[1]])
    q = np.array([Q[od] for od in ods])
    assert abs(q.sum() - total) < 1e-6, (q.sum(), total)
    net.update(ods=ods, q=q)
    return net


def load_benchmark(cat_fn=CATALOGUE):
    raw = open(cat_fn, "rb").read()
    cat = json.loads(raw)
    tn = load_network()
    links = list(zip(tn["init"].tolist(), tn["term"].tolist()))
    assert [tuple(l) for l in cat["links"]] == links and [tuple(o) for o in cat["ods"]] == list(tn["ods"])
    rows, cols, od = [], [], []
    for j, p in enumerate(cat["paths"]):
        assert [links[a][0] for a in p["links"]] + [links[p["links"][-1]][1]] == p["nodes"]
        for a in p["links"]:
            rows.append(a); cols.append(j)
        od.append(p["od_index"])
    D = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(links), len(cat["paths"])))
    return dict(links=links, fft=tn["fft"], cap=tn["cap"], alpha=tn["alpha"], beta=tn["beta"], length=tn["length"],
                q=tn["q"], ods=list(tn["ods"]), n_od=len(tn["ods"]), D=D, od=np.array(od), paths=cat["paths"],
                sha256=hashlib.sha256(raw).hexdigest())


def intervention(net, kind, j):
    """Keyword arguments of SUE for one intervention on link j."""
    if kind in CAP_FACTOR:
        cap = net["cap"].copy(); cap[j] *= CAP_FACTOR[kind]
        return dict(cap=cap)
    if kind == "toll":
        return dict(toll=np.where(np.arange(len(net["links"])) == j, TOLL, 0.0), vot=VOT)
    if kind == "closure":
        return dict(closed=[j])
    raise ValueError(kind)


class SUE:
    """Logit SUE over the catalogue paths with the TNTP BPR curves, solved by path-based Fisk line search."""

    def __init__(self, net, theta_tt=-1.0, demand_scale=1.0, cap=None, toll=None, vot=1.0, closed=None):
        self.n = net
        self.D = net["D"]; self.od = net["od"]
        self.q = net["q"] * demand_scale
        L, P = self.D.shape
        self.cap = net["cap"].copy() if cap is None else np.asarray(cap, float)
        self.alpha, self.beta = net["alpha"], net["beta"]
        self.th = theta_tt
        self.lin = theta_tt * (np.zeros(L) if toll is None else np.asarray(toll, float)) / vot
        self.avail = np.ones(P, bool)
        if closed:
            self.avail &= ~(np.asarray(self.D[np.asarray(closed)].sum(axis=0)).ravel() > 0)
        assert np.all(np.bincount(self.od[self.avail], minlength=net["n_od"]) > 0), "an O-D pair lost all its paths"
        self.qp = np.where(self.avail, self.q[self.od], 0.0)

    def tt(self, x):
        return self.n["fft"] * (1 + self.alpha * (x / self.cap) ** self.beta)

    def logit(self, x):
        V = self.D.T @ (self.th * self.tt(x) + self.lin)
        V = np.where(self.avail, V, -np.inf)
        vmax = np.full(self.n["n_od"], -np.inf); np.maximum.at(vmax, self.od, V)
        e = np.where(self.avail, np.exp(V - vmax[self.od]), 0.0)
        den = np.bincount(self.od, weights=e, minlength=self.n["n_od"])
        p = np.divide(e, den[self.od], out=np.zeros_like(e), where=den[self.od] > 0)
        return self.qp * p

    def fisk_deriv(self, f, d, lam):
        fl = f + lam * d
        x = self.D @ fl; dx = self.D @ d
        g = -(self.th * self.tt(x) + self.lin) @ dx
        m = self.avail & (fl > 0)
        return g + np.sum(d[m] * np.log(fl[m]))

    def solve(self, tol=1e-8, max_iter=5000, bis=40):
        f = self.logit(np.zeros(self.D.shape[0]))
        reason = "max_iter"
        for k in range(1, max_iter + 1):
            x = self.D @ f
            y = self.logit(x)
            r = np.abs(self.D @ y - x).sum() / max(np.abs(x).sum(), 1e-12)
            if r <= tol:
                reason = "tol"; break
            d = y - f
            lo, hi = 0.0, 1.0
            if self.fisk_deriv(f, d, 1.0) <= 0:
                lam = 1.0
            else:
                for _ in range(bis):
                    mid = 0.5 * (lo + hi)
                    if self.fisk_deriv(f, d, mid) > 0: hi = mid
                    else: lo = mid
                lam = 0.5 * (lo + hi)
            f = f + lam * d
        x = self.D @ f
        t = self.tt(x)
        return dict(f=f, x=x, t=t, iters=k, resid=r, reason=reason, sttt=float(x @ t))
