"""Calibrates on baseline observations only (exact SUE + adjoint gradient + Adam), then re-solves the 24 cases."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import engine  # noqa: E402
from reference import CAP_FACTOR, LEVELS, RUNS, TOLL, VOT  # noqa: E402

STEPS = 960
LR = dict(fe_raw=0.1, G_raw=10.0, U_raw=0.1, logc=0.1, w_raw=0.1)
SOLVE = dict(tol=1e-6, max_iter=20000)
PRED = dict(tol=1e-8, max_iter=50000)
MODELS = {"cubic": "cubic (city configuration)", "bpr4": "quartic BPR"}


def start(model):
    """Shared start; the cubic keeps MaTE's pretrained polynomial, the quartic BPR starts at alpha = 1 with beta = 4 fixed."""
    I = dict(np.load(os.path.join(RUNS, "init.npz")))
    groups = ["fe_raw", "G_raw", "U_raw", "logc", "w_raw"]
    if model == "bpr4":
        I["logc"], I["w_raw"] = np.array([-np.inf, -np.inf, -np.inf, 0.0]), 0.0
        groups.remove("logc")
    return I, groups


def alpha(Ps):
    return float(np.exp(Ps["w_raw"]) * np.exp(np.asarray(Ps["logc"]).reshape(-1)[-1]))


def gradcheck(S, I, groups):
    """Central finite differences of the equilibrium loss along random directions (tight solves)."""
    rng = np.random.default_rng(2024)
    P0 = dict(I); P0["fe_raw"] = -0.05 * np.abs(rng.standard_normal(S.n_links))
    eps = dict(fe_raw=1e-4, G_h=1e-2, U_h=1e-4, logc=1e-5, w_raw=1e-5)
    keys = [{"G_raw": "G_h", "U_raw": "U_h"}.get(g, g) for g in groups]
    res = []
    for h in range(len(LEVELS)):
        H = engine.Hour(S, h)
        H.set_params(P0["fe_raw"], P0["G_raw"][h], P0["U_raw"][h], P0["logc"], P0["w_raw"])
        _, G, _ = H.value_and_grad(x_init=I["x_init"][h], tol=1e-6, polish_tol=1e-13, max_iter=3000, gmres_tol=1e-12)
        Pst = H.P.copy(); r2 = np.random.default_rng(100 + h)
        dirs = dict(fe_raw=r2.standard_normal(S.n_links), G_h=r2.standard_normal(S.n_nodes) * (S.refmask[h] > 0),
                    U_h=r2.standard_normal(S.n_od), logc=np.array([0, 0, 1.0]), w_raw=1.0)
        dirs = {k: v / np.abs(v).max() for k, v in dirs.items()}
        for key in keys:
            vals = []
            for s in (+1, -1):
                p = dict(fe_raw=P0["fe_raw"].copy(), G_h=P0["G_raw"][h].copy(), U_h=P0["U_raw"][h].copy(),
                         logc=np.asarray(P0["logc"]).reshape(-1).copy(), w_raw=float(P0["w_raw"]))
                p[key] = p[key] + s * eps[key] * dirs[key]
                H.set_params(p["fe_raw"], p["G_h"], p["U_h"], p["logc"], p["w_raw"]); H.P = Pst.copy()
                x, info = H.equilibrate(tol=1e-6, polish_tol=1e-13, max_iter=3000)
                assert info["reason"] == "tolerance", info
                vals.append(H.loss_terms(x)[0])
            fd = (vals[0] - vals[1]) / (2 * eps[key]); an = float(np.sum(np.asarray(G[key]) * dirs[key]))
            res.append(dict(period=h, group=key, analytic=an, finite_difference=fd,
                            rel_err=abs(an - fd) / max(abs(fd), abs(an), 1e-300)))
    worst = max(r["rel_err"] for r in res if max(abs(r["analytic"]), abs(r["finite_difference"])) > 1e-14)
    return dict(results=res, worst_rel_err=worst, passed=bool(worst <= 1e-3))


def calibrate(S, I, groups, lr_scale, log):
    Ps = {k: np.array(I[k], np.float64) for k in ["fe_raw", "G_raw", "U_raw", "logc", "w_raw"]}
    hours = [engine.Hour(S, h) for h in range(len(LEVELS))]
    adam = engine.Adam(); lrs = {k: v * lr_scale for k, v in LR.items()}
    rows, loss0 = [], None
    for step in range(STEPS + 1):
        res = []
        for h, H in enumerate(hours):
            H.set_params(Ps["fe_raw"], Ps["G_raw"][h], Ps["U_raw"][h], Ps["logc"], Ps["w_raw"])
            res.append(H.value_and_grad(x_init=I["x_init"][h], **SOLVE))
        loss = float(sum(r[0] for r in res)); diags = [r[2] for r in res]
        failed = any(d["solve"]["reason"] != "tolerance" for d in diags)
        loss0 = loss if loss0 is None else loss0
        rows.append(dict(step=step, loss=loss, loss_count=sum(d["loss_count"] for d in diags),
                         loss_speed=sum(d["loss_speed"] for d in diags), alpha=alpha(Ps),
                         count_mape_pct=100 * float(np.mean(np.concatenate([np.abs(H.x[H.cl] - H.cv) / H.cv for H in hours]))),
                         speed_mape_pct=100 * sum(d["ape_speed_sum"] for d in diags) / sum(d["n_speed"] for d in diags),
                         demand_q=diags[0]["demand"], demand_08q=diags[1]["demand"],
                         solve_iters_max=max(d["solve"]["iters"] for d in diags)))
        if step % 160 == 0:
            r = rows[-1]
            log(f"step {step:4d} loss {loss:.5g} count MAPE {r['count_mape_pct']:.2f}% speed MAPE {r['speed_mape_pct']:.2f}% "
                f"alpha {r['alpha']:.4f}")
        if not np.isfinite(loss) or loss > 2 * loss0 or failed:
            log(f"unstable at step {step}")
            return Ps, rows, False
        if step == STEPS:
            break
        grads = dict(fe_raw=sum(r[1]["fe_raw"] for r in res), logc=sum(r[1]["logc"] for r in res),
                     w_raw=sum(r[1]["w_raw"] for r in res),
                     G_raw=np.stack([r[1]["G_h"] for r in res]), U_raw=np.stack([r[1]["U_h"] for r in res]))
        Ps = adam.step(Ps, {k: grads[k] for k in groups}, lrs)
    return Ps, rows, True


def solve_state(S, Ps, x_init, h, kind=None, j=None, Pwarm=None):
    """Re-solves the calibrated equilibrium of period h with an intervention on link j (None = baseline)."""
    S_, lin_extra, closed = S, None, None
    if kind in CAP_FACTOR:
        c = S.cap.copy(); c[j] *= CAP_FACTOR[kind]; S_ = S.with_cap(c)
    elif kind == "toll":
        lin_extra = np.zeros(S.n_links); lin_extra[j] = S.theta[h] * TOLL / VOT
    elif kind == "closure":
        closed = np.asarray(S.D[j].toarray()).ravel() > 0
    H = engine.Hour(S_, h); H.set_intervention(lin_extra=lin_extra, closed_paths=closed)
    H.set_params(Ps["fe_raw"], Ps["G_raw"][h], Ps["U_raw"][h], Ps["logc"], Ps["w_raw"])
    if Pwarm is not None:
        Pw = Pwarm.copy()
        if closed is not None:
            Pw[closed] = 0.0
            Pw = Pw / np.bincount(S.od, weights=Pw, minlength=S.n_od)[S.od]
        H.P = Pw
    x, info = H.equilibrate(x_init=x_init[h], **PRED)
    t = engine.tt(S_, x, H.w, H.c)[0]
    return dict(x=x, t=t, sttt=float(x @ t), info=info, H=H)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    model = ap.parse_args().model
    out = os.path.join(RUNS, model); os.makedirs(out, exist_ok=True)
    t0 = time.time()

    def log(*a):
        print(f"[{model} {time.time() - t0:6.1f}s]", *a, flush=True)

    S = engine.Static(os.path.join(RUNS, "static.npz"))
    I, groups = start(model)
    gc = gradcheck(S, I, groups)
    json.dump(gc, open(os.path.join(out, "gradcheck.json"), "w"), indent=1)
    log(f"gradient check worst rel. error {gc['worst_rel_err']:.1e}: {'passed' if gc['passed'] else 'FAILED'}")
    assert gc["passed"], "gradient check failed"

    Ps, trace, ok = calibrate(S, I, groups, 1.0, log)
    if not ok:  # restart once at half the learning rates
        Ps, trace2, ok = calibrate(S, I, groups, 0.5, log)
        trace += trace2
    assert ok, "calibration did not complete"
    pd.DataFrame(trace).to_csv(os.path.join(out, "calib_log.csv"), index=False)
    np.savez(os.path.join(out, "params.npz"), **Ps)

    ref = np.load(os.path.join(RUNS, "reference_solutions.npz"))
    cases = pd.read_csv(os.path.join(RUNS, "scenarios.csv")).drop_duplicates("case")[["case", "link_index", "intervention"]]
    rows, states = [], {}
    for h, s in enumerate(LEVELS):
        b = solve_state(S, Ps, I["x_init"], h)
        assert b["info"]["reason"] == "tolerance", b["info"]
        states[f"baseline_p{h}"] = b["x"]
        x0, tb = ref["base_x"][h], float(ref["base_x"][h] @ ref["base_t"][h])
        for c, (case, j, kind) in enumerate(cases.values.tolist()):
            st = solve_state(S, Ps, I["x_init"], h, kind, j, Pwarm=b["H"].P)
            assert st["info"]["reason"] == "tolerance", (case, st["info"])
            states[f"{case}_p{h}"] = st["x"]
            x1 = ref["case_x"][c, h]
            d_ref = float(x1 @ ref["case_t"][c, h]) - tb
            d_pred = st["sttt"] - b["sttt"]
            rows.append(dict(case=case, link_index=j, intervention=kind, level=s, sttt_ref_base=tb, dsttt_ref=d_ref,
                             dsttt_pred=d_pred, rel_error=(d_pred - d_ref) / abs(d_ref), error_over_base=(d_pred - d_ref) / tb,
                             sign_ok=bool(np.sign(d_pred) == np.sign(d_ref)),
                             flow_error=float(np.abs((st["x"] - b["x"]) - (x1 - x0)).sum() / np.abs(x0).sum()),
                             sttt_pred_base=b["sttt"], solve_resid=st["info"]["resid"]))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "cases.csv"), index=False)
    np.savez_compressed(os.path.join(out, "states.npz"), **states)
    fin = trace[-1]
    summary = dict(model=MODELS[model], alpha=alpha(Ps), final_loss=float(fin["loss"]), count_mape_pct=float(fin["count_mape_pct"]),
                   speed_mape_pct=float(fin["speed_mape_pct"]), gradcheck_worst_rel_err=gc["worst_rel_err"],
                   sign_agreement=f"{int(df.sign_ok.sum())}/{len(df)}", median_abs_rel_error=float(df.rel_error.abs().median()),
                   max_abs_error_over_base=float(df.error_over_base.abs().max()), seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    power = 3 if model == "cubic" else 4
    log(f"done: t0 (1 + {summary['alpha']:.4f} u^{power}); sign {summary['sign_agreement']}; "
        f"median |rel. error| {summary['median_abs_rel_error']:.3f}; max |error| / base STTT {summary['max_abs_error_over_base']:.4f}")


if __name__ == "__main__":
    main()
