"""Checks every predicted state with MaTE's own route-choice map: per-period gap |y(x) - x|_1 / |x|_1."""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prepare import build_mate_network, city_model, make_X, mate_workdir, quiet  # noqa: E402
from reference import CAP_FACTOR, LEVELS, RUNS, TOLL, VOT, load_benchmark  # noqa: E402

TARGET = 1e-6  # MaTE runs in float32, whose floor for this gap is about 1e-7


def certify(model, net, ref):
    import tensorflow as tf
    from mate.models import BPR
    L, P = len(net.links), len(LEVELS)
    Z = np.load(os.path.join(RUNS, "static.npz"))
    Ps = np.load(os.path.join(RUNS, model, "params.npz"))
    st = np.load(os.path.join(RUNS, model, "states.npz"))
    pf = None
    if model == "bpr4":
        pf = BPR(keys=["alpha", "beta"], initial_values={"alpha": 1.0, "beta": 4.0}, trainables={"alpha": True, "beta": False},
                 capacities=[l.bpr.k for l in net.links], free_flow_traveltimes=[l.bpr.tf for l in net.links],
                 dtype=tf.float32)
    m = city_model(net, tf.constant(Z["g_prior"], tf.float32), performance_function=pf)
    X = make_X(P, L)
    m.endogenous_flows = True; m.build(); m.setup_period_ids(X_train=X)
    for var, key in [(m._fixed_effect, "fe_raw"), (m._fixed_effect_generation, "G_raw"), (m._fixed_effect_od, "U_raw")]:
        var.assign(np.asarray(Ps[key], np.float32).reshape(var.shape))
    pf = m.performance_function
    if model == "cubic":
        pf.polynomial_layer._pretrain_weights = False
        pf.polynomial_layer._poly_weights.assign(np.asarray(Ps["logc"], np.float32).reshape(pf.polynomial_layer._poly_weights.shape))
        pf.model.layers[0]._parameter.assign(np.float32(Ps["w_raw"]))
    else:
        pf._alpha.assign(np.float32(Ps["w_raw"])); pf._beta.assign(np.float32(np.log(4.0)))
    cap0 = ref["cap"].astype(np.float32)
    paths_full = {od: list(ps) for od, ps in net.paths_od.items()}

    def set_cap(c):
        pf.capacities = tf.constant(c, tf.float32)
        if hasattr(pf, "k"):
            pf.k = tf.constant(c, tf.float32)

    def apply(kind, j):
        if kind in CAP_FACTOR:
            c = cap0.copy(); c[j] *= CAP_FACTOR[kind]; set_cap(c)
        elif kind == "toll":
            p_ = np.zeros((1, L), np.float32); p_[0, j] = TOLL
            m.price_intervention = True; m.vot = VOT; m.p = tf.constant(p_)
        elif kind == "closure":
            lk = net.links[j]
            with quiet():
                net.load_paths(paths_od={od: [pp for pp in ps if lk not in pp.links] for od, ps in paths_full.items()},
                               update_incidence_matrices=True)

    def undo():
        set_cap(cap0)
        m.price_intervention = False; m.p = tf.constant(np.zeros((1, L), np.float32))
        with quiet():
            net.load_paths(paths_od=paths_full, update_incidence_matrices=True)

    def gap(x):
        m._flows.assign(np.sqrt(np.maximum(x, 0)).astype(np.float32))
        y = m.compute_link_flows(X).numpy().astype(np.float64)
        return [float(np.abs(y[p] - x[p]).sum() / np.abs(x[p]).sum()) for p in range(P)]

    rows = [dict(state="baseline", gaps=gap(np.stack([st[f"baseline_p{p}"] for p in range(P)])))]
    cases = pd.read_csv(os.path.join(RUNS, "scenarios.csv")).drop_duplicates("case")
    for case, j, kind in cases[["case", "link_index", "intervention"]].values.tolist():
        apply(kind, j)
        rows.append(dict(state=case, gaps=gap(np.stack([st[f"{case}_p{p}"] for p in range(P)]))))
        undo()
    worst = max(max(r["gaps"]) for r in rows)
    ok = sum(max(r["gaps"]) <= TARGET for r in rows)
    json.dump(dict(states=rows, max_gap=worst, target=TARGET, n_states=len(rows), n_meeting=ok),
              open(os.path.join(RUNS, model, "certify.json"), "w"), indent=1)
    print(f"{model}: {ok}/{len(rows)} states with MaTE gap <= {TARGET:g} at both demand levels; max gap {worst:.2e}")
    return ok == len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["cubic", "bpr4"], nargs="*", default=["cubic", "bpr4"])
    models = ap.parse_args().model
    ref = load_benchmark()
    with mate_workdir():
        net = build_mate_network(ref)
        ok = [certify(m, net, ref) for m in models]
    assert all(ok), "some predicted states are not MaTE equilibria"


if __name__ == "__main__":
    main()
