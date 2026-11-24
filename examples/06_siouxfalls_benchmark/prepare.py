"""Builds the route catalogue, reference solutions, the 24 cases, baseline observations, O-D prior and MaTE start."""
import contextlib
import hashlib
import io
import itertools
import json
import os
import sys
import tempfile
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU keeps the float32 initialisation reproducible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import engine  # noqa: E402
from reference import CATALOGUE, LEVELS, RUNS, TNTP_DIR, VOT, SUE, intervention, load_benchmark, load_network  # noqa: E402

FIGURE6_CATALOGUE_SHA256 = "46f793bae1ab0bfbdc54604ef82c027cf08dd06ab1926efbd5e51a86999ec162"
K = 5
TOL = 1e-8
SEED = 20261110
SEED_OD = [20261110, 100]
INTERVENTIONS = ("cap+25%", "cap-50%", "closure", "toll")
SCREEN, THRESH, N_LINKS = ("cap+25%", "cap-50%", "toll"), 0.5, 3


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def mate_workdir():
    """Temporary working directory under runs/, since MaTE writes scratch files under ./output."""
    warnings.filterwarnings("ignore")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(dir=RUNS) as wd:
        os.chdir(wd)
        try:
            yield
        finally:
            os.chdir(cwd)


def sha256_file(fn):
    return hashlib.sha256(open(fn, "rb").read()).hexdigest()


def build_catalogue():
    """k = 5 free-flow shortest loopless paths per O-D, plus closure alternatives so no single closure strands an O-D."""
    net = load_network()
    links = list(zip(net["init"].tolist(), net["term"].tolist()))
    lid = {l: i for i, l in enumerate(links)}
    G = nx.DiGraph()
    for i, (u, v) in enumerate(links):
        G.add_edge(u, v, fft=float(net["fft"][i]))
    ods = net["ods"]

    def path_links(p):
        return [lid[(p[i], p[i + 1])] for i in range(len(p) - 1)]

    ps = {od: [list(p) for p in itertools.islice(nx.shortest_simple_paths(G, od[0], od[1], weight="fft"), K)]
          for od in ods}
    src = {od: ["k%d_rank%d" % (K, r + 1) for r in range(len(ps[od]))] for od in ods}
    for j, (u, v) in enumerate(links):
        H = G.copy(); H.remove_edge(u, v)
        for od in ods:
            if all(j in path_links(p) for p in ps[od]):
                alt = nx.shortest_path(H, od[0], od[1], weight="fft")
                if alt not in ps[od]:
                    ps[od].append(list(alt)); src[od].append("closure_alt_link%d_%d-%d" % (j, u, v))
    for j in range(len(links)):
        assert not any(all(j in path_links(p) for p in ps[od]) for od in ods), f"closing link {j} strands an O-D"
    paths = [{"od_index": i, "od": [int(od[0]), int(od[1])], "nodes": [int(n) for n in p],
              "links": [int(j) for j in path_links(p)], "source": s}
             for i, od in enumerate(ods) for p, s in zip(ps[od], src[od])]
    meta = {
        "name": "SiouxFalls route catalogue v1",
        "rule": "free-flow k=5 loopless shortest paths (Yen, networkx.shortest_simple_paths, weight=TNTP free_flow_time)"
                " + for every link j and every O-D all of whose k paths use j, the free-flow shortest path avoiding j",
        "k": K,
        "networkx_version": nx.__version__,
        "node_ids": "TNTP 1-based node ids",
        "link_order": "row order of SiouxFalls_net.tntp; 'links' entries are [init, term]; path 'links' are 0-based"
                      " indices into this list",
        "od_order": "origin ascending then destination ascending, pairs with demand > 0 only",
        "tntp_sha256": {f: sha256_file(os.path.join(TNTP_DIR, f)) for f in ["SiouxFalls_net.tntp", "SiouxFalls_trips.tntp"]},
        "q_note": "q copied from SiouxFalls_trips.tntp for equality checks only; every solver must parse its own q and"
                  " assert equality",
    }
    cat = {"meta": meta, "links": [[int(u), int(v)] for (u, v) in links], "ods": [[int(o), int(d)] for (o, d) in ods],
           "q": [float(x) for x in net["q"]], "paths": paths}
    with open(CATALOGUE, "w") as fh:
        json.dump(cat, fh, sort_keys=True, separators=(",", ":"))
    digest = sha256_file(CATALOGUE)
    print(f"route catalogue: {len(paths)} paths for {len(ods)} O-D pairs; sha256 {digest}"
          + ("" if digest == FIGURE6_CATALOGUE_SHA256 else " (differs from the catalogue used for Figure 6)"))


def reference_solutions(net):
    """Baselines, link screen at q, seeded draw of 3 links and the 24 reference cases."""
    L = len(net["links"])
    base = [SUE(net, demand_scale=s).solve(tol=TOL) for s in LEVELS]
    pct = {(j, k): (SUE(net, **intervention(net, k, j)).solve(tol=TOL)["sttt"] / base[0]["sttt"] - 1) * 100
           for j in range(L) for k in SCREEN}
    elig = [j for j in range(L) if all(abs(pct[j, k]) >= THRESH for k in SCREEN)]
    chosen = []
    for j in [elig[i] for i in np.random.default_rng(SEED).permutation(len(elig))]:
        u, v = net["links"][j]
        if any(net["links"][c] == (v, u) for c in chosen):
            continue
        chosen.append(j)
        if len(chosen) == N_LINKS:
            break
    print(f"links with |dSTTT| >= {THRESH}% at q for {', '.join(SCREEN)}: {elig}; drawn: {chosen}")
    cases = [(f"L{r + 1}-{kind}", j, kind) for r, j in enumerate(chosen) for kind in INTERVENTIONS]
    case_x, case_t, rows = np.zeros((len(cases), len(LEVELS), L)), np.zeros((len(cases), len(LEVELS), L)), []
    for c, (case, j, kind) in enumerate(cases):
        for p, s in enumerate(LEVELS):
            o = SUE(net, demand_scale=s, **intervention(net, kind, j)).solve(tol=TOL)
            assert o["reason"] == "tol", (case, s, o["resid"])
            case_x[c, p], case_t[c, p] = o["x"], o["t"]
            rows.append(dict(case=case, link_index=j, init=net["links"][j][0], term=net["links"][j][1], intervention=kind,
                             level=s, sttt_ref_base=base[p]["sttt"], sttt_ref=o["sttt"], dsttt_ref=o["sttt"] - base[p]["sttt"],
                             dsttt_ref_pct=(o["sttt"] / base[p]["sttt"] - 1) * 100, resid=o["resid"], iters=o["iters"]))
    np.savez(os.path.join(RUNS, "reference_solutions.npz"), base_x=np.stack([b["x"] for b in base]),
             base_t=np.stack([b["t"] for b in base]), case_x=case_x, case_t=case_t, levels=np.array(LEVELS))
    pd.DataFrame(rows).to_csv(os.path.join(RUNS, "scenarios.csv"), index=False)
    np.savez(os.path.join(RUNS, "observations.npz"), x_obs=np.stack([b["x"] for b in base]),
             t_obs=np.stack([b["t"] for b in base]), levels=np.array(LEVELS))
    print("reference baseline STTT: " + ", ".join(f"{s}q {b['sttt']:.1f}" for s, b in zip(LEVELS, base)))


def build_mate_network(ref):
    """MaTE's own TNTP network with the catalogue paths, checked against the reference specification."""
    with quiet():
        from pesuelogit.networks import build_tntp_network
        from isuelogit.paths import Path
        from mate.etl import read_tntp_od
    folder = os.path.dirname(TNTP_DIR) + os.sep
    with quiet():
        net = build_tntp_network(network_name="SiouxFalls", folderpath=folder)
        net.load_OD(Q=read_tntp_od(network_name="SiouxFalls", folderpath=folder))
    assert [(int(l.key[0]) + 1, int(l.key[1]) + 1) for l in net.links] == ref["links"]
    assert np.allclose([l.bpr.tf for l in net.links], ref["fft"]) and np.allclose([l.bpr.k for l in net.links], ref["cap"])
    assert [(int(o) + 1, int(d) + 1) for (o, d) in net.ods] == [tuple(o) for o in ref["ods"]]
    assert np.array_equal(np.asarray(net.OD.q_true).flatten().astype(float), ref["q"])
    lk = {(int(l.key[0]) + 1, int(l.key[1]) + 1): l for l in net.links}
    paths_od = {od: [] for od in net.ods}
    for p in ref["paths"]:
        od = net.ods[p["od_index"]]
        paths_od[od].append(Path(origin=od[0], destination=od[1], links=[lk[ref["links"][a]] for a in p["links"]]))
    with quiet():
        net.load_paths(paths_od=paths_od, update_incidence_matrices=True)
    assert np.array_equal(np.asarray(net.D), ref["D"].toarray())
    assert np.array_equal(np.asarray(net.M).argmax(0), ref["od"])
    net.links_len = ref["length"].copy()
    return net


def make_X(n_periods, n_links):
    import tensorflow as tf
    pid = np.repeat(np.arange(n_periods, dtype=np.float32)[:, None], n_links, axis=1)
    return tf.constant(pid[:, :, None], dtype=tf.float32)


def city_model(net, reference_g, performance_function=None):
    """MaTE city configuration: theta = -1 fixed, link fixed effects, generation x distribution demand."""
    import tensorflow as tf
    from mate.models import create_mattwin_model
    with quiet():
        return create_mattwin_model(network=net, n_periods=len(LEVELS), features_Z_utility=[], reference_g=reference_g,
                                    performance_function=performance_function, dtype=tf.float32, vot=VOT,
                                    p=np.zeros((1, len(net.links))), price_intervention=False)


def mate_start(ref):
    """O-D prior, MaTE's initialisation (single-pass loading and cubic pretraining) and the engine inputs."""
    import scipy.sparse as sps
    import tensorflow as tf
    tf.random.set_seed(SEED); np.random.seed(SEED)
    net = build_mate_network(ref)
    from mate.metrics import z2score
    from mate.models import compute_generated_trips
    L, P = len(net.links), len(LEVELS)
    obs = np.load(os.path.join(RUNS, "observations.npz"))
    X = make_X(P, L)
    Y = tf.constant(np.stack([obs["t_obs"], obs["x_obs"]], -1), tf.float32)

    # O-D prior: true demand plus N(0, sd^2) noise, sd = 10% of the mean cell of a 24 x 24 matrix
    rng = np.random.default_rng(SEED_OD)
    q_true = np.stack([s * ref["q"] for s in LEVELS])
    q_prior = q_true.copy()
    for p in range(P):
        sd = 0.1 * q_true[p].sum() / (24 * 24)
        q_prior[p] = q_true[p] + rng.normal(loc=0.0, scale=sd, size=q_true.shape[1])
        q_prior[p][q_prior[p] < 0] = 1e-7
    reference_g = compute_generated_trips(q=tf.stack(q_prior), ods=net.ods, n_nodes=len(net.nodes))

    m = city_model(net, reference_g)
    m.endogenous_flows = True; m.build()
    m.setup_period_ids(X_train=X)
    pf0 = m.compute_link_flows(X)
    pf0 = tf.concat([tf.expand_dims(tf.reduce_mean(lf, axis=0), 0) for lf in m.split_link_flows_by_period(pf0)], axis=0)
    m._flows.assign(tf.math.sqrt(pf0))
    pl = m.performance_function.polynomial_layer
    with quiet():
        pl.pretrain_weights(flows=tf.math.pow(m._flows, 2), free_flow_traveltimes=m.tt_ff,
                            capacities=m.performance_function.capacities)
    pl._pretrain_weights = False
    init = dict(fe_raw=m._fixed_effect.numpy(), G_raw=m._fixed_effect_generation.numpy(), U_raw=m._fixed_effect_od.numpy(),
                logc=pl._poly_weights.numpy(), w_raw=m.performance_function.model.layers[0]._parameter.numpy())
    theta = m.theta.numpy().reshape(-1).astype(np.float64)
    assert np.all(theta == -1.0), theta

    D = sps.csr_matrix(np.asarray(net.D, dtype=np.float64)); Mc = sps.csc_matrix(np.asarray(net.M, dtype=np.float64))
    pf = m.performance_function
    static = dict(D_data=D.data, D_indices=D.indices, D_indptr=D.indptr, D_shape=np.array(D.shape),
                  od=Mc.indices.astype(np.int64), n_od=Mc.shape[0], orig=np.asarray(m.o, np.int64),
                  n_nodes=np.asarray(reference_g).shape[1], refmask=np.asarray(reference_g) > 0,
                  tf_=np.asarray(pf.free_flow_traveltimes.numpy(), np.float64),
                  cap=np.asarray(pf.capacities.numpy(), np.float64), theta=theta,
                  links_len=np.asarray(net.links_len, np.float64),
                  obs_flow=m.observed_flow(Y=Y).numpy().astype(np.float64),
                  obs_tt=m.observed_traveltime(Y=Y).numpy().astype(np.float64),
                  q_true=q_true, q_prior=q_prior, g_prior=np.asarray(reference_g, np.float64))
    np.savez(os.path.join(RUNS, "static.npz"), **static)
    x_init = m._flows.numpy().astype(np.float64) ** 2
    np.savez(os.path.join(RUNS, "init.npz"), x_init=x_init, **{k: np.asarray(v, np.float64) for k, v in init.items()})

    # the numpy engine must reproduce MaTE's demand, route-choice map, travel times and loss at the start
    S = engine.Static(os.path.join(RUNS, "static.npz"))
    y_tf = m.compute_link_flows(X).numpy().astype(np.float64)
    t_tf = m.traveltimes(flows=tf.constant(y_tf, tf.float32)).numpy().astype(np.float64)
    q_tf = m.q.numpy().astype(np.float64)
    wq = wm = wt = loss_np = 0.0
    for h in range(P):
        H = engine.Hour(S, h)
        H.set_params(init["fe_raw"], init["G_raw"][h], init["U_raw"][h], init["logc"], init["w_raw"])
        wq = max(wq, float(np.max(np.abs(H.q - q_tf[h])) / np.max(np.abs(q_tf))))
        y = S.D @ H.pathflows(x_init[h])[0]
        wm = max(wm, float(np.abs(y - y_tf[h]).sum() / np.abs(y_tf[h]).sum()))
        wt = max(wt, float(np.max(np.abs(engine.tt(S, y_tf[h], H.w, H.c)[0] - t_tf[h]) / t_tf[h])))
        loss_np += H.loss_terms(y_tf[h])[0]
    lm = m.loss_function(X=X, Y=Y, lambdas={"od": 0, "traveltime": 1, "flow": 1, "equilibrium": 1}, loss_metric=z2score)
    loss_tf = (float(lm["loss_flow"]) + float(lm["loss_traveltime"])) / 3.0
    alpha = float(np.exp(init["w_raw"]) * np.exp(init["logc"]).reshape(-1)[-1])
    chk = dict(cubic_start_alpha=alpha, q_max_rel=wq, map_max_rel_l1=wm, tt_max_rel=wt, loss_numpy=loss_np, loss_mate=loss_tf,
               loss_rel_diff=abs(loss_np - loss_tf) / abs(loss_tf))
    chk["passed"] = bool(max(wq, wm, wt, chk["loss_rel_diff"]) <= 1e-6)
    json.dump(chk, open(os.path.join(RUNS, "engine_check.json"), "w"), indent=1)
    print(f"MaTE start: cubic t0 (1 + {alpha:.4f} u^3); engine vs MaTE max rel. diff {max(wq, wm, wt):.1e}, "
          f"loss {chk['loss_rel_diff']:.1e}: {'passed' if chk['passed'] else 'FAILED'}")
    assert chk["passed"], chk


def main():
    for f in ("SiouxFalls_net.tntp", "SiouxFalls_trips.tntp"):
        if not os.path.exists(os.path.join(TNTP_DIR, f)):
            sys.exit(f"{f} not found: run python download_tntp.py first")
    os.makedirs(RUNS, exist_ok=True)
    build_catalogue()
    ref = load_benchmark()
    reference_solutions(ref)
    with mate_workdir():
        mate_start(ref)


if __name__ == "__main__":
    main()
