"""
Robustness sweep for the replicator experiment.

For every combination of ensemble x states x noise x mutation x selection
strength, start from Wolfram's uniform population, evolve, and record
time-averaged outcomes over the final quarter of the run (so cycling runs
are summarised fairly rather than by a single snapshot).

  python sweep.py --states 2          # ~1 min
  python sweep.py --states 3 --ensemble wolfram   # run these two in
  python sweep.py --states 3 --ensemble full      # separate terminals
Writes results/sweep_s{S}_{ensemble}.csv after every run.
"""
import argparse, csv, os, itertools
import numpy as np
import evolve as ev
from ensembles import build


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def get_matrix(machines, P, first, eps, path):
    if os.path.exists(path):
        return np.load(path)
    A = ev.payoff_matrix_exact(machines, P, first) if eps == 0 else \
        ev.payoff_matrix_noisy(machines, P, eps, first=first)
    np.save(path, A); return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--horizon", type=float, default=None,
                    help="run length in units of beta*generations")
    ap.add_argument("--ensemble", default="both", choices=["both", "wolfram", "full"],
                    help="run one ensemble per terminal to use two cores")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    horizon = a.horizon or (20000 if a.states == 2 else 6000)
    P = ev.PAYOFFS["prison"]
    rows = []
    ens_list = ("wolfram", "full") if a.ensemble == "both" else (a.ensemble,)
    out = os.path.join(a.out, f"sweep_s{a.states}_{a.ensemble}.csv")
    for ens in ens_list:
        machines, raw, cls, first = build(a.states, ens)
        ev.raw_index = {m: i for i, m in enumerate(raw)}
        n = len(machines)
        K = ev.mutation_kernel(raw, cls, n)
        T = [ev.traits(m, first) for m in machines]
        trait_keys = ["nice", "provocable", "forgiving", "tolerant"]
        names = [ev.name_of(m, ens) for m in machines]
        for eps in (0.0, 0.01, 0.05):
            A = get_matrix(machines, P, first, eps,
                           os.path.join(a.out, f"A_{ens}_s{a.states}_prison_eps{eps}_v2.npy"))
            F0 = A.mean(1); static_rank = np.argsort(np.argsort(-F0)) + 1
            top10 = np.argsort(-F0)[:10]
            for mu, beta in itertools.product((0.0, 0.001), (0.1, 0.3, 1.0)):
                gens = int(horizon / beta)
                hist, meanF = ev.evolve(A, K, gens, beta, mu, np.full(n, 1 / n))
                tail = hist[-gens // 4:]
                pbar = tail.mean(0)
                i_var = tail.var(0).argmax()
                swing = float(np.ptp(tail[:, i_var]))
                dom = int(pbar.argmax())
                row = dict(ensemble=ens, states=a.states, n=n, noise=eps, mu=mu,
                           beta=beta, gens=gens,
                           static_top10_share=round(float(pbar[top10].sum()), 4),
                           static_winner_share=round(float(pbar[top10[0]]), 4),
                           rank_corr=round(spearman(F0, pbar), 3),
                           dominant=names[dom], dominant_share=round(float(pbar[dom]), 3),
                           dominant_static_rank=int(static_rank[dom]),
                           oscillating=swing > 0.05, swing=round(swing, 3),
                           mean_payoff=round(float(meanF[-gens // 4:].mean()), 3))
                for k in trait_keys:
                    row[k] = round(float(pbar @ np.array([t[k] for t in T], float)), 3)
                rows.append(row)
                with open(out, "w", newline="") as f:      # save as we go
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
                print({k: row[k] for k in ("ensemble", "noise", "mu", "beta",
                       "static_top10_share", "rank_corr", "dominant",
                       "dominant_static_rank", "oscillating", "nice", "forgiving", "tolerant")},
                      flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
