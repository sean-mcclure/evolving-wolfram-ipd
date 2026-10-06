"""
Second evolutionary rule: a finite-population Moran process with mutation.

N individuals. Each step one individual reproduces (chosen with probability
proportional to count * exp(beta * payoff), payoff measured against the
OTHER N-1 individuals), its offspring mutates with probability u using
Wolfram's mutation operators, and the offspring replaces a randomly chosen
individual. One generation = N steps. Drift is now real, so outcomes are
reported as averages over seeds.

  python moran.py --states 2 --noise 0.01 --seeds 5
  python moran.py --states 3 --noise 0.01 --seeds 3 --gens 3000
"""
import argparse, os, json
import numpy as np
import evolve as ev
from ensembles import build
from sweep import get_matrix, spearman


def run(A, K, N, beta, u, gens, rng, sample_every=None):
    n = len(A)
    c = rng.multinomial(N, np.full(n, 1 / n)).astype(float)   # Wolfram's uniform start
    raw_pi = A @ c
    Kcum = np.cumsum(K, axis=0)
    steps = gens * N
    sample_every = sample_every or N
    samples = []
    diagA = np.diag(A).copy()
    R = rng.random((steps, 4))          # pre-drawn randomness: birth, mutate?, mutant, death
    for t in range(steps):
        alive = c > 0
        pi = (raw_pi - diagA * alive) / (N - 1)
        w = c * np.exp(beta * (pi - pi[alive].max()))
        cw = np.cumsum(w)
        i = int(np.searchsorted(cw, R[t, 0] * cw[-1], side="right"))
        if R[t, 1] < u:
            i = int(np.searchsorted(Kcum[:, i], R[t, 2] * Kcum[-1, i], side="right"))
        cc = np.cumsum(c)
        j = int(np.searchsorted(cc, R[t, 3] * N, side="right"))
        if i != j:
            c[i] += 1; c[j] -= 1
            raw_pi += A[:, i] - A[:, j]
        if t % sample_every == 0:
            samples.append(c / N)
    return np.array(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--ensemble", default="wolfram", choices=["wolfram", "full"])
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--u", type=float, default=0.005, help="mutation prob per birth")
    ap.add_argument("--gens", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    machines, raw, cls, first = build(a.states, a.ensemble)
    ev.raw_index = {m: i for i, m in enumerate(raw)}
    n = len(machines)
    K = ev.mutation_kernel(raw, cls, n)
    A = get_matrix(machines, ev.PAYOFFS["prison"], first, a.noise,
                   os.path.join(a.out, f"A_{a.ensemble}_s{a.states}_prison_eps{a.noise}_v2.npy"))
    names = [ev.name_of(m, a.ensemble) for m in machines]
    T = [ev.traits(m, first) for m in machines]
    F0 = A.mean(1); srank = np.argsort(np.argsort(-F0)) + 1; top10 = np.argsort(-F0)[:10]

    pbars = []
    for seed in range(a.seeds):
        S = run(A, K, a.N, a.beta, a.u, a.gens, np.random.default_rng(seed))
        pbars.append(S[len(S) // 2:].mean(0))
        d = int(pbars[-1].argmax())
        print(f"seed {seed}: most abundant {names[d]} ({pbars[-1][d]:.2f}, static rank "
              f"{srank[d]}), static top-10 share {pbars[-1][top10].sum():.3f}", flush=True)
    pbar = np.mean(pbars, 0)
    summary = dict(
        static_top10_share=float(pbar[top10].sum()),
        rank_corr=spearman(F0, pbar),
        most_abundant={names[i]: [round(float(pbar[i]), 3), int(srank[i])]
                       for i in np.argsort(-pbar)[:6]},
        traits={k: round(float(pbar @ np.array([t[k] for t in T], float)), 3)
                for k in ("nice", "provocable", "forgiving", "tolerant")})
    print(json.dumps(summary, indent=2))
    tag = f"{a.ensemble}_s{a.states}_eps{a.noise}_N{a.N}_b{a.beta}_u{a.u}"
    with open(os.path.join(a.out, f"moran_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
