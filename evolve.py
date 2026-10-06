"""
Strogatz's proposed experiment, on Wolfram's finite-state-machine strategies.

Step 1 (Wolfram): every distinct s-state machine plays every other in the
iterated Prisoner's Dilemma; rank by average payoff against the uniform field.
Step 2 (Strogatz): use those scores to reweight the population, recompute
fitness against the NEW population, and repeat. Watch p(t) flow.

Usage examples
  python evolve.py --states 2
  python evolve.py --states 2 --noise 0.02 --mu 0.001
  python evolve.py --states 3 --noise 0.01 --mu 0.0005 --gens 20000
"""
import argparse, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fsm_core import canonical, play_exact, response
from ensembles import build, nonempty_key

# Payoffs as Wolfram appears to use them (prison-years convention):
# both cooperate -1, sucker -3, temptation 0, both defect -2.
PAYOFFS = {
    "prison": [[-1, -3], [0, -2]],   # P[my_move][opp_move], 0=C 1=D
    "axelrod": [[3, 0], [5, 1]],
}

NAMES = {
    canonical(((0, 0, 0),)): "AllC",
    canonical(((1, 0, 0),)): "AllD",
    canonical(((0, 0, 1), (1, 0, 1))): "TitForTat",
    canonical(((0, 0, 1), (1, 1, 1))): "GrimTrigger",
    canonical(((0, 0, 1), (1, 1, 0))): "WinStayLoseShift",
    canonical(((1, 1, 0), (0, 1, 0))): "SuspiciousTFT",
    canonical(((0, 1, 1), (1, 0, 0))): "Alternator(CD)",
    canonical(((1, 1, 1), (0, 0, 0))): "Alternator(DC)",
}


# In the 'wolfram' ensemble every machine opens with C, so machines that
# differ only in their opening move merge (e.g. Suspicious TFT becomes TFT).
NAMES_NE = {nonempty_key(k, 6): v for k, v in NAMES.items()
            if v not in ("SuspiciousTFT", "Alternator(DC)")}
NAMES_NE[nonempty_key(canonical(((1, 0, 0),)), 6)] = "AllD(opens C)"


def name_of(k, mode="full"):
    if mode == "full":
        return NAMES.get(k, "M" + "".join(f"{c}{a}{b}" for c, a, b in k))
    k2 = canonical(k)
    return NAMES_NE.get(nonempty_key(k, 6), "W" + "".join(f"{c}{a}{b}" for c, a, b in k2))


# ---------------------------------------------------------------- payoffs
def payoff_matrix_exact(machines, P, first=None):
    n = len(machines)
    A = np.empty((n, n))
    for i, a in enumerate(machines):
        for j, b in enumerate(machines):
            A[i, j] = play_exact(a, b, P, first)
    return A


def with_opening(m, first):
    """Equivalent machine whose opening move is `first`: a new start state
    with that colour, wired exactly like the old start state."""
    c0, a0, b0 = m[0]
    return ((first, a0 + 1, b0 + 1),) + tuple((c, a + 1, b + 1) for c, a, b in m)


def payoff_matrix_noisy(machines, P, eps, log2_steps=10, first=None):
    """If `first` is set, every machine opens with that (intended) move,
    matching the 'wolfram' ensemble convention used in the exact case.
Each intended move is flipped with probability eps (trembling hand).
    Play is then a Markov chain on joint states; we take the Cesaro mean of
    the payoff over 2**log2_steps rounds starting from (0,0), via doubling."""
    if first is not None:
        machines = [with_opening(m, first) for m in machines]
    n, s = len(machines), max(len(m) for m in machines)
    col = np.zeros((n, s), int); nxt = np.zeros((n, s, 2), int)
    for i, m in enumerate(machines):
        for q, (c, a, b) in enumerate(m):
            col[i, q], nxt[i, q] = c, (a, b)
    P = np.array(P, float)
    S = s * s
    A = np.empty((n, n))
    q1, q2 = np.divmod(np.arange(S), s)
    for i in range(n):
        T = np.zeros((n, S, S)); r = np.zeros((n, S))
        for x1 in (0, 1):            # actual move of i
            for x2 in (0, 1):        # actual move of opponent
                p1 = np.where(col[i, q1] == x1, 1 - eps, eps)              # (S,)
                p2 = np.where(col[:, q2] == x2, 1 - eps, eps)              # (n,S)
                prob = p1[None, :] * p2
                dest = nxt[i, q1, x2][None, :] * s + nxt[:, q2, x1]       # (n,S)
                np.add.at(T, (np.arange(n)[:, None], np.arange(S)[None, :], dest), prob)
                r += prob * P[x1, x2]
        # Cesaro sum via doubling: Sum_{t<2^k} T^t
        Pw, Sm = T.copy(), np.broadcast_to(np.eye(S), (n, S, S)).copy()
        Sm = Sm + T
        Pw = T @ T
        for _ in range(log2_steps - 1):
            Sm = Sm + Sm @ Pw
            Pw = Pw @ Pw
        steps = 2 ** log2_steps
        A[i] = (Sm[:, 0, :] * r).sum(1) / steps
    return A


# ---------------------------------------------------------------- mutation
def mutation_kernel(raw, cls, n):
    """K[i, j] = prob. a random single mutation of a class-j machine lands in
    class i. Mutations as Wolfram describes: flip a state's colour, or
    re-point one edge to a different state."""
    K = np.zeros((n, n)); counts = np.zeros(n)
    s = len(raw[0])
    for idx, m in enumerate(raw):
        j = cls[idx]; out = []
        for q in range(s):
            c, a, b = m[q]
            out.append(m[:q] + ((1 - c, a, b),) + m[q + 1:])
            for t in range(s):
                if t != a: out.append(m[:q] + ((c, t, b),) + m[q + 1:])
                if t != b: out.append(m[:q] + ((c, a, t),) + m[q + 1:])
        for mm in out:
            K[cls[raw_index[mm]], j] += 1.0 / len(out)
        counts[j] += 1
    return K / counts[None, :]


# ---------------------------------------------------------------- traits
def traits(m, first=None):
    """Axelrod-style behavioural tests, using fixed opponent probe sequences
    (a Moore machine's moves depend only on the opponent's history)."""
    if first is not None:                                  # forced opening move
        _resp = response
        response_ = lambda mm, w: [first] + _resp(mm, w)[1:]
    else:
        response_ = response
    allc = response_(m, [0] * 12)
    nice = all(x == 0 for x in allc)                       # never defects first
    sustained = response_(m, [0, 0, 0] + [1] * 6)
    provocable = any(x == 1 for x in sustained[4:])        # hits back eventually
    one_off = response_(m, [0, 0, 0, 1] + [0] * 10)
    tolerant = provocable and all(x == 0 for x in one_off) # shrugs off a single D
    back = response_(m, [0, 0, 0] + [1] * 3 + [0] * 12)
    forgiving = provocable and all(x == 0 for x in back[-6:])
    return dict(nice=nice, provocable=provocable, forgiving=forgiving,
                tolerant=tolerant, exploits_AllC=not nice, states=len(m))


# ---------------------------------------------------------------- dynamics
def evolve(A, K, gens, beta, mu, p0):
    """Discrete replicator with exponential fitness:
         p_i <- p_i exp(beta * F_i) / Z,   F = A p
       then mutation: p <- (1-mu) p + mu K p.
       (Exponential form is invariant to shifting payoffs, so negative
       prison-year payoffs are fine.)"""
    p = p0.copy(); hist = np.empty((gens + 1, len(p))); hist[0] = p
    meanF = np.empty(gens + 1)
    for t in range(gens):
        F = A @ p
        meanF[t] = p @ F
        w = p * np.exp(beta * (F - F.max()))
        p = w / w.sum()
        if mu > 0:
            p = (1 - mu) * p + mu * (K @ p)
        hist[t + 1] = p
    meanF[gens] = p @ (A @ p)
    return hist, meanF


def detect_cycle(hist, tail=20000):
    """Classify the late-time behaviour of p(t)."""
    H = hist[-min(tail, len(hist) // 2):]
    i = H.var(0).argmax()
    swing = np.ptp(H[:, i])
    if swing < 1e-4:
        return "settled (fixed point or neutral resting set)"
    y = H[:, i] - H[:, i].mean()
    up = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if len(up) >= 3:
        d = np.diff(up)
        return (f"oscillating: period ~{d.mean():.0f} gens (sd {d.std():.1f}), "
                f"largest share swing {swing:.2f}")
    return f"still drifting (swing {swing:.2f}) - run longer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--game", default="prison", choices=PAYOFFS)
    ap.add_argument("--noise", type=float, default=0.0, help="execution error per move")
    ap.add_argument("--mu", type=float, default=0.0, help="mutation rate per generation")
    ap.add_argument("--beta", type=float, default=1.0, help="selection strength")
    ap.add_argument("--gens", type=int, default=5000)
    ap.add_argument("--ensemble", default="wolfram", choices=["wolfram", "full"],
                    help="wolfram = 22/956 strategies (his counts); full = 26/1054")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.ensemble}_s{a.states}_{a.game}_eps{a.noise}_mu{a.mu}_b{a.beta}"

    machines, raw, cls, first = build(a.states, a.ensemble)
    global raw_index
    raw_index = {m: i for i, m in enumerate(raw)}
    n = len(machines)
    names = [name_of(m, a.ensemble) for m in machines]
    P = PAYOFFS[a.game]
    print(f"{n} distinct strategies ({a.ensemble} ensemble, <= {a.states} states)")

    cache = os.path.join(a.out, f"A_{tag.split('_mu')[0]}.npy")
    if os.path.exists(cache):
        A = np.load(cache)
    else:
        A = payoff_matrix_exact(machines, P, first) if a.noise == 0 else \
            payoff_matrix_noisy(machines, P, a.noise, first=first)
        np.save(cache, A)

    # ---- Step 1: Wolfram's static ranking (generation 0)
    F0 = A.mean(1)
    order = np.argsort(-F0)
    print("\nSTATIC RANKING vs uniform field (Wolfram's experiment):")
    for r, i in enumerate(order[:10], 1):
        print(f"  {r:2d}. {names[i]:<22s} {F0[i]: .4f}")
    if "TitForTat" in names:
        t = names.index("TitForTat")
        print(f"  ... TitForTat ranks {list(order).index(t) + 1} of {n} ({F0[t]:.4f})")

    # ---- Step 2: Strogatz's experiment
    K = mutation_kernel(raw, cls, n) if a.mu > 0 else None
    hist, meanF = evolve(A, K, a.gens, a.beta, a.mu, np.full(n, 1 / n))
    pf = hist[-1]
    print(f"\nAFTER {a.gens} GENERATIONS of selection "
          f"(noise={a.noise}, mu={a.mu}, beta={a.beta}):")
    for i in np.argsort(-pf)[:8]:
        if pf[i] < 1e-4: break
        print(f"  {names[i]:<22s} share {pf[i]:.4f}   (static rank "
              f"{list(order).index(i) + 1})")
    print("  dynamics:", detect_cycle(hist))

    # ---- Axelrod's question under dynamics
    T = [traits(m, first) for m in machines]
    keys = ["nice", "provocable", "forgiving", "tolerant", "exploits_AllC"]
    tw = {k: hist @ np.array([x[k] for x in T], float) for k in keys}
    tw["mean_states"] = hist @ np.array([x["states"] for x in T], float)
    print("\nPopulation-weighted traits, start -> end:")
    for k, v in tw.items():
        print(f"  {k:<14s} {v[0]:.3f} -> {v[-1]:.3f}")

    # ---- plots
    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    top = np.argsort(-hist.max(0))[:8]
    for i in top:
        ax[0].plot(hist[:, i], label=names[i])
    ax[0].set_ylabel("population share"); ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_xscale("symlog", linthresh=10)
    ax[1].plot(meanF); ax[1].set_ylabel("mean payoff")
    for k in keys:
        ax[2].plot(tw[k], label=k)
    ax[2].set_ylabel("trait frequency"); ax[2].legend(fontsize=7)
    ax[2].set_xlabel("generation")
    fig.suptitle(tag); fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"evo_{tag}.png"), dpi=130)

    with open(os.path.join(a.out, f"summary_{tag}.json"), "w") as f:
        json.dump(dict(static_top=[names[i] for i in order[:10]],
                       final={names[i]: float(pf[i]) for i in np.argsort(-pf)[:10]},
                       traits_end={k: float(v[-1]) for k, v in tw.items()},
                       dynamics=detect_cycle(hist)), f, indent=2)
    np.save(os.path.join(a.out, f"hist_{tag}.npy"), hist)


if __name__ == "__main__":
    main()
