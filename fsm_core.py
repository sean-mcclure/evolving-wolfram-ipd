"""
Core machinery: enumerate s-state finite-state-machine strategies the way
Wolfram describes them, reduce to behaviourally distinct ones, and play them.

A machine has s states. Each state has a colour (the move it outputs:
0 = Cooperate, 1 = Defect) and two outgoing edges: where to go if the
opponent's last move was C, and where to go if it was D.
Play starts in state 0; the first move is the colour of state 0.
There are (2 s^2)^s raw machines, matching Wolfram's count.
"""
import itertools
import numpy as np

C, D = 0, 1


def enumerate_raw(s):
    """All raw machines as tuples of (colour, next_on_C, next_on_D) per state."""
    per_state = [(c, a, b) for c in (0, 1) for a in range(s) for b in range(s)]
    return list(itertools.product(per_state, repeat=s))


def canonical(m):
    """Canonical form of the minimal equivalent Moore machine.
    Two machines get the same key iff they respond identically to every
    possible opponent history."""
    # keep reachable states only
    reach, stack = {0}, [0]
    while stack:
        q = stack.pop()
        for nxt in (m[q][1], m[q][2]):
            if nxt not in reach:
                reach.add(nxt); stack.append(nxt)
    states = sorted(reach)
    # Moore partition refinement
    block = {q: m[q][0] for q in states}
    while True:
        sig = {q: (block[q], block[m[q][1]], block[m[q][2]]) for q in states}
        ids = {v: i for i, v in enumerate(sorted(set(sig.values())))}
        new = {q: ids[sig[q]] for q in states}
        if len(set(new.values())) == len(set(block.values())):
            block = new
            break
        block = new
    # relabel blocks in BFS order from the start state
    order, seen, queue = [], {block[0]: 0}, [0]
    rep = {}
    for q in states:
        rep.setdefault(block[q], q)
    while queue:
        q = queue.pop(0)
        b = block[q]
        order.append(b)
        for nxt in (m[q][1], m[q][2]):
            nb = block[nxt]
            if nb not in seen:
                seen[nb] = len(seen); queue.append(rep[nb])
    out = []
    for b in order:
        q = rep[b]
        out.append((m[q][0], seen[block[m[q][1]]], seen[block[m[q][2]]]))
    return tuple(out)


def distinct_machines(s):
    """Return (classes, raw, class_of_raw). Each class is represented by the
    lowest-index raw machine in it (Wolfram numbers machines; his exact
    numbering scheme may differ, so we also keep canonical forms)."""
    raw = enumerate_raw(s)
    key_to_class, class_of_raw, reps = {}, [], []
    for idx, m in enumerate(raw):
        k = canonical(m)
        if k not in key_to_class:
            key_to_class[k] = len(reps)
            reps.append(idx)
        class_of_raw.append(key_to_class[k])
    return reps, raw, np.array(class_of_raw)


def play_exact(m1, m2, payoff, first=None):
    """Deterministic play. Joint state must cycle; return the exact limiting
    mean payoff to m1 (average over the cycle). If `first` is given, both
    machines open with that move instead of their start state's colour."""
    q1 = q2 = 0
    if first is not None:
        q1, q2 = m1[0][1 + first], m2[0][1 + first]
    seen, hist = {}, []
    t = 0
    while (q1, q2) not in seen:
        seen[(q1, q2)] = t
        a1, a2 = m1[q1][0], m2[q2][0]
        hist.append(payoff[a1][a2])
        q1, q2 = m1[q1][1 + a2], m2[q2][1 + a1]
        t += 1
    start = seen[(q1, q2)]
    cyc = hist[start:]
    return sum(cyc) / len(cyc)


def play_steps(m1, m2, payoff, steps):
    """Mean payoff to m1 over the first `steps` rounds."""
    q1 = q2 = 0
    tot = 0.0
    for _ in range(steps):
        a1, a2 = m1[q1][0], m2[q2][0]
        tot += payoff[a1][a2]
        q1, q2 = m1[q1][1 + a2], m2[q2][1 + a1]
    return tot / steps


def response(m, inputs):
    """Moves a machine makes when fed a fixed opponent sequence."""
    q, out = 0, []
    for x in inputs:
        out.append(m[q][0])
        q = m[q][1 + x]
    out.append(m[q][0])
    return out


if __name__ == "__main__":
    for s in (1, 2, 3):
        reps, raw, _ = distinct_machines(s)
        print(f"s={s}: raw={len(raw)}  distinct={len(reps)}")
