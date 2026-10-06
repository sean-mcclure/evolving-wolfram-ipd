"""
Two ways to define the starting population of 'distinct' strategies.

full     : machines are identical iff they respond identically to every
           opponent history INCLUDING the empty one (so the first move counts).
           Gives 26 (s=2) and 1054 (s=3) strategies.
wolfram  : machines are identical iff they respond identically to every
           NON-EMPTY opponent history. This reproduces Wolfram's published
           counts exactly: 22 (s=2) and 956 (s=3). It implies the start
           state's colour is not what decides the first move, so here every
           machine opens with C (the opening Wolfram describes for Grim and
           TFT). Exact first-move convention still to be confirmed against
           his notebook.
"""
import itertools
import numpy as np
from fsm_core import enumerate_raw, canonical, response


def nonempty_key(m, L=None):
    s = len(m)
    L = L or 2 * s           # words up to 2s distinguish any two s-state machines
    return tuple(response(m, list(w))[-1]
                 for l in range(1, L + 1) for w in itertools.product((0, 1), repeat=l))


def build(s, mode):
    raw = enumerate_raw(s)
    keyf = canonical if mode == "full" else nonempty_key
    key_to_cls, cls, reps = {}, [], []
    for m in raw:
        k = keyf(m)
        if k not in key_to_cls:
            key_to_cls[k] = len(reps); reps.append(m)
        cls.append(key_to_cls[k])
    machines = [canonical(m) if mode == "full" else m for m in reps]
    first = None if mode == "full" else 0
    return machines, raw, np.array(cls), first
