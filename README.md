# evolving-wolfram-ipd

What happens to Stephen Wolfram's Prisoner's Dilemma strategies when the population is allowed to evolve?

## Background

In [Games between Programs](https://writings.stephenwolfram.com/2026/06/games-between-programs-the-ruliology-of-competition/) (June 2026), Wolfram took every strategy that a small finite-state machine can encode and had each one play every other in the iterated Prisoner's Dilemma. He then ranked them by average payoff. Grim Trigger came out on top, and Tit for Tat ranked well down the list.

Steven Strogatz pointed out that this is one round of a tournament against a fixed, uniform field of opponents. Nothing reproduces, so successful strategies never become more common and weak ones never disappear. He proposed the next step: let the scores change the population, recompute everyone's fitness against the new population, and repeat.

This repo runs that experiment. It keeps Wolfram's strategy space and adds population dynamics.

## Quick start

Requires Python 3.9+.

```
git clone https://github.com/sean-mcclure/evolving-wolfram-ipd
cd evolving-wolfram-ipd
python3 -m pip install -r requirements.txt
python3 evolve.py --states 2 --noise 0.01 --mu 0.001 --gens 20000
```

This takes a few seconds. It prints Wolfram's static ranking, then the population after evolution, and saves a plot to `results/`.

## What each script does

| Script | Purpose | Typical runtime |
|---|---|---|
| `evolve.py` | One evolutionary run, with plots of strategy shares, mean payoff and trait frequencies | Seconds (2 states), about a minute (3 states) |
| `sweep.py` | Runs every combination of noise, mutation and selection strength and saves a summary table | About 1 minute (2 states), up to a few hours (3 states) |
| `moran.py` | The same question in a finite population, where random drift matters. Averages over several random seeds | Minutes to tens of minutes |
| `fsm_core.py` | Enumerates the machines and plays them against each other | (library) |
| `ensembles.py` | Defines which machines count as distinct strategies | (library) |

## Reproducing the results

The 2-state sweep:

```
python3 sweep.py --states 2
```

The 3-state sweep. Run the two halves in separate terminals to use two cores:

```
python3 sweep.py --states 3 --ensemble wolfram
python3 sweep.py --states 3 --ensemble full
```

The finite-population check:

```
python3 moran.py --states 2 --noise 0.01 --seeds 3 --gens 3000
python3 moran.py --states 3 --noise 0.01 --seeds 3 --gens 1000
```

Sweep results are written to `results/sweep_s<states>_<ensemble>.csv` after every run, so you can stop partway without losing finished work. Payoff matrices are cached in `results/`, so later runs start faster.

## Options

| Flag | Meaning | Default |
|---|---|---|
| `--states` | Maximum number of states per machine (2 or 3) | 2 |
| `--ensemble` | `wolfram` (22 / 956 strategies) or `full` (26 / 1054) | `wolfram` |
| `--noise` | Chance that each intended move is flipped by mistake | 0 |
| `--mu` | Mutation rate per generation (replicator model) | 0 |
| `--u` | Mutation chance per birth (Moran model) | 0.005 |
| `--beta` | Selection strength | 1.0 |
| `--gens` | Number of generations | 5000 |
| `--game` | Payoffs: `prison` (Wolfram-style) or `axelrod` (3/0/5/1) | `prison` |

## How it works

**Strategies.** Each machine has some states. Each state outputs a move (cooperate or defect) and has two arrows: one to follow if the opponent last cooperated, one if they defected. This matches Wolfram's construction.

**Evolution.** Each generation, a strategy's fitness is its average payoff against the current population:

```
F_i(t) = sum over j of A[i, j] * p_j(t)
```

`A[i, j]` is strategy i's long-run payoff against j, and `p_j(t)` is how common j currently is. Strategies then grow or shrink in proportion to `exp(beta * F_i)`. Mutation uses Wolfram's own mutation operators: flip a state's move, or re-point one arrow.

**Errors.** With `--noise`, each intended move is flipped with that probability. Payoffs are then computed exactly as the long-run average of a Markov chain over the two machines' joint states.

**Traits.** The code tracks four Axelrod-style traits, measured by feeding each machine fixed opponent sequences:

- nice: never defects first
- provocable: eventually hits back against repeated defection
- forgiving: returns to cooperation after the opponent does
- tolerant: ignores a single defection

These are operational stand-ins for Axelrod's principles, not his exact definitions. See `traits()` in `evolve.py`.

## Matching Wolfram

**Reproduced:** the raw machine counts (64 and 5,832), the distinct strategy counts (22 and 956), Grim Trigger as the top 2-state strategy, and Tit for Tat ranking well down the list.

**Not yet matched exactly:** Grim Trigger's average payoff comes out at −0.909 here, against Wolfram's reported −0.866. The likely cause is his payoff table, which appears only as an image in the article, or a detail of how he averages. Corrections are welcome.

**Assumed:** in the `wolfram` ensemble, every machine opens with Cooperate. Wolfram's strategy count implies that the opening move isn't set by the start state, and he describes both Grim Trigger and Tit for Tat as opening with Cooperate. The `full` ensemble drops this assumption and lets each machine's start state set its opening move. Results are reported for both.

## Results

*[Fill in once the 3-state sweep is complete: headline findings, key figures, and the robustness table.]*

A short write-up is available at *[link]*.

## Citation

If you use this code, please cite:

> McClure, S. (2026). *evolving-wolfram-ipd* [Software]. https://github.com/sean-mcclure/evolving-wolfram-ipd

and Wolfram's original article.

## License

MIT
