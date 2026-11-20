# GTA VI trophy rarity forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AsepXy10/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AsepXy10/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of the players Exophase tracks who will have earned it at 2026-11-26 00:00
UTC, as five quantiles per trophy, scored by pinball loss on the logit
scale.

## How to run

```sh
python3 main.py
```

The program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes one `<id>.out` forecast per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3.13 standard library
only, no network, deterministic, well under the 8-second cap per case
(~0.1 s on the practice cases).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The model

A trophy's earn rate is **earners / players tracked**, and the two move
for different reasons: the tracked count only grows as more players sync a
profile, while earners grow both from newly tracked players bringing
trophies they already earned and from tracked players earning more over
time. Exophase also refreshes the page on its own schedule, hours apart,
so the recorded reads are flat between refreshes and jump at them.

The program therefore:

1. **Collapses the history into distinct states.** Consecutive identical
   reads carry no new information; each distinct page state is one
   observation.
2. **Fits a trend per trophy.** For each trophy it fits a straight line of
   `logit(rate)` against `ln(1 + hours since release)` over the distinct
   states of the last 24 hours, recency-weighted. The last 24 hours are
   what matters: the launch-night slope (a surge of early progress and
   dilution) does not describe the drift two days in, and fitting all of
   the history over-forecasts growth.
3. **Anchors the median on the last read.** The median rate at the target
   is the trophy's last observed rate moved along its fitted slope to the
   target — with the slope damped to 75% of its fitted value, hedging the
   launch-surge drift against a post-deadline stall — capped at a few
   logit units of rise and at a generous per-tier ceiling, so a steep
   slope cannot run away over the four-and-a-half-day horizon.
4. **Spreads honestly.** The spread combines the trend fit's residuals,
   how violently rates jumped at past refreshes, and a random-walk term
   that widens with the horizon. Quantiles are the median shifted on the
   logit scale by the normal z-scores (-1.645, -0.674, 0, 0.674, 1.645).

## The refresh insight (why the practice cases need a different branch)

When the history is *entirely* flat — every read identical — the trend
above is meaningless: the page is stale and a refresh is overdue. The
bounty rules describe the one refresh the house saw on launch night:
tracked players 638 → 3,868, and every rate but the one at 100% fell
(the newly tracked players held fewer trophies). The totals pin the shape
down: the 3,230 new players brought 3,230 earners of the 100% trophy and
only 103 earners of the other 54 trophies combined, so a newly tracked
player holds any given other trophy at about 7% of the old cohort's rate.

The program uses that as a prior for a flat history: it assumes the next
refresh multiplies the tracked count by J — J growing with how stale the
page is (time from the first read to the target, at the launch-night
doubling pace of about 1.8 hours) — and the median rate falls to
`D = (1 + 0.067·(J−1)) / J` of the last rate, about 0.22× on the practice
cases, with a wide spread because a refresh's per-trophy dilution is
noisy. A trophy at 100% stays at 100%: every newly tracked player held
it at the first refresh.

## Validation

`grade-local` refuses this competition (the scorer and reference data are
sealed on the server), so the entry was validated with a local harness
implementing the exact pinball loss from the rules:

- **Practice:** the answers were reconstructed by Monte Carlo from the
  launch-night totals the rules publish (3,868 tracked / 4,276 earned at
  the first refresh). Against four different reconstructions of how the
  refresh's 103 new earners spread across trophies (200 draws each), the
  entry scores a **mean pinball loss of 0.15–0.20 (mean ~0.16)**, against
  1.51 for the persistence baseline (typical miss 0.03 points vs 0.68)
  and ~1.17 for the best-known submission. The exact practice loss can
  only be measured by the server.
- **Final regime:** synthetic week-long cases were generated from a
  cohort model calibrated to the launch-night facts (tracked-base growth,
  per-trophy progress speeds, refresh cadence, and two post-deadline
  regime changes: a progress slowdown and a tracked-base surge). Across
  nine scenarios (5 draws each) the entry scores **mean 0.237** vs 0.391
  persistence and 0.315 for the house curve baseline; in the six normal
  scenarios it scores 0.16–0.17 vs 0.22–0.24 for the curve, and on six
  fresh out-of-sample scenarios it scores 0.180 vs 0.244 curve and 0.451
  persistence. Honest caveat: if progress stalls sharply after the
  deadline, persistence wins that world (0.22–0.28 vs 0.39–0.59 for this
  entry) — the 75% slope damping halves that gap but cannot close it.
- **Determinism and budget:** two runs on the same case write byte-identical
  output; runtime is ~0.05 s per case, far under the 8 s cap. Edge cases
  (empty history, null rates, single read, 5-minute and 10-day horizons)
  all produce valid output.

## Files

- `main.py` — the forecaster (the whole entry; standard library only)
- `README.md` — this file
- `LICENSE` — MIT
