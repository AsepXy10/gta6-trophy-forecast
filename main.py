"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles of its earn
rate on Exophase at the case target, from the house's recorded reads before
the checkpoint. Python 3.13 standard library only, no network, deterministic.

Model
-----
A trophy's earn rate is earners / players_tracked, and the page only changes
when Exophase refreshes it, hours apart, so the history is flat between
refreshes and jumps at them. The program collapses consecutive identical
reads into distinct states and works on those.

For each trophy it fits a straight line of logit(rate) against log time
since release, ln(1 + hours), over the distinct states -- the same shape
the house's curve baseline uses, but with three changes that matter over a
four-and-a-half-day horizon:

1. It uses only the last 24 hours of states, recency-weighted
   (half-life a day). The launch-night slope -- a surge of early progress
   and dilution -- does not describe the drift two days in, and fitting
   all of the history over-forecasts growth; the last day is the regime
   that matters.
2. Each trophy's slope is damped (scaled down): the pre-deadline drift
   is the launch surge, and progress after the deadline may stall as the
   binge ends, so the forecast hedges toward persistence instead of
   extrapolating the surge at full speed.
3. The extrapolated rise is capped: a rate cannot climb more than a few
   logit units above its last read, and never past a generous tier ceiling,
   so a steep fitted slope cannot run away over the long horizon.

The median is anchored on the trophy's last read rate (not on the fitted
line's level), which keeps a fresh refresh from being averaged away.

The spread is honest about three things: how well the trend fit
reproduces the rates seen so far, how violently rates jumped at the
refreshes, and a random-walk term that widens with the horizon. Quantiles
are the median shifted on the logit scale by the normal z-scores.

Flat history (no refresh seen at all)
------------------------------------
When every read is identical the trend above is meaningless: the page is
stale and a refresh is overdue. The rules describe the one refresh the
house saw on launch night -- tracked 638 -> 3,868, every rate but the 100%
one falling -- and the program uses that shape as a prior: it assumes the
next refresh multiplies the tracked count by J, where J grows with how
stale the page is (time from the first read to the target, at the launch
night doubling pace of about 1.8 hours), and that newly tracked players
hold each trophy at about 7% of the rate the old cohort does (the share
the launch-night totals imply). The median then falls to
D = (1 + c*(J-1)) / J of the last rate, with a wide spread, because a
refresh's per-trophy dilution is noisy. A trophy at 100% stays 100%: every
newly tracked player held it at the first refresh.

Edge cases are handled without crashing: missing rates fall back to the
trophy's last seen rate, a trophy with no rate at all to the cross-trophy
median, and a history with no reads at all to a flat 1% prior with a wide
spread, so the program always writes a valid forecast.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuning constants (chosen on synthetic final-regime cases; see README)
TREND_HALF_LIFE_H = 24.0   # recency weight half-life for the trend fit
TREND_WINDOW_H = 24.0      # only states this fresh count (None: all states)
ANCHOR_LAST = True         # anchor the median on the last read rate
SLOPE_SHRINK = 0.0         # pull each slope this far toward the median slope
SLOPE_DAMP = 0.75          # hedge: scale every slope (progress may stall)
RISE_CAP = 4.0             # max logit rise of the median above the last rate
TIER_CEILING = {"platinum": 25.0, "gold": 85.0, "silver": 95.0, "bronze": 99.0}
SPREAD_FLOOR = 0.30
SPREAD_CAP = 2.2
RANDOM_WALK_SIGMA = 0.5    # logit units per sqrt(day)
JUMP_WEIGHT = 1.0          # weight on refresh-jump volatility in the spread
SPREAD_MULT = 0.6          # overall spread multiplier

# Launch-night refresh prior (see module docstring): the tracked base
# doubles about every 1.8 h while the launch surge lasts, and newly tracked
# players hold each trophy at ~7% of the old cohort's rate.
LAUNCH_DOUBLING_HOURS = 1.8
NEW_COHORT_RATE = 0.067
MAX_JUMP = 8.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg > 20.0:
        return 99.99
    if lg < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def wols(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0.0:
        return (sum(ys) / len(ys) if ys else 0.0), 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def distinct_states(case):
    """Collapse consecutive identical reads into states.

    Returns a list of (epoch_seconds, players_tracked, rates list) with one
    entry per distinct page state, oldest first.
    """
    states = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev:
            continue
        prev = key
        states.append((parse_time(read["at"]), read["players_tracked"],
                       list(read["rates"])))
    return states


def last_rates(case):
    """Last non-null rate per trophy (trophies are in `trophies` order)."""
    n = len(case["trophies"])
    out = [None] * n
    for read in reversed(case["history"]):
        for i, r in enumerate(read["rates"]):
            if out[i] is None and r is not None:
                out[i] = r
        if all(v is not None for v in out):
            break
    seen = [v for v in out if v is not None]
    fallback = sorted(seen)[len(seen) // 2] if seen else 1.0
    return [fallback if v is None else v for v in out]


def quantiles(median_pct, spread):
    lg = logit(median_pct)
    qs = [round4(clamp_pct(inv_logit_pct(lg + z * spread))) for z in Z]
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def flat_history_forecast(case, last, horizon_days):
    """No refresh seen: expect the overdue refresh to dilute the rates."""
    target = parse_time(case["target"])
    first = parse_time(case["history"][0]["at"])
    stale_hours = max(0.0, (target - first) / 3600.0)
    jump = 2.0 ** (stale_hours / LAUNCH_DOUBLING_HOURS)
    jump = min(MAX_JUMP, max(1.5, jump))
    dilution = (1.0 + NEW_COHORT_RATE * (jump - 1.0)) / jump
    # A longer horizon means more refreshes after the first; let the
    # dilution deepen slowly toward the new cohort's own rate, floored.
    extra = 0.5 * math.log1p(horizon_days)
    dilution = max(0.08, dilution * math.exp(-0.25 * extra))
    spread = 0.5 + 0.25 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last[i]
        if r >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.0, 99.8, 100, 100, 100]})
        else:
            rows.append({"id": trophy["id"],
                         "q": quantiles(max(0.01, r * dilution), spread)})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    last = last_rates(case)
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")

    if not case["history"]:
        return {"trophies": [{"id": t["id"], "q": quantiles(1.0, 1.2)}
                             for t in trophies]}

    states = distinct_states(case)
    if len(states) < 2:
        return flat_history_forecast(case, last, horizon_days)

    t_last = states[-1][0]
    h_target = (target - release) / 3600.0
    x_target = math.log1p(max(0.0, h_target))
    x_last = math.log1p(max(0.0, (t_last - release) / 3600.0))

    # Per-trophy trend fits: logit(rate) on ln(1 + hours since release),
    # recency-weighted.
    slopes = []
    fits = []
    for i in range(n_tr):
        xs, ys, ws = [], [], []
        for (t, _n, rates) in states:
            r = rates[i]
            if r is None:
                continue
            age_h = max(0.0, (t_last - t) / 3600.0)
            if TREND_WINDOW_H is not None and age_h > TREND_WINDOW_H:
                continue
            h = (t - release) / 3600.0
            w = 0.5 ** (age_h / TREND_HALF_LIFE_H)
            xs.append(math.log1p(max(0.0, h)))
            ys.append(logit(r))
            ws.append(w)
        if len(xs) < 2:  # window left too little: fall back to all states
            xs, ys, ws = [], [], []
            for (t, _n, rates) in states:
                r = rates[i]
                if r is None:
                    continue
                h = (t - release) / 3600.0
                age_h = max(0.0, (t_last - t) / 3600.0)
                w = 0.5 ** (age_h / TREND_HALF_LIFE_H)
                xs.append(math.log1p(max(0.0, h)))
                ys.append(logit(r))
                ws.append(w)
        if len(xs) >= 2:
            a_i, b_i = wols(xs, ys, ws)
            sw = sum(ws)
            x_bar = sum(w * x for w, x in zip(ws, xs)) / sw
            y_bar = sum(w * y for w, y in zip(ws, ys)) / sw
        else:
            a_i, b_i = (ys[0] if ys else logit(last[i])), 0.0
            x_bar, y_bar = (xs[0] if xs else x_last), (ys[0] if ys else logit(last[i]))
        slopes.append(b_i)
        fits.append((a_i, b_i, xs, ys, ws, x_bar, y_bar))

    med_slope = sorted(slopes)[len(slopes) // 2]

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last[i]
        if r_last >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.0, 99.8, 100, 100, 100]})
            continue
        a_i, b_i, xs, ys, ws, x_bar, y_bar = fits[i]
        b_i = (1.0 - SLOPE_SHRINK) * b_i + SLOPE_SHRINK * med_slope
        b_i *= SLOPE_DAMP
        if ANCHOR_LAST:
            rise = b_i * (x_target - x_last)
            rise = min(RISE_CAP, rise)
            med_logit = logit(r_last) + rise
        else:
            med_logit = y_bar + b_i * (x_target - x_bar)
            med_logit = min(med_logit, logit(r_last) + RISE_CAP)
        ceiling = logit(TIER_CEILING.get(trophy["tier"], 99.0))
        med_logit = min(med_logit, ceiling)
        median = clamp_pct(inv_logit_pct(med_logit))

        # Spread: trend residuals, refresh-jump volatility, random walk.
        if len(xs) >= 3:
            resid = [y - (a_i + b_i * x) for x, y in zip(xs, ys)]
            m_r = sum(resid) / len(resid)
            var_r = sum((v - m_r) ** 2 for v in resid) / (len(resid) - 1)
            sig_fit = math.sqrt(var_r)
        else:
            sig_fit = 0.3
        jumps = []
        for s_idx in range(1, len(states)):
            r0 = states[s_idx - 1][2][i]
            r1 = states[s_idx][2][i]
            if (r0 is not None and r1 is not None
                    and states[s_idx][1] != states[s_idx - 1][1]):
                jumps.append(abs(logit(r1) - logit(r0)))
        sig_jump = (sum(jumps) / len(jumps)) if jumps else 0.0
        spread = math.sqrt(sig_fit ** 2 + (JUMP_WEIGHT * sig_jump) ** 2
                           + (RANDOM_WALK_SIGMA ** 2) * horizon_days)
        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, SPREAD_MULT * spread))
        rows.append({"id": trophy["id"], "q": quantiles(median, spread)})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
