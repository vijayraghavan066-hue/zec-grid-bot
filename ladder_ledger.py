"""Full-range ladder ledgers — one CSV per ladder, spanning the ENTIRE
anchor_low..anchor_high price range at that ladder's own step_pct (geometric
spacing), not just the current live window. Only rungs inside the live
window (pct_ladder.compute_ladder_zones) ever become real exchange orders;
every other rung here is a "planned" bookkeeping row with no order and no
committed capital. This file never places, cancels, or resizes anything —
it only reads state and writes a report, so it can never affect live
trading even if something in it is wrong.

Percentage-spacing adaptation of dash-grid-bot's ladder_ledger.py (which
uses fixed-dollar arithmetic spacing) — ZEC's ladders are geometric
(pct_ladder.build_ladder_points), so each ladder's full-range price points
are generated the same geometric way, not by a fixed dollar step.
"""
import csv
import os

from pct_ladder import OFFSET_TOLERANCE_PCT, build_ladder_points, ladder_specs

FIELDNAMES = ["buy_price", "sell_price", "status", "real_qty", "target_usd"]


def _row_from_level(lvl):
    qty = lvl.target_qty()
    return {
        "buy_price": f"{lvl.buy_price:.6f}",
        "sell_price": f"{lvl.sell_price:.6f}",
        "status": lvl.state,
        "real_qty": f"{qty:.6f}",
        "target_usd": f"{qty * lvl.buy_price:.2f}",
    }


def _planned_row(bp, sp):
    return {
        "buy_price": f"{bp:.6f}",
        "sell_price": f"{sp:.6f}",
        "status": "planned",
        "real_qty": "",
        "target_usd": "",
    }


def _bucket_index(price, anchor_low, step_pct):
    import math
    if price <= 0 or anchor_low <= 0:
        return None
    return round(math.log(price / anchor_low) / math.log(1.0 + step_pct))


def _rows_for_ladder(state, step_pct, offset_pct, anchor_low, anchor_high):
    candidates = [lvl for lvl in state.levels
                  if not lvl.one_shot and lvl.buy_price > 0
                  and abs((lvl.sell_price / lvl.buy_price - 1.0) - offset_pct) < OFFSET_TOLERANCE_PCT]
    live_by_bucket = {}
    for lvl in candidates:
        live_by_bucket[_bucket_index(lvl.buy_price, anchor_low, step_pct)] = lvl

    rows = []
    for bp in build_ladder_points(anchor_low, anchor_high, step_pct):
        sp = round(bp * (1.0 + offset_pct), 10)
        lvl = live_by_bucket.get(_bucket_index(bp, anchor_low, step_pct))
        rows.append(_row_from_level(lvl) if lvl is not None else _planned_row(bp, sp))
    return rows


def _write_csv(path, rows):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def write_ladder_ledgers(state, cfg, out_dir=None):
    """Writes ladder1_ledger.csv .. ladder5_ledger.csv into out_dir (defaults
    to the same directory as cfg.state_file). Cheap and read-only: at most a
    few hundred rows per ladder, safe to call every poll cycle."""
    if not cfg.pct_ladder_enabled:
        return
    if out_dir is None:
        out_dir = os.path.dirname(cfg.state_file) or "."

    for step_pct, offset_pct, _window_pct, name in ladder_specs(cfg):
        rows = _rows_for_ladder(state, step_pct, offset_pct, cfg.anchor_low, cfg.anchor_high)
        _write_csv(os.path.join(out_dir, f"{name}_ledger.csv"), rows)
