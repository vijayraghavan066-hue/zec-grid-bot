"""Shrinks (never cancels outright) buy_open rungs that sit at or beyond a
given depth below current price -- the farthest, DCA-heaviest rungs, same
ones identified as "least likely to fill soon" -- freeing USD back to the
account's real free balance without losing any rung's place in the ladder.

Each eligible rung's target_qty() is cut by --trim-fraction (default 50%).
A rung is skipped (left at full size) if trimming it would drop its
notional below the exchange's min_notional -- same floor every other
script in this project already respects, never built to place an order
the exchange would reject.

Cancels the old resting buy order and places a new smaller one at the
SAME price (same rung, same place in the ladder, just smaller) -- never
touches sell_open/held coin or any level outside the depth threshold.

Plan-only by default; pass --execute to actually cancel+replace orders.

    python trim_deep_buy_rungs.py 0.30                # dry run, 30%+ depth
    python trim_deep_buy_rungs.py 0.30 --execute
    python trim_deep_buy_rungs.py 0.30 --trim-fraction 0.7 --execute
"""
import argparse
import shutil
import time

from binance_client import BinanceUSTrader
from config import load_config
from state import load_state, save_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("depth_pct", type=float,
                         help="Trim buy_open rungs at or beyond this fraction below current "
                              "price (0.30 = 30%% or more below).")
    parser.add_argument("--trim-fraction", type=float, default=0.5,
                         help="Fraction of target_qty to cut (default 0.5 = halve).")
    parser.add_argument("--execute", action="store_true",
                         help="Actually cancel+replace orders. Omit for a plan-only dry run.")
    args = parser.parse_args()

    cfg = load_config()
    trader = BinanceUSTrader(cfg.api_key, cfg.api_secret, cfg.symbol, cfg.dry_run)
    filters = trader.get_symbol_filters()
    current_price = trader.get_price()

    state = load_state(cfg.state_file)

    candidates = []
    for lvl in state.levels:
        if lvl.state != "buy_open":
            continue
        depth = (current_price - lvl.buy_price) / current_price
        if depth < args.depth_pct:
            continue
        old_qty = lvl.target_qty()
        new_qty = filters.round_qty(old_qty * (1.0 - args.trim_fraction))
        new_notional = new_qty * lvl.buy_price
        if new_qty <= 0 or new_notional < filters.min_notional:
            continue  # trimming would clear below min_notional -- left untouched
        freed = (old_qty - new_qty) * lvl.buy_price
        if freed <= 1e-9:
            continue
        candidates.append((lvl, old_qty, new_qty, freed))

    total_freed = sum(c[3] for c in candidates)
    print(f"{cfg.symbol}: current price {current_price}, depth threshold {args.depth_pct * 100:.0f}%, "
          f"trim fraction {args.trim_fraction * 100:.0f}%")
    print(f"{len(candidates)} rung(s) eligible (others at this depth skipped -- would clear below "
          f"min_notional {filters.min_notional} after trim)")
    for lvl, old_qty, new_qty, freed in candidates:
        print(f"  buy {lvl.buy_price:.6f}: {old_qty:.6f} -> {new_qty:.6f}  (frees ${freed:.4f})")
    print(f"TOTAL freed: ${total_freed:.4f}")

    if not args.execute:
        print()
        print("PLAN ONLY -- nothing changed. Re-run with --execute to apply.")
        return

    print()
    print("EXECUTING...")
    backup_path = f"{cfg.state_file}.pre_trim_deep_buy_{int(time.time())}.bak"
    shutil.copy(cfg.state_file, backup_path)
    print(f"Backed up state to {backup_path}")

    done = 0
    actual_freed = 0.0
    for lvl, old_qty, new_qty, freed in candidates:
        try:
            if lvl.buy_order_id:
                trader.cancel_order(lvl.buy_order_id)
        except Exception as e:
            print(f"  SKIPPED buy {lvl.buy_price:.6f}: cancel failed ({e}) -- left untouched")
            continue
        try:
            new_order_id = trader.place_limit_buy(lvl.buy_price, new_qty)
        except Exception as e:
            # Cancelled but failed to replace -- mark idle so the bot's own
            # next cycle re-arms it at the new (smaller) size, same safe
            # idle-retry pattern used everywhere else in this project.
            lvl.state = "idle"
            lvl.buy_order_id = None
            lvl.base_qty = new_qty
            lvl.extra_qty = 0.0
            print(f"  PARTIAL buy {lvl.buy_price:.6f}: cancelled old order but replace failed ({e}) "
                  f"-- marked idle, bot will re-arm at new size next cycle")
            continue
        lvl.base_qty = new_qty
        lvl.extra_qty = 0.0
        lvl.buy_order_id = new_order_id
        done += 1
        actual_freed += freed

    save_state(cfg.state_file, state)
    print(f"Done: {done}/{len(candidates)} rung(s) trimmed cleanly, ~${actual_freed:.4f} freed to real free balance.")


if __name__ == "__main__":
    main()
