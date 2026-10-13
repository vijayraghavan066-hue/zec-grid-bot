"""One-time seed for each of ZEC's 5 ladders' coin_reserve: carves out
COIN_RESERVE_PCT of that ladder's OWN current capital (idle + buy_open,
no new money), places ONE real immediate-fill buy per ladder, and records
the result into state.coin_reserve_qty / state.coin_reserve_cost_basis.
From that point on, pct_ladder.py's maybe_fund_sell_from_reserve lists
this reserve as a real sell order the moment that ladder has zero live
sells of its own, and replenish_coin_reserve keeps it self-sustaining
from its own sale proceeds afterward -- this script only ever runs once
per ladder (or again by hand later if you want to top it back up).

Plan-only by default (prints what it would do, changes nothing). Pass
--execute to actually cancel orders, place a real buy, and write state.

SAFETY RULE, same as every other state-mutating script in this project:
every currently sell_open (already-bought, real) position is left
completely untouched. Only a ladder's own idle/buy_open capital is ever
touched, and only up to COIN_RESERVE_PCT of it.
"""
import argparse
import shutil
import time

from binance_client import BinanceUSTrader
from config import load_config
from pct_ladder import OFFSET_TOLERANCE_PCT, _band_key, ladder_specs
from state import load_state, save_state

MIN_NOTIONAL_SAFETY_MARGIN = 2.0  # same convention as _thin_to_affordable elsewhere in this project


def _ladder_offset(cfg, name):
    return next(o for _, o, _, n in ladder_specs(cfg) if n == name)


def _ladder_candidates(state, offset_pct):
    """idle/buy_open levels belonging to this ladder (gap matches its own
    offset_pct exactly), smallest dollar value first -- so freeing capital
    disturbs the fewest, smallest resting orders first."""
    members = [lvl for lvl in state.levels
               if not lvl.one_shot and lvl.state in ("idle", "buy_open")
               and abs((lvl.sell_price / lvl.buy_price - 1.0) - offset_pct) < OFFSET_TOLERANCE_PCT]
    members.sort(key=lambda lvl: lvl.target_qty() * lvl.buy_price)
    return members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually cancel orders, place a real buy, and write new state. Omit for a plan-only dry run.")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.pct_ladder_enabled:
        raise SystemExit("PCT_LADDER_ENABLED is not true in .env -- this script only makes "
                          "sense once the ladder design is live.")
    if cfg.coin_reserve_pct <= 0:
        raise SystemExit("COIN_RESERVE_PCT is 0 in .env -- set it to seed a reserve.")

    trader = BinanceUSTrader(cfg.api_key, cfg.api_secret, cfg.symbol, cfg.dry_run)
    filters = trader.get_symbol_filters()
    current_price = trader.get_price()
    buy_price = filters.round_price(current_price * 1.002)  # crosses the spread on purpose, same technique DASH's bulk ladder uses

    state = load_state(cfg.state_file)
    if state is None:
        raise SystemExit(f"No {cfg.state_file} found.")

    print(f"Current price: {current_price}, seed buy price (crosses spread): {buy_price}")
    print(f"COIN_RESERVE_PCT: {cfg.coin_reserve_pct * 100:g}%")
    print()

    plan = []  # (name, to_cancel[], freed, target, qty)
    for _, offset_pct, _, name in ladder_specs(cfg):
        already = state.coin_reserve_qty.get(name, 0.0)
        if already > 0:
            print(f"{name}: already has a reserve of {already} -- skipping (re-run manually per-ladder if you want to top it up)")
            continue

        candidates = _ladder_candidates(state, offset_pct)
        own_capital = sum(lvl.target_qty() * lvl.buy_price for lvl in candidates)
        target = cfg.coin_reserve_pct * own_capital

        if target * MIN_NOTIONAL_SAFETY_MARGIN < filters.min_notional:
            print(f"{name}: own capital ${own_capital:.4f}, target seed ${target:.4f} -- "
                  f"too thin to clear min_notional (${filters.min_notional}) safely, skipping")
            continue

        to_cancel = []
        freed = 0.0
        for lvl in candidates:
            if freed >= target:
                break
            to_cancel.append(lvl)
            freed += lvl.target_qty() * lvl.buy_price

        qty = filters.round_qty(min(freed, target) / buy_price)
        if qty <= 0 or qty * buy_price < filters.min_notional:
            print(f"{name}: freed ${freed:.4f} of ${target:.4f} target but resulting qty too thin, skipping")
            continue

        plan.append((name, to_cancel, freed, target, qty))
        print(f"{name}: own capital ${own_capital:.4f} -> target seed ${target:.4f} "
              f"(cancelling {len(to_cancel)} level(s), freeing ${freed:.4f}) -> "
              f"buy {qty} ZEC @ ${buy_price:.6f} (${qty * buy_price:.4f}), "
              f"${freed - qty * buy_price:.4f} overshoot returns to pending-reinvest")

    if not plan:
        print("\nNothing to seed.")
        return

    if not args.execute:
        print("\nPLAN ONLY -- nothing changed. Re-run with --execute to apply.")
        return

    print("\nEXECUTING...")
    backup_path = f"{cfg.state_file}.pre_coin_reserve_seed_{int(time.time())}.bak"
    shutil.copy(cfg.state_file, backup_path)
    print(f"Backed up current state to {backup_path}")

    for name, to_cancel, freed, target, qty in plan:
        cancelled = 0
        for lvl in to_cancel:
            if lvl.state == "buy_open" and lvl.buy_order_id is not None:
                try:
                    trader.cancel_order(lvl.buy_order_id)
                    cancelled += 1
                except Exception as exc:
                    print(f"  {name}: could not cancel level {lvl.index} (order {lvl.buy_order_id}): {exc} "
                          f"-- likely already filled for real, PRESERVING it untouched")
                    lvl.state = "sell_open"  # if it just filled, it's a real position now, not idle/discarded
                    continue
            state.levels.remove(lvl)

        fresh_price = trader.get_price()
        fresh_buy_price = filters.round_price(fresh_price * 1.002)
        fresh_qty = filters.round_qty(min(freed, target) / fresh_buy_price)
        if fresh_qty <= 0 or fresh_qty * fresh_buy_price < filters.min_notional:
            print(f"  {name}: price moved, freed capital no longer sufficient -- "
                  f"${freed:.4f} returned to pending-reinvest instead of buying")
            band_key = _band_key(_ladder_offset(cfg, name))
            state.pending_reinvest_by_band[band_key] = state.pending_reinvest_by_band.get(band_key, 0.0) + freed
            continue

        try:
            order_id = trader.place_limit_buy(fresh_buy_price, fresh_qty)
        except Exception as exc:
            print(f"  {name}: seed buy FAILED ({exc}) -- ${freed:.4f} returned to pending-reinvest")
            band_key = _band_key(_ladder_offset(cfg, name))
            state.pending_reinvest_by_band[band_key] = state.pending_reinvest_by_band.get(band_key, 0.0) + freed
            continue

        filled = trader.is_filled(order_id, trader.get_price())
        if not filled:
            print(f"  {name}: WARNING -- seed buy placed (order {order_id}) but not confirmed filled yet "
                  f"(crossing-the-spread order should fill almost immediately; check manually). "
                  f"Recording the reserve anyway since the order is real and resting/filling.")

        state.coin_reserve_qty[name] = fresh_qty
        state.coin_reserve_cost_basis[name] = fresh_buy_price
        overshoot = freed - fresh_qty * fresh_buy_price
        if overshoot > 1e-9:
            band_key = _band_key(_ladder_offset(cfg, name))
            state.pending_reinvest_by_band[band_key] = state.pending_reinvest_by_band.get(band_key, 0.0) + overshoot
        print(f"  {name}: cancelled {cancelled}/{len(to_cancel)} level(s), bought {fresh_qty} ZEC @ "
              f"${fresh_buy_price:.6f} (order {order_id}), ${overshoot:.4f} overshoot to pending-reinvest")

    save_state(cfg.state_file, state)
    print("\nWrote new state. Restart the bot service now to pick up COIN_RESERVE_PCT and the "
          "ongoing maybe_fund_sell_from_reserve/replenish_coin_reserve mechanism (config_signature "
          "unaffected -- it will resume, not rebuild).")


if __name__ == "__main__":
    main()
