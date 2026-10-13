"""Fresh build of ZEC's 5 percentage ladders on ZECUSDT, from scratch --
there is no existing ZEC position or grid anywhere in this project to
migrate from (unlike XRP/HBAR's cutovers, which preserved and reshaped an
existing live grid, or UNI's revival, which folded in coin already held).
This is the simplest of the three build patterns: one capital source, no
double-counting to worry about, no coin-reserve-from-owned-coin step on
day one (there is no owned ZEC yet to fold in -- seed_coin_reserve.py is
the way to carve a coin reserve out of each ladder's OWN capital later,
once the ladders are actually holding real ZEC from fills, via its usual
crossing-the-spread buy).

Capital source: a USDT amount you pass in explicitly (`usdt_capital`),
split 20% per ladder, DCA-weighted into buy_open rungs via the same
_dca_build_levels used by every other ladder build in this project. This
is NOT read automatically from the account's free USDT balance -- USDT on
this account is SHARED with DASH and HBAR (both also quote in USDT), so
silently consuming "whatever is free" would eat into their own buy-side
cushion (exactly the shortfall found live during UNI's revival). The
script still reads real free USDT live and refuses to run if
usdt_capital exceeds it, but the actual number to commit is always an
explicit, deliberate choice, not an assumption.

Run this ONLY after SYMBOL=ZECUSDT, PCT_LADDER_ENABLED=true, and the
LADDERn_* / ANCHOR_LOW / ANCHOR_HIGH values are already live in .env --
this script bakes a config_signature from whatever load_config() returns
right now, and that signature has to match what bot.py computes on its
own first start for it to resume instead of refuse.

Plan-only by default (prints what it would do, changes nothing). Pass
--execute to actually place orders and write state.

    python build_zec_pct_ladders.py 100          # dry run
    python build_zec_pct_ladders.py 100 --execute

SAFETY: this intentionally creates state.json from scratch -- there is no
prior ZEC state to preserve. If a state.json already exists (e.g. a
previous partial attempt), it is backed up before being overwritten.
"""
import argparse
import shutil
import time

from binance_client import BinanceUSTrader
from config import load_config
from pct_ladder import build_ladder_points, ladder_specs, _dca_build_levels, _thin_to_affordable
from state import BotState, load_state, save_state

LADDER_CAPITAL_SPLIT = {"ladder1": 0.20, "ladder2": 0.20, "ladder3": 0.20, "ladder4": 0.20, "ladder5": 0.20}


def build_current_signature(cfg):
    """Must match bot.py's run() EXACTLY, field for field -- this is what
    makes the bot resume instead of refuse on first start after this
    script writes state.json."""
    return {
        "symbol": cfg.symbol,
        "dry_run": cfg.dry_run,
        "grid_mode": cfg.grid_mode,
        "pilot_points": cfg.pilot_points,
        "anchor_low": cfg.anchor_low,
        "anchor_high": cfg.anchor_high,
        "anchor_step": cfg.anchor_step,
        "dense_low": cfg.dense_low,
        "dense_high": cfg.dense_high,
        "dense_step": cfg.dense_step,
        "dense_sell_offset": cfg.dense_sell_offset,
        "total_capital_usdt": cfg.total_capital_usdt,
        "reference_price": cfg.reference_price,
        "dca_exponent": cfg.dca_exponent,
        "bulk_amount_usdt": cfg.bulk_amount_usdt,
        "bulk_sell_high": cfg.bulk_sell_high,
        "bulk_sell_step": cfg.bulk_sell_step,
    }


def _get_free_balance(client, asset):
    bal = client.get_asset_balance(asset=asset)
    if bal is None:
        return 0.0
    return float(bal["free"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usdt_capital", type=float,
                         help="How much USDT to commit to the new buy-side ladders. Required, "
                              "and must not exceed what's actually free -- USDT on this account "
                              "is SHARED with DASH and HBAR, so this script never assumes 100%% "
                              "of free USDT belongs to ZEC.")
    parser.add_argument("--execute", action="store_true",
                         help="Actually place orders and write new state. Omit for a plan-only dry run.")
    args = parser.parse_args()

    if args.usdt_capital <= 0:
        raise SystemExit("usdt_capital must be positive.")

    cfg = load_config()
    if not cfg.pct_ladder_enabled:
        raise SystemExit("PCT_LADDER_ENABLED is not true in .env -- set it (and SYMBOL=ZECUSDT, "
                          "the LADDERn_* values) before running this build, so the signature this "
                          "script bakes in matches what bot.py will compute on first start.")
    if cfg.symbol != "ZECUSDT":
        raise SystemExit(f"SYMBOL is {cfg.symbol!r}, not ZECUSDT -- this script builds the ZECUSDT "
                          f"ladders specifically (the more liquid of ZEC's two Binance.US pairs).")

    trader = BinanceUSTrader(cfg.api_key, cfg.api_secret, cfg.symbol, cfg.dry_run)
    filters = trader.get_symbol_filters()
    current_price = trader.get_price()

    free_usdt = _get_free_balance(trader._client, "USDT")

    print(f"Current ZECUSDT price: {current_price}")
    print(f"Real free USDT balance: ${free_usdt:.4f} (SHARED with DASH and HBAR)")
    print(f"USDT capital requested for this build: ${args.usdt_capital:.4f}")
    print()

    if args.usdt_capital > free_usdt:
        raise SystemExit(f"Requested ${args.usdt_capital:.4f} but only ${free_usdt:.4f} USDT is free. "
                          f"Re-run with an amount at or below that.")
    total_capital = args.usdt_capital

    new_signature = build_current_signature(cfg)

    new_levels = []
    new_ladder_reserved = {}
    next_index = 0
    for step_pct, offset_pct, window_pct, name in ladder_specs(cfg):
        ladder_capital = total_capital * LADDER_CAPITAL_SPLIT[name]
        lo = max(current_price * (1.0 - window_pct), cfg.anchor_low)
        hi = min(current_price, cfg.anchor_high)
        points = [p for p in build_ladder_points(lo, hi, step_pct) if p < current_price]
        points = _thin_to_affordable(points, ladder_capital, cfg.reference_price, cfg.dca_exponent, filters.min_notional)
        levels = _dca_build_levels(points, ladder_capital, cfg.reference_price, cfg.dca_exponent, offset_pct, filters, next_index)
        next_index += len(levels)
        new_levels.extend(levels)

        built_capital = sum(l.target_qty() * l.buy_price for l in levels)
        leftover = ladder_capital - built_capital
        if leftover > 1e-9:
            new_ladder_reserved[name] = new_ladder_reserved.get(name, 0.0) + leftover

        status = f"{len(levels)} buy_open level(s)" if levels else "0 levels -- capital too thin for min_notional, going to its own reserve instead"
        print(f"{name}: window ${lo:.4f}-${hi:.4f}, buy capital ${ladder_capital:.4f} -> {status}"
              + (f" (${leftover:.4f} to reserve)" if leftover > 1e-9 and levels else ""))

    print()
    print(f"Total new levels: {len(new_levels)}")
    if new_ladder_reserved:
        print(f"Seeded into per-ladder reserves (not lost, available for auto top-up once big enough): {new_ladder_reserved}")

    if not args.execute:
        print()
        print("PLAN ONLY -- nothing changed. Re-run with --execute to apply.")
        return

    print()
    print("EXECUTING...")

    prior = load_state(cfg.state_file)
    if prior is not None:
        backup_path = f"{cfg.state_file}.pre_zec_pct_ladder_build_{int(time.time())}.bak"
        shutil.copy(cfg.state_file, backup_path)
        print(f"Backed up prior state to {backup_path}")

    # Levels are written IDLE, not buy_open -- this script never places a
    # real order itself. bot.py's own apply_order_window places the real
    # buy orders (using target_qty(), already correctly sized by
    # _dca_build_levels above) on its first cycle after restart. Same
    # division of responsibility as migrate_to_pct_ladder.py's cutover and
    # build_uni_pct_ladders.py's revival.
    final_state = BotState(
        levels=new_levels,
        reserved_total=0.0,
        total_profit_realized=prior.total_profit_realized if prior is not None else 0.0,
        config_signature=new_signature,
        next_level_index=next_index,
        ladder_reserved=dict(new_ladder_reserved),
        ladder_reserved_contributed=dict(new_ladder_reserved),
        ladder_reserved_drawn={},
        pending_reinvest_by_band={},
        coin_reserve_qty={},
        coin_reserve_cost_basis={},
    )
    save_state(cfg.state_file, final_state)
    print(f"Wrote new state: {len(final_state.levels)} idle level(s) (not yet real orders)")
    print("Restart the bot service now -- it will see the matching signature, RESUME "
          "(not rebuild), and place the real buy orders for these idle levels on its "
          "first cycle via apply_order_window.")
    print()
    print("Once this ladder structure has run for a while and holds real ZEC from fills, "
          "seed_coin_reserve.py can carve COIN_RESERVE_PCT out of each ladder's own capital "
          "to seed the sell-side coin reserve backstop -- not needed on day one.")


if __name__ == "__main__":
    main()
