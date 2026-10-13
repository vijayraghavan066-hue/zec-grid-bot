import logging
import logging.handlers
import os
import time
import traceback

from binance_client import BinanceUSTrader
from circuit_breaker import check_circuit_breaker
from config import load_config
from health import write_heartbeat
from ladder_ledger import write_ladder_ledgers
from ledger import log_buy_fill, log_sell_fill
from pct_ladder import (distribute_profit_ladder_aware, is_legacy_offset, ladder_specs,
                         liquidate_coin_reserve_position, liquidate_legacy_position,
                         maybe_fund_sell_from_reserve, maybe_reshape_ladders)
from state import load_state, save_state

log = logging.getLogger("zec-grid-bot")


def setup_logging(log_file: str, log_level: str):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
    )


def apply_order_window(state, current_price, trader, filters, max_open_orders,
                        taker_avoidance_buffer, halted=False):
    """Binance.US caps open orders per symbol at 200. We never touch
    sell_open levels here (we already hold that ZEC — its sell order must
    stay live no matter what). The remaining order-count budget goes to
    whichever idle/buy_open levels are closest to the current price;
    anything that drifts outside that window gets its buy order cancelled
    (no loss — it's an unfilled resting order) so the slot can go to a level
    nearer the action. As price moves, the window slides with it."""
    sell_open_count = sum(1 for lvl in state.levels if lvl.state == "sell_open")
    budget = max(0, max_open_orders - sell_open_count)

    eligible = [lvl for lvl in state.levels
                if lvl.buy_price < current_price and lvl.state in ("idle", "buy_open")]
    eligible.sort(key=lambda lvl: current_price - lvl.buy_price)
    desired_indices = {lvl.index for lvl in eligible[:budget]}

    # A fresh price snapshot right before placing — the `current_price` this
    # function was called with may already be a few seconds old by the time
    # we get here (a full cycle of fill-checks runs first). If a level's own
    # buy_price has already been reached/crossed by the time we'd place it,
    # placing there would fill immediately as a taker (0.02% fee vs 0%
    # maker). At scale that adds up, so instead we place a bit below the
    # fresh price — still a real resting maker order, capturing effectively
    # the same trade a moment later instead of paying to jump the queue.
    to_place = [lvl for lvl in eligible
                if lvl.index in desired_indices and lvl.state == "idle"]
    fresh_price = trader.get_price() if to_place else current_price

    armed = 0
    dropped = 0
    rejected = 0
    for lvl in eligible:
        in_window = lvl.index in desired_indices
        if in_window and lvl.state == "idle" and not halted:
            qty = filters.round_qty(lvl.target_qty())
            if qty <= 0 or qty * lvl.buy_price < filters.min_notional:
                continue

            place_price = lvl.buy_price
            if place_price >= fresh_price:
                adjusted = filters.round_price(fresh_price - taker_avoidance_buffer)
                if adjusted > 0 and qty * adjusted >= filters.min_notional:
                    log.info(f"Grid {lvl.index}: buy price ${lvl.buy_price} already at/above "
                              f"fresh market ${fresh_price} — placing ${adjusted} instead to "
                              f"stay a maker order (sell target ${lvl.sell_price} unchanged, "
                              f"so this level now captures a wider margin)")
                    place_price = adjusted
                # else: adjusted price would be unusable (too small / below
                # min notional) — fall through and place at the original
                # price; worst case it's a taker fill this one time.

            try:
                lvl.buy_order_id = trader.place_limit_buy(place_price, qty)
                lvl.buy_price = place_price
                lvl.qty = qty
                lvl.state = "buy_open"
                armed += 1
            except Exception as exc:
                # e.g. Binance's PERCENT_PRICE filter rejecting a price too
                # far from the current market — leave it idle, it'll be
                # retried automatically next cycle if conditions change,
                # without blocking every other level's processing.
                rejected += 1
                log.warning(f"Grid {lvl.index} (${lvl.buy_price}): buy order rejected: {exc}")
        elif not in_window and lvl.state == "buy_open":
            try:
                trader.cancel_order(lvl.buy_order_id)
            except Exception as exc:
                log.warning(f"Grid {lvl.index}: could not cancel order {lvl.buy_order_id}: {exc}")
            lvl.buy_order_id = None
            lvl.qty = 0.0
            lvl.state = "idle"
            dropped += 1

    if armed or dropped or rejected:
        log.info(f"Order window: {len(desired_indices)} level(s) targeted "
                  f"(budget={budget}, sell_open={sell_open_count}) — "
                  f"armed {armed} new, dropped {dropped} out-of-window, {rejected} rejected")


def run():
    cfg = load_config()
    setup_logging(cfg.log_file, cfg.log_level)

    log.info(f"Starting zec-grid-bot | symbol={cfg.symbol} dry_run={cfg.dry_run}")
    if not cfg.dry_run:
        log.warning("LIVE TRADING ENABLED — real orders will be placed on Binance.US")

    trader = BinanceUSTrader(cfg.api_key, cfg.api_secret, cfg.symbol, cfg.dry_run)
    filters = trader.get_symbol_filters()

    current_signature = {
        "symbol": cfg.symbol,
        "dry_run": cfg.dry_run,  # switching sim<->live must never resume fake order IDs as real
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

    prior_state = load_state(cfg.state_file)

    if prior_state is not None and prior_state.config_signature == current_signature:
        # Same grid config as last run — this is a crash/reboot restart, not
        # a deliberate reconfiguration. Resume exactly where we left off:
        # open positions, compounded quantities, reserved/profit totals.
        state = prior_state
        trader.rehydrate_from_levels(state.levels)
        open_buys = sum(1 for l in state.levels if l.state == "buy_open")
        open_sells = sum(1 for l in state.levels if l.state == "sell_open")
        log.info(f"Resumed prior state (config unchanged): {len(state.levels)} levels, "
                  f"{open_buys} open buy(s), {open_sells} open sell(s), "
                  f"reserved_total={state.reserved_total:.4f}, "
                  f"total_profit_realized={state.total_profit_realized:.4f}")
    else:
        # ZEC's real live grid is built and maintained exclusively by a
        # dedicated, reviewed, position-preserving migration script run by
        # hand (build_zec_pct_ladders.py — the fresh-build onto ZECUSDT,
        # 2026-10-09) — never by this function. Auto-rebuilding here on any
        # config change would cancel every real order, including sell_open
        # ones protecting already-owned ZEC, and replace them with whatever
        # shape this function guesses — which may not match what's actually
        # live. Refusing to run is safer than guessing.
        if prior_state is None:
            raise SystemExit(
                "No state.json found for ZEC, and bot.py does not build ZEC's "
                "real grid directly. Run build_zec_pct_ladders.py --execute "
                "to create state.json first, then start the bot normally."
            )
        raise SystemExit(
            "Config signature changed since the last run. Refusing to "
            "auto-rebuild — that would cancel real orders (including "
            "protected owned-ZEC sells) and replace them with a guessed grid "
            "shape. If this .env change was intentional, use "
            "build_zec_pct_ladders.py (or another dedicated, "
            "position-preserving migration script) instead of just "
            "restarting the bot. If it wasn't intentional, revert the .env "
            "change."
        )

    try:
        consecutive_errors = 0
        while True:
            last_error = None
            current_price = None
            try:
                current_price = trader.get_price()

                # Runs first, ahead of fill-checks, so a tripped breaker
                # pulls resting buy orders before this cycle's fill-check
                # loop would otherwise process any of them filling for real.
                halted = check_circuit_breaker(state, current_price, trader, cfg)

                # Fill-checks (and placing the resulting protective sell for
                # anything that just bought) run BEFORE the order window, so
                # protecting an already-owned position always gets first
                # claim on the exchange's 200-order budget — never crowded
                # out by the window opening brand-new speculative buys.
                levels_to_remove = []
                for lvl in state.levels:
                    if lvl.state == "buy_open":
                        if trader.is_filled(lvl.buy_order_id, current_price):
                            log.info(f"Grid {lvl.index} (${lvl.buy_price}): BUY filled qty={lvl.qty}")
                            log_buy_fill(cfg.ledger_file, lvl.index, lvl.buy_price, lvl.qty,
                                         lvl.one_shot, cfg.dry_run)
                            try:
                                lvl.sell_order_id = trader.place_limit_sell(lvl.sell_price, lvl.qty)
                                lvl.state = "sell_open"
                            except Exception as exc:
                                # We already own this ZEC — leave state as buy_open so
                                # is_filled() (idempotent on an already-filled order) keeps
                                # detecting the fill and retries placing the sell next cycle.
                                log.warning(f"Grid {lvl.index} (${lvl.sell_price}): sell order "
                                             f"rejected after buy filled, will retry: {exc}")

                    elif lvl.state == "sell_open":
                        if trader.is_filled(lvl.sell_order_id, current_price):
                            profit = (lvl.sell_price - lvl.buy_price) * lvl.qty
                            log.info(f"Grid {lvl.index} (${lvl.buy_price}): SELL filled "
                                      f"qty={lvl.qty} profit={profit:.4f} USD"
                                      + (" [one_shot, closing for good]" if lvl.one_shot else ""))
                            reserved_amount = profit * cfg.profit_reserve_ratio
                            log_sell_fill(cfg.ledger_file, lvl.index, lvl.buy_price, lvl.sell_price,
                                          lvl.qty, profit, reserved_amount, profit - reserved_amount,
                                          lvl.one_shot, cfg.dry_run)

                            sold_offset_pct = (lvl.sell_price / lvl.buy_price - 1.0) if lvl.buy_price else 0.0
                            is_legacy = (cfg.pct_ladder_enabled and not lvl.one_shot
                                         and is_legacy_offset(sold_offset_pct, cfg))

                            if lvl.from_coin_reserve:
                                # Checked BEFORE is_legacy on purpose: a
                                # reserve-funded sell's gap can land slightly
                                # wider than its ladder's exact offset_pct
                                # (the taker-avoidance floor in
                                # maybe_fund_sell_from_reserve can win over
                                # the natural cost_basis*(1+offset_pct) price
                                # if ZEC rose since seeding), which would
                                # otherwise misfire the legacy-liquidation
                                # path instead of replenishment.
                                liquidate_coin_reserve_position(state, lvl, profit, trader, filters, cfg)
                                levels_to_remove.append(lvl)
                            elif is_legacy:
                                # Fully dissolve this position instead of
                                # leaving it idle to re-arm at its own
                                # original price forever, outside the
                                # 5-ladder structure -- its ENTIRE principal
                                # (not just profit) merges into the ladders.
                                principal = lvl.buy_price * lvl.qty
                                liquidate_legacy_position(state, profit, principal, cfg, trader, filters)
                                levels_to_remove.append(lvl)
                                log.info(f"Grid {lvl.index}: legacy position liquidated -- "
                                          f"${principal:.4f} principal merged into the 5 ladders "
                                          f"alongside its profit, level removed")
                            else:
                                lvl.state = "closed" if lvl.one_shot else "idle"
                                lvl.buy_order_id = None
                                lvl.sell_order_id = None
                                lvl.qty = 0.0
                                if cfg.pct_ladder_enabled and not lvl.one_shot:
                                    distribute_profit_ladder_aware(state, profit, sold_offset_pct, cfg, trader, filters)
                                else:
                                    # No ladder to attribute this to (pct_ladder
                                    # disabled, or a one_shot level) — reserve
                                    # the whole amount rather than guess.
                                    state.reserved_total += profit
                                    state.total_profit_realized += profit

                for lvl in levels_to_remove:
                    state.levels.remove(lvl)

                maybe_reshape_ladders(state, current_price, trader, filters, cfg)

                maybe_fund_sell_from_reserve(state, current_price, trader, filters, cfg)

                apply_order_window(state, current_price, trader, filters, cfg.max_open_orders,
                                    cfg.taker_avoidance_buffer, halted=halted)

                write_ladder_ledgers(state, cfg)

                save_state(cfg.state_file, state)
                consecutive_errors = 0

            except KeyboardInterrupt:
                raise

            except Exception as exc:
                consecutive_errors += 1
                last_error = f"{exc.__class__.__name__}: {exc}"
                log.error(f"Iteration failed (consecutive_errors={consecutive_errors}): {last_error}")
                log.debug(traceback.format_exc())
                if consecutive_errors >= 5:
                    log.error("5+ consecutive failures — backing off for 5x the normal poll interval")
                    time.sleep(cfg.poll_interval_seconds * 5)

            write_heartbeat(cfg.heartbeat_file, current_price, state, last_error, cfg.dry_run, zone_config={
                "anchor_low": cfg.anchor_low,
                "dense_low": cfg.dense_low,
                "dense_high": cfg.dense_high,
                "anchor_high": cfg.anchor_high,
                "dense_step": cfg.dense_step,
                "dense_sell_offset": cfg.dense_sell_offset,
                "pct_ladder_enabled": cfg.pct_ladder_enabled,
                "pct_ladder_specs": [
                    {"name": name, "step_pct": step_pct, "offset_pct": offset_pct, "window_pct": window_pct}
                    for step_pct, offset_pct, window_pct, name in ladder_specs(cfg)
                ] if cfg.pct_ladder_enabled else [],
            })
            time.sleep(cfg.poll_interval_seconds)

    except KeyboardInterrupt:
        log.info("Shutdown requested, saving state...")
        save_state(cfg.state_file, state)
        log.info("Stopped cleanly.")


if __name__ == "__main__":
    run()
