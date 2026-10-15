"""Percentage-based, price-following, 5-ladder buy/sell structure for ZEC --
direct port of xrp-grid-bot's pct_ladder.py (itself adapted from
hbar-grid-bot's pct_ladder.py, itself adapted from dash-grid-bot's
ladder_window.py), for ZEC's fresh build onto ZECUSDT 2026-10-09: offsets
1%/2%/3%/4%/5%, built in alongside the coin-reserve sell-side backstop and
bidirectional reinvestment from day one, not retrofitted later.

1. Steps and offsets are PERCENTAGES of price, not fixed dollar amounts --
   scale-invariant, so the same module works at any ZEC price level.

2. Each ladder gets its OWN separate reserve pool (`state.ladder_reserved`,
   keyed by ladder name), not one shared `reserved_total`. A shared pool was
   found (live, on DASH) to pool CORRELATED risk, not diversified risk --
   when price falls, every ladder on the same coin gets stressed at the same
   time, so a shared reserve just means every ladder competes for the same
   shrinking pot during exactly the moment it matters most. Separate pools
   mean each ladder's safety net is sized and funded independently, with no
   cross-ladder drain.

Each ladder's zone is a ONE-SIDED window ending exactly at current price: a
resting buy order above current price can never legitimately fill (it would
cross the spread and execute as a taker), so a symmetric window always
pre-builds roughly half its own capital above price -- permanently idle
unless price rises back past it. A one-sided window doubles real downward
reach for the same capital and keeps every rung positioned to become a real
order as price falls toward it.
"""
import logging
import math
import time

log = logging.getLogger("zec-grid-bot")

OFFSET_TOLERANCE_PCT = 0.0005  # 0.05 percentage points -- float-safe matching of a level's own gap to its ladder's offset_pct

# Below this, a ladder's reserve is treated as fully exhausted rather than
# left to the floor-capped draw's geometric decay toward (never reaching)
# zero -- found live 2026-10-13: left unchecked, a long-dry ladder's reserve
# can decay to subatomic float dust (1e-96, 1e-210 seen on DASH/HBAR) and
# keep triggering a no-op reshape every single poll cycle forever.
_NEGLIGIBLE_RESERVE_USD = 1e-6


def _cancel_and_collect(members, trader, id_attr):
    """Try to cancel each member's order. Returns only the members whose
    cancel succeeded -- anything that failed (most likely because it
    already filled for real, moments before we got to it) is left
    completely untouched so the normal fill-check picks it up next cycle.
    Inlined here (rather than imported from zone_shift.py, as DASH/HBAR/XRP's
    copies do) since ZEC's old zone_shift.py was retired along with the rest
    of the hybrid anchor+dense design it belonged to."""
    collected = []
    for lvl in members:
        order_id = getattr(lvl, id_attr)
        if order_id is None:
            collected.append(lvl)
            continue
        try:
            trader.cancel_order(order_id)
            collected.append(lvl)
        except Exception as exc:
            log.warning(f"pct_ladder: level {lvl.index} order {order_id} not cancellable "
                        f"({exc}) -- likely already filled, leaving untouched")
    return collected


def _band_key(offset_pct: float) -> str:
    """Stable dict key for a percentage offset, e.g. 0.02 -> '0.0200'."""
    return f"{offset_pct:.4f}"


def _exact_ladder_match(offset_pct: float, cfg):
    """Which of the 5 ladders a level's own (sell/buy - 1) gap EXACTLY
    matches (within OFFSET_TOLERANCE_PCT), or None if it matches none of
    them. A real rung built by reshape_ladder always has a gap matching its
    own ladder's offset_pct exactly, so this reliably distinguishes "this
    position IS one of the 5 ladders" from "this is a legacy position held
    over from whatever design preceded them". ZEC has no adopted/legacy
    order history the way XRP did, so this will rarely if ever return None
    in practice -- kept for consistency and because a zone-shift-merged or
    manually-adjusted level could theoretically drift outside exact-match
    in the future."""
    for _, ladder_offset_pct, _, name in ladder_specs(cfg):
        if abs(ladder_offset_pct - offset_pct) < OFFSET_TOLERANCE_PCT:
            return name
    return None


def is_legacy_offset(offset_pct: float, cfg) -> bool:
    """True if offset_pct matches none of the 5 ladders exactly -- i.e. a
    position left over from whatever design preceded the ladders, not a
    real rung of any of them."""
    return _exact_ladder_match(offset_pct, cfg) is None


def liquidate_legacy_position(state, profit: float, principal: float, cfg, trader, filters) -> None:
    """Fully dissolves a legacy position on sale instead of leaving it to
    re-arm itself at its own original price forever, outside the 5-ladder
    structure. The caller is responsible for actually removing the level
    from state.levels -- this function only handles the capital.

    PRINCIPAL (what was originally spent to acquire the position) merges
    evenly into the 5 ladders' pending_reinvest_by_band, same as always --
    it's returned capital, not profit, so it's not eligible for the
    coin-growth split below (only profit-derived amounts are).

    PROFIT splits reserve/reinvest same as any ladder-native sale, evenly
    across the 5 ladders; each ladder's own reinvest share then reinvests
    in BOTH directions via _apply_coin_growth_split (coin-reserve growth +
    buy-side pending-reinvest), same as distribute_profit_ladder_aware.

    Principal is NOT subject to profit_reserve_ratio -- reserving part of
    your own principal as a 'safety net' doesn't make sense; only genuine
    profit gets reserved."""
    profit = max(0.0, profit)
    reserve_amount = profit * cfg.profit_reserve_ratio
    distribute_amount = profit - reserve_amount

    state.total_profit_realized += profit

    specs = ladder_specs(cfg)
    if reserve_amount > 0:
        reserve_share = reserve_amount / len(specs)
        for _, _, _, name in specs:
            state.ladder_reserved[name] = state.ladder_reserved.get(name, 0.0) + reserve_share
            state.ladder_reserved_contributed[name] = state.ladder_reserved_contributed.get(name, 0.0) + reserve_share

    principal_share = principal / len(specs)
    distribute_share = distribute_amount / len(specs)
    for _, ladder_offset_pct, _, name in specs:
        band_key = _band_key(ladder_offset_pct)
        state.pending_reinvest_by_band[band_key] = state.pending_reinvest_by_band.get(band_key, 0.0) + principal_share
        _apply_coin_growth_split(state, name, distribute_share, trader, filters, cfg)

    log.info(f"pct_ladder: legacy position liquidated -- principal ${principal:.4f} split evenly "
             f"into buy-side reinvest across all {len(specs)} ladders, profit ${profit:.4f} "
             f"({reserve_amount:.4f} reserved, {distribute_amount:.4f} split between buy-side "
             f"reinvest and coin-reserve growth)")


def distribute_profit_ladder_aware(state, profit: float, sold_offset_pct: float, cfg, trader, filters) -> str:
    """Replaces reinvest.py's distribute_profit for the new ladder system:
    reserve_ratio of profit goes into reserve. The remainder (the reinvest
    share) now reinvests in BOTH directions, via _apply_coin_growth_split:
    cfg.coin_growth_pct of it buys more coin into that ladder's own
    coin_reserve (permanently growing it over time), the rest queues into
    pending_reinvest_by_band, picked up by reshape_ladder's future_pool on
    its next cycle.

    A position whose gap EXACTLY matches one of the 5 ladders (a real rung
    of that ladder) has its ENTIRE profit routed to that one ladder.

    A position whose gap matches NONE of them -- a legacy position held
    over from whatever design preceded the ladders -- has its profit SPLIT
    EVENLY across all 5 ladders instead of funneling to whichever is merely
    closest (found live on XRP: "nearest ladder" routing would have
    funneled everything into one ladder, starving the rest).

    Returns the ladder name profit was routed to (exact-match case), or the
    literal string "legacy-even-split" (even-split case), for logging."""
    if profit <= 0:
        return ""

    specs = ladder_specs(cfg)
    state.total_profit_realized += profit
    exact_name = _exact_ladder_match(sold_offset_pct, cfg)

    if exact_name is not None:
        reserve_amount = profit * cfg.profit_reserve_ratio
        distribute_amount = profit - reserve_amount
        state.ladder_reserved[exact_name] = state.ladder_reserved.get(exact_name, 0.0) + reserve_amount
        state.ladder_reserved_contributed[exact_name] = state.ladder_reserved_contributed.get(exact_name, 0.0) + reserve_amount
        _apply_coin_growth_split(state, exact_name, distribute_amount, trader, filters, cfg)
        log.info(f"pct_ladder: profit {profit:.4f} attributed to {exact_name} (exact offset match) -- "
                 f"{reserve_amount:.4f} to its own reserve (now ${state.ladder_reserved[exact_name]:.4f}), "
                 f"{distribute_amount:.4f} split between buy-side reinvest and coin-reserve growth")
        return exact_name

    share = profit / len(specs)
    for _, ladder_offset_pct, _, name in specs:
        reserve_amount = share * cfg.profit_reserve_ratio
        distribute_amount = share - reserve_amount
        state.ladder_reserved[name] = state.ladder_reserved.get(name, 0.0) + reserve_amount
        state.ladder_reserved_contributed[name] = state.ladder_reserved_contributed.get(name, 0.0) + reserve_amount
        _apply_coin_growth_split(state, name, distribute_amount, trader, filters, cfg)
    log.info(f"pct_ladder: profit {profit:.4f} from a legacy position (offset {sold_offset_pct:.4f}, "
             f"no exact ladder match) -- split evenly across all 5 ladders (${share:.4f} each before reserve split)")
    return "legacy-even-split"


def ladder_specs(cfg):
    """(step_pct, offset_pct, window_pct, name) for ladders 1-5, in order.
    step_pct/offset_pct/window_pct are fractions (0.01 = 1%), not percent
    points, matching this project's existing _pct config convention."""
    return [
        (cfg.ladder1_step_pct, cfg.ladder1_offset_pct, cfg.ladder1_window_pct, "ladder1"),
        (cfg.ladder2_step_pct, cfg.ladder2_offset_pct, cfg.ladder2_window_pct, "ladder2"),
        (cfg.ladder3_step_pct, cfg.ladder3_offset_pct, cfg.ladder3_window_pct, "ladder3"),
        (cfg.ladder4_step_pct, cfg.ladder4_offset_pct, cfg.ladder4_window_pct, "ladder4"),
        (cfg.ladder5_step_pct, cfg.ladder5_offset_pct, cfg.ladder5_window_pct, "ladder5"),
    ]


def compute_ladder_zones(current_price, cfg):
    """Returns {name: (lo, hi)} -- one one-sided window per ladder, each
    [current_price * (1 - window_pct), current_price], clipped to
    [anchor_low, anchor_high]. Uses current_price directly (not a
    whole-dollar-rounded center)."""
    zones = {}
    for step_pct, offset_pct, window_pct, name in ladder_specs(cfg):
        hi = min(current_price, cfg.anchor_high)
        lo = max(current_price * (1.0 - window_pct), cfg.anchor_low)
        zones[name] = (lo, hi)
    return zones


def build_ladder_points(lo, hi, step_pct):
    """Geometric (percentage) spacing price points from lo to hi inclusive,
    each step_pct above the previous -- NOT arithmetic, since a fixed
    dollar step is meaningless across a range that itself spans a wide
    price ratio."""
    if lo <= 0 or hi <= 0 or hi < lo:
        return []
    if step_pct <= 0:
        raise ValueError("step_pct must be positive")
    points = []
    p = lo
    ratio = 1.0 + step_pct
    # ceiling-safe loop count so the last point lands at or just above hi
    n = int(math.floor(math.log(hi / lo) / math.log(ratio) + 1e-9)) + 1 if hi > lo else 0
    for i in range(n + 1):
        points.append(round(lo * (ratio ** i), 10))
    return [p for p in points if p <= hi + 1e-9]


def _in_range(price, lo, hi):
    return lo - 1e-9 <= price <= hi + 1e-9


def _is_stuck_idle(lvl, current_price, min_notional):
    """An idle level whose buy price is AT OR ABOVE current price can
    never fill without crossing the spread as a taker. Also flags a level
    whose target notional has decayed (via repeated partial fills/float
    drift) below the exchange minimum, which would otherwise sit forever
    rejecting."""
    if lvl.state != "idle":
        return False
    if lvl.buy_price >= current_price:
        return True
    notional = lvl.target_qty() * lvl.buy_price
    return 0 < notional < min_notional


def _draw_from_ladder_reserve(state, cfg, name, max_amount=None, ignore_floor=False):
    """Draws from state.ladder_reserved[name] -- THIS ladder's own pool
    only, never another ladder's. Capped so the pool never drops below
    reserved_draw_floor_pct of its OWN value at the moment of THIS draw
    (geometric decay toward, never reaching, zero across repeated draws
    during a sustained decline) -- UNLESS ignore_floor=True, used only for
    the completely-dry emergency-bootstrap case in reshape_ladder: a
    ladder that can never place a single order can also never earn the
    profit that would replenish its own reserve, so it stays stuck at
    zero rungs forever regardless of how much sits in reserve. Escaping
    that dead end takes priority over the normal retention floor."""
    available = state.ladder_reserved.get(name, 0.0)
    if available <= _NEGLIGIBLE_RESERVE_USD:
        if available > 0:
            state.ladder_reserved[name] = 0.0  # snap dust to exactly zero, stop the decay loop for good
        return 0.0
    drawable = available if ignore_floor else available * (1.0 - cfg.reserved_draw_floor_pct)
    draw = min(drawable, max_amount) if max_amount is not None else drawable
    draw = max(0.0, draw)
    if draw <= 0:
        return 0.0
    remainder = available - draw
    state.ladder_reserved[name] = 0.0 if remainder <= _NEGLIGIBLE_RESERVE_USD else remainder
    state.ladder_reserved_drawn[name] = state.ladder_reserved_drawn.get(name, 0.0) + draw
    return draw


def _min_capital_for_one_level(lo, hi, step_pct, reference_price, dca_exponent, min_notional):
    """Smallest total_capital that would let the single best (highest-DCA-
    weight) point in this window clear min_notional, under the same
    weighting _dca_build_levels uses. Lets reshape_ladder tell, before
    drawing anything, whether the normal floor-capped reserve draw would
    be enough to escape a completely-dry ladder, or whether this specific
    bootstrap needs to draw past the floor -- see _draw_from_ladder_reserve's
    ignore_floor."""
    points = build_ladder_points(lo, hi, step_pct)
    if not points:
        return float("inf")
    weights = [(reference_price / p) ** dca_exponent for p in points]
    best_weight = max(weights)
    total_weight = sum(weights)
    return min_notional * total_weight / best_weight


def _dca_build_levels(points, total_capital, reference_price, dca_exponent, offset_pct, filters, start_index):
    """Equal-ratio DCA weighting: cheaper points get a bigger USD share,
    weight = (reference_price / buy_price) ** dca_exponent. Sell price is
    buy_price scaled up by (1 + offset_pct), not a dollar addition."""
    if not points:
        return []
    weights = [(reference_price / p) ** dca_exponent for p in points]
    total_weight = sum(weights)
    from grid import GridLevel
    levels = []
    for i, buy_price in enumerate(points):
        usd_alloc = total_capital * weights[i] / total_weight
        qty = usd_alloc / buy_price
        qty = filters.round_qty(qty) if hasattr(filters, "round_qty") else qty
        if qty <= 0:
            continue
        sell_price = round(buy_price * (1.0 + offset_pct), 10)
        levels.append(GridLevel(
            index=start_index + i, buy_price=buy_price, sell_price=sell_price, base_qty=qty,
        ))
    return levels


def _thin_to_affordable(points, total_capital, reference_price, dca_exponent, min_notional):
    """Drops points whose DCA-weighted share would fall under min_notional
    -- prevents building levels the exchange would reject outright."""
    if not points:
        return points
    weights = [(reference_price / p) ** dca_exponent for p in points]
    total_weight = sum(weights)
    kept = [p for p, w in zip(points, weights) if total_capital * w / total_weight >= min_notional]
    return kept


def reshape_ladder(state, name, lo, hi, step_pct, offset_pct, trader, filters, cfg, current_price):
    """Fills genuine gaps in the current window using this ladder's own
    future-buy pool and reserve draw, then (only once the window is fully
    covered) grows existing survivors. A real resting order (buy_open), once
    live, is NEVER cancelled just because the window has since drifted past
    it (2026-10-15, explicit request: coverage should only ever grow, never
    churn -- a rung earned its spot, the window moving on doesn't evict it).
    The only levels ever collected are ones that were never real orders to
    begin with: idle levels that are structurally stuck (would cross the
    spread as a taker, or have decayed below min_notional). Held (sell_open)
    and any one_shot levels are never inspected -- a cancel failure (already
    filled for real) leaves that level completely untouched."""
    band_key = _band_key(offset_pct)
    future_pool = state.pending_reinvest_by_band.get(band_key, 0.0)

    def needs_reshaping(lvl):
        return _is_stuck_idle(lvl, current_price, filters.min_notional)

    same_ladder = [lvl for lvl in state.levels
                   if not lvl.one_shot and lvl.state in ("idle", "buy_open")
                   and abs((lvl.sell_price / lvl.buy_price - 1.0) - offset_pct) < OFFSET_TOLERANCE_PCT]
    mismatched = [lvl for lvl in same_ladder if needs_reshaping(lvl)]
    own_capital = sum(lvl.target_qty() * lvl.buy_price for lvl in same_ladder)

    reserve_draw = 0.0
    if future_pool <= 0:
        if not same_ladder:
            available = state.ladder_reserved.get(name, 0.0)
            needed_min = _min_capital_for_one_level(lo, hi, step_pct, cfg.reference_price,
                                                      cfg.dca_exponent, filters.min_notional)
            normal_preview = available * (1.0 - cfg.reserved_draw_floor_pct)
            if normal_preview < needed_min <= available:
                # The normal floor-respecting draw would leave this ladder
                # stuck at zero rungs indefinitely (it can never earn the
                # profit that would replenish its own reserve while dry),
                # but the full reserve would be enough to place one real
                # order -- bootstrap takes priority over retention here.
                reserve_draw = _draw_from_ladder_reserve(state, cfg, name, ignore_floor=True)
                if reserve_draw > 0:
                    log.warning(f"pct_ladder: {name} emergency bootstrap -- the normal floor-capped "
                                f"draw (${normal_preview:.4f}) wasn't enough to place a single order "
                                f"(needs ${needed_min:.4f}); drew the full remaining reserve "
                                f"(${reserve_draw:.4f}) instead of staying stuck at zero rungs")
            else:
                reserve_draw = _draw_from_ladder_reserve(state, cfg, name)
                if reserve_draw > 0:
                    log.warning(f"pct_ladder: {name} is completely dry -- drew ${reserve_draw:.4f} "
                                f"from its own reserve (${state.ladder_reserved.get(name, 0.0):.4f} remaining)")
        elif own_capital < cfg.reserved_topup_threshold_usd:
            needed = cfg.reserved_topup_threshold_usd - own_capital
            reserve_draw = _draw_from_ladder_reserve(state, cfg, name, max_amount=needed)
            if reserve_draw > 0:
                log.warning(f"pct_ladder: {name} low (${own_capital:.4f} < ${cfg.reserved_topup_threshold_usd:.2f}) "
                            f"-- drew ${reserve_draw:.4f} from its own reserve")

    if not mismatched and future_pool <= 0 and reserve_draw <= 0:
        return 0

    collected = _cancel_and_collect(mismatched, trader, "buy_order_id") if mismatched else []
    if mismatched and not collected and future_pool <= 0 and reserve_draw <= 0:
        return 0

    collected_capital = sum(l.target_qty() * l.buy_price for l in collected)
    total_capital = collected_capital + future_pool + reserve_draw
    if future_pool > 0:
        state.pending_reinvest_by_band[band_key] = 0.0
    for lvl in collected:
        state.levels.remove(lvl)
    if total_capital <= 0:
        return 0

    matching = [lvl for lvl in state.levels
                if not lvl.one_shot and lvl.state in ("idle", "buy_open")
                and abs((lvl.sell_price / lvl.buy_price - 1.0) - offset_pct) < OFFSET_TOLERANCE_PCT
                and not needs_reshaping(lvl)]
    all_points = build_ladder_points(lo, hi, step_pct)
    covered = {round(lvl.buy_price, 10) for lvl in matching}
    gap_points = [p for p in all_points if round(p, 10) not in covered and p < current_price]

    if gap_points:
        gap_points = _thin_to_affordable(gap_points, total_capital, cfg.reference_price, cfg.dca_exponent, filters.min_notional)
        fresh = _dca_build_levels(gap_points, total_capital, cfg.reference_price, cfg.dca_exponent,
                                   offset_pct, filters, state.next_level_index)
        if fresh:
            state.next_level_index += len(fresh)
            state.levels.extend(fresh)
            log.info(f"pct_ladder: {name} reshaped {len(collected)} level(s) (+${future_pool:.4f} future-buy, "
                     f"+${reserve_draw:.4f} from reserve) into {len(fresh)} fresh level(s) (${total_capital:.4f}), "
                     f"{len(matching)} survivor(s) untouched")
            return len(fresh)
        # gap_points existed before thinning/min-notional filtering but
        # nothing survived to build -- fall through instead of discarding
        # total_capital (see the no-op branch below for why this matters).

    if matching:
        per_level = total_capital / len(matching)
        now = time.time()
        for lvl in matching:
            extra_qty = per_level / lvl.buy_price
            lvl.extra_qty += extra_qty
            if per_level > 1e-6:  # skip float-dust "growths" (seen live: 4e-95) -- not a real reinvestment
                lvl.last_reinvest_usd = per_level
                lvl.last_reinvest_ts = now
        log.info(f"pct_ladder: {name} reshaped {len(collected)} level(s) by growing {len(matching)} "
                 f"existing level(s) (${total_capital:.4f} added, window already fully covered)")
        return 0

    # Nothing buildable (too thin to clear min_notional) and no survivor to
    # grow either -- return the capital to where it came from instead of
    # letting it vanish.
    if reserve_draw > 0:
        state.ladder_reserved[name] = state.ladder_reserved.get(name, 0.0) + reserve_draw
        # Undo the ladder_reserved_drawn increment _draw_from_ladder_reserve
        # just made for this same draw -- otherwise a ladder stuck in this
        # draw-then-return loop inflates its "all-time drawn" figure by the
        # same amount every single poll cycle forever, even though no real
        # capital is actually lost (it's returned right here).
        state.ladder_reserved_drawn[name] = max(
            0.0, state.ladder_reserved_drawn.get(name, 0.0) - reserve_draw)
    if collected_capital > 0:
        state.ladder_reserved[name] = state.ladder_reserved.get(name, 0.0) + collected_capital
    if future_pool > 0:
        state.pending_reinvest_by_band[band_key] = state.pending_reinvest_by_band.get(band_key, 0.0) + future_pool
    if reserve_draw > 0 or future_pool > 0 or collected_capital > 0:
        log.info(f"pct_ladder: {name} drew ${total_capital:.4f} but had nothing viable to build "
                 f"or grow -- returned to its own reserve/pending pool untouched")
    return 0


def maybe_reshape_ladders(state, current_price, trader, filters, cfg):
    """Runs every poll cycle. Deliberately NOT gated on price having moved
    -- a ladder can get stuck while price just sits still (capital eroded
    under min_notional), and reshape_ladder is a cheap no-op when nothing
    is actually wrong."""
    if not cfg.pct_ladder_enabled:
        return 0
    zones = compute_ladder_zones(current_price, cfg)
    total_reshaped = 0
    for step_pct, offset_pct, window_pct, name in ladder_specs(cfg):
        lo, hi = zones[name]
        if hi <= lo:
            continue
        total_reshaped += reshape_ladder(state, name, lo, hi, step_pct, offset_pct, trader, filters, cfg, current_price)
    return total_reshaped


def maybe_fund_sell_from_reserve(state, current_price, trader, filters, cfg):
    """Runs every poll cycle, after maybe_reshape_ladders. A ladder can only
    ever have something to sell once one of its own buy rungs has actually
    filled -- unlike the buy side, there's no reserve a ladder can draw on
    to conjure up a sell order, because a sell order requires actually
    owning the coin. seed_coin_reserve.py (one-time) and
    replenish_coin_reserve (below, ongoing) keep state.coin_reserve_qty
    stocked with a standing, unlisted amount of coin per ladder precisely
    for this: if a ladder currently has ZERO live sell_open positions of
    its own (gap matches its offset_pct exactly) but still holds reserve
    coin, list it as a real sell order right now -- the ladder is never
    left with nothing to offer just because price hasn't dipped into its
    window yet. Cheap no-op in the common case (any ladder with a live
    sell already, or an empty reserve)."""
    from grid import GridLevel

    listed = 0
    for _, offset_pct, _, name in ladder_specs(cfg):
        qty = state.coin_reserve_qty.get(name, 0.0)
        if qty <= 0:
            continue

        own_sells = sum(1 for lvl in state.levels
                         if lvl.state == "sell_open" and lvl.buy_price > 0
                         and abs((lvl.sell_price / lvl.buy_price - 1.0) - offset_pct) < OFFSET_TOLERANCE_PCT)
        if own_sells > 0:
            continue

        cost_basis = state.coin_reserve_cost_basis.get(name, 0.0)
        round_qty = filters.round_qty(qty)
        if cost_basis <= 0 or round_qty <= 0:
            continue

        # Fresh price right before placing, same reasoning apply_order_window
        # uses for buys: the cycle's current_price may already be a few
        # seconds stale by the time we get here.
        fresh_price = trader.get_price()
        natural_price = cost_basis * (1.0 + offset_pct)
        sell_price = filters.round_price(max(natural_price, fresh_price + cfg.taker_avoidance_buffer))
        if round_qty * sell_price < filters.min_notional:
            log.warning(f"pct_ladder: {name} coin reserve ({round_qty} @ ~${sell_price:.6f}) too thin "
                        f"to clear min_notional -- left untouched")
            continue

        try:
            order_id = trader.place_limit_sell(sell_price, round_qty)
        except Exception as exc:
            log.warning(f"pct_ladder: {name} coin-reserve sell placement failed ({exc}) -- "
                        f"reserve left untouched, will retry next cycle")
            continue

        new_lvl = GridLevel(index=state.next_level_index, buy_price=cost_basis, sell_price=sell_price,
                             base_qty=round_qty, qty=round_qty, state="sell_open",
                             sell_order_id=order_id, from_coin_reserve=True, coin_reserve_ladder=name)
        state.next_level_index += 1
        state.levels.append(new_lvl)
        state.coin_reserve_qty[name] = 0.0
        listed += 1
        log.info(f"pct_ladder: {name} had zero live sells of its own -- listed its coin reserve "
                 f"({round_qty} @ ${sell_price:.6f}, cost basis ${cost_basis:.6f}) as a real sell order")
    return listed


def replenish_coin_reserve(state, name, amount, trader, filters, cfg):
    """Buys more of this ladder's coin_reserve_qty[name] with `amount` USD
    -- same immediate-fill technique seed_coin_reserve.py and DASH's bulk
    ladder both use (a limit buy placed slightly ABOVE current price on
    purpose, to cross the spread and fill now rather than rest and wait).
    Two callers, two purposes:
    - liquidate_coin_reserve_position: `amount` = a just-sold reserve
      position's own principal, closing the loop back to its original
      seeded size (replenishment).
    - _apply_coin_growth_split: `amount` = a slice of ordinary profit
      reinvestment (coin_growth_pct), permanently GROWING the reserve
      beyond its original seed size over time, gradually, across every
      ladder-native/legacy/reserve sale -- not just restoring it.
    Either way this never touches the ladder's buying power or its own
    USD reserve pool.

    Returns True if the buy succeeded, False otherwise -- the caller is
    responsible for routing `amount` somewhere safe instead (e.g.
    pending_reinvest_by_band) on a False return. This function never
    discards capital itself, it only reports whether it could put it to
    use. No synchronous fill verification here (unlike the one-time seed
    script) -- same lighter-weight, already-proven pattern DASH's bulk
    ladder uses for its own immediate-fill purchase, since this runs
    unattended on every qualifying sale rather than as a manually-reviewed
    script."""
    if amount <= 0:
        return False
    fresh_price = trader.get_price()
    buy_price = filters.round_price(fresh_price * 1.002)
    qty = filters.round_qty(amount / buy_price) if buy_price > 0 else 0.0
    if qty <= 0 or qty * buy_price < filters.min_notional:
        log.warning(f"pct_ladder: {name} coin-reserve buy amount (${amount:.4f}) too thin to "
                    f"clear min_notional -- skipped this cycle")
        return False
    try:
        trader.place_limit_buy(buy_price, qty)
    except Exception as exc:
        log.warning(f"pct_ladder: {name} coin-reserve buy failed ({exc}) -- reserve not grown")
        return False
    old_qty = state.coin_reserve_qty.get(name, 0.0)
    old_cost_basis = state.coin_reserve_cost_basis.get(name, 0.0)
    new_qty = old_qty + qty
    # Weighted average, not a plain overwrite -- matters once coin_growth_pct
    # can call this repeatedly while the reserve is ALREADY nonzero (gradual
    # growth from many small profit events), not just the old draw-then-
    # replenish case where it always started at exactly zero.
    state.coin_reserve_cost_basis[name] = (
        (old_qty * old_cost_basis + qty * buy_price) / new_qty if new_qty > 0 else buy_price)
    state.coin_reserve_qty[name] = new_qty
    log.info(f"pct_ladder: {name} coin reserve grew -- bought {qty} @ ${buy_price:.6f} "
             f"(${qty * buy_price:.4f} of ${amount:.4f}), now {new_qty} @ avg "
             f"${state.coin_reserve_cost_basis[name]:.6f}")
    return True


def _apply_coin_growth_split(state, name, amount, trader, filters, cfg):
    """Reinvest in both directions: splits a per-ladder profit-reinvest
    share into a coin-growth portion (buys more coin into that ladder's
    own coin_reserve via replenish_coin_reserve -- permanently growing it
    over time, not just restoring it to its seeded size) and a buy-side
    portion (queued into pending_reinvest_by_band). `cfg.coin_growth_pct`
    (0 = all buy-side; 1 = all coin-growth) controls the split.

    A single sale's coin-growth slice is usually far too small to clear
    min_notional on its own -- found live 2026-10-13 on XRP (same pct_ladder
    design): 48 of 52 attempts since this feature went live had been
    silently skipped, with the slice falling back to buy-side every single
    time instead of ever becoming coin. Rather than discarding that
    opportunity each time, the slice now ACCUMULATES in
    state.pending_coin_growth[name] across sales until the running total
    is big enough to actually clear min_notional, then resets to zero. Only
    the non-coin-growth portion (band_amount) still goes to buy-side reinvest
    every cycle, same as always -- never discarding capital either way."""
    coin_growth_amount = amount * cfg.coin_growth_pct
    band_amount = amount - coin_growth_amount
    if coin_growth_amount > 0:
        pending = state.pending_coin_growth.get(name, 0.0) + coin_growth_amount
        if replenish_coin_reserve(state, name, pending, trader, filters, cfg):
            state.pending_coin_growth[name] = 0.0
        else:
            state.pending_coin_growth[name] = pending
    band_key = _band_key(next(o for _, o, _, n in ladder_specs(cfg) if n == name))
    state.pending_reinvest_by_band[band_key] = state.pending_reinvest_by_band.get(band_key, 0.0) + band_amount


def liquidate_coin_reserve_position(state, lvl, profit, trader, filters, cfg) -> bool:
    """Called from bot.py's sell-fill loop when a from_coin_reserve=True
    level sells. Routes profit directly to lvl.coin_reserve_ladder -- the
    ladder recorded explicitly on the level itself, NOT re-derived from the
    sell/buy gap, since the taker-avoidance floor in
    maybe_fund_sell_from_reserve can make that gap land slightly wider than
    the ladder's exact offset_pct (which would otherwise misroute via
    distribute_profit_ladder_aware's own offset matching). Reserve share of
    profit goes to that ladder's own reserve, same as any normal sale. The
    caller is responsible for removing the level from state.levels -- this
    function only handles capital.

    Its own PROFIT reinvest share (separate from the principal
    replenishment above) also reinvests in both directions via
    _apply_coin_growth_split, same as any other sale -- so a reserve
    position's profit can grow the reserve FURTHER, on top of the
    principal already restoring it to its seeded size.

    Returns True if the coin reserve was successfully replenished from this
    sale's principal."""
    name = lvl.coin_reserve_ladder
    principal = lvl.buy_price * lvl.qty
    profit = max(0.0, profit)
    reserve_amount = profit * cfg.profit_reserve_ratio
    distribute_amount = profit - reserve_amount

    state.ladder_reserved[name] = state.ladder_reserved.get(name, 0.0) + reserve_amount
    state.ladder_reserved_contributed[name] = state.ladder_reserved_contributed.get(name, 0.0) + reserve_amount
    state.total_profit_realized += profit

    refilled = replenish_coin_reserve(state, name, principal, trader, filters, cfg)
    if not refilled:
        band_key = _band_key(next(o for _, o, _, n in ladder_specs(cfg) if n == name))
        state.pending_reinvest_by_band[band_key] = state.pending_reinvest_by_band.get(band_key, 0.0) + principal

    _apply_coin_growth_split(state, name, distribute_amount, trader, filters, cfg)

    log.info(f"pct_ladder: {name} coin-reserve position liquidated -- profit {profit:.4f} "
             f"({reserve_amount:.4f} reserved, {distribute_amount:.4f} split between buy-side "
             f"reinvest and further coin-reserve growth), principal ${principal:.4f} "
             + ("replenished the reserve" if refilled else "queued to pending-reinvest (replenishment failed)"))
    return refilled
