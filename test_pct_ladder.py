import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from grid import GridLevel
from state import BotState
import pct_ladder as pl

# NOTE: make_cfg's anchor_low/anchor_high/reference_price below are kept at
# the same values as xrp-grid-bot's test suite this was ported from (NOT
# ZEC's real $200-$2000 range) -- pct_ladder.py is fully scale-invariant
# (percentage-based), so these tests are pure logic/math checks, and several
# specific price literals throughout this file (e.g. 1.33-1.40) were chosen
# to interact correctly with these particular default bounds (clipping
# tests in particular). Changing the defaults without rescaling every
# literal would silently break those interactions. ZEC's real config uses
# ANCHOR_LOW=200.00/ANCHOR_HIGH=2000.00 in its actual .env -- unrelated to
# this file.


def make_cfg(**overrides):
    base = dict(
        anchor_low=0.90, anchor_high=2.20,
        ladder1_step_pct=0.01, ladder1_offset_pct=0.01, ladder1_window_pct=0.05,
        ladder2_step_pct=0.02, ladder2_offset_pct=0.02, ladder2_window_pct=0.10,
        ladder3_step_pct=0.03, ladder3_offset_pct=0.03, ladder3_window_pct=0.15,
        ladder4_step_pct=0.04, ladder4_offset_pct=0.04, ladder4_window_pct=0.20,
        ladder5_step_pct=0.05, ladder5_offset_pct=0.05, ladder5_window_pct=0.25,
        reserved_draw_floor_pct=0.5, reserved_topup_threshold_usd=5.0,
        reference_price=1.40, dca_exponent=1.0, profit_reserve_ratio=0.5,
        pct_ladder_enabled=True, taker_avoidance_buffer=0.005,
        coin_growth_pct=0.0,  # off by default so most tests' 100%-to-buy-side assertions hold; override per-test to exercise growth
    )
    base.update(overrides)
    return type("Cfg", (), base)()


class FakeTrader:
    def __init__(self, fail_cancel_ids=None, price=1.40, fail_sell=False, fail_buy=False):
        self.cancelled = []
        self.fail_cancel_ids = fail_cancel_ids or set()
        self.price = price
        self.fail_sell = fail_sell
        self.fail_buy = fail_buy
        self.placed_sells = []  # list[(price, qty)]
        self.placed_buys = []   # list[(price, qty)]
        self._next_id = 0

    def cancel_order(self, order_id):
        if order_id in self.fail_cancel_ids:
            raise RuntimeError("already filled")
        self.cancelled.append(order_id)

    def get_price(self):
        return self.price

    def place_limit_sell(self, price, qty):
        if self.fail_sell:
            raise RuntimeError("sell rejected")
        self._next_id += 1
        self.placed_sells.append((price, qty))
        return f"sell-{self._next_id}"

    def place_limit_buy(self, price, qty):
        if self.fail_buy:
            raise RuntimeError("buy rejected")
        self._next_id += 1
        self.placed_buys.append((price, qty))
        return f"buy-{self._next_id}"


class FakeFilters:
    def __init__(self, min_notional=0.01):
        self.min_notional = min_notional

    def round_qty(self, qty):
        return round(qty, 6)

    def round_price(self, price):
        return round(price, 6)


class TestBuildLadderPoints:
    def test_geometric_spacing_covers_range(self):
        points = pl.build_ladder_points(1.20, 1.40, 0.01)
        assert points[0] == pytest.approx(1.20)
        assert points[-1] <= 1.40 + 1e-9
        for a, b in zip(points, points[1:]):
            assert b / a == pytest.approx(1.01, rel=1e-6)

    def test_empty_when_hi_below_lo(self):
        assert pl.build_ladder_points(1.40, 1.20, 0.01) == []

    def test_empty_on_nonpositive_bounds(self):
        assert pl.build_ladder_points(0.0, 1.40, 0.01) == []

    def test_rejects_nonpositive_step(self):
        with pytest.raises(ValueError):
            pl.build_ladder_points(1.20, 1.40, 0.0)


class TestComputeLadderZones:
    def test_one_sided_window_ends_at_current_price(self):
        cfg = make_cfg()
        zones = pl.compute_ladder_zones(1.40, cfg)
        lo, hi = zones["ladder1"]
        assert hi == pytest.approx(1.40)
        assert lo == pytest.approx(1.40 * 0.95)  # 5% window

    def test_clipped_to_anchor_bounds(self):
        cfg = make_cfg(anchor_low=1.35)
        zones = pl.compute_ladder_zones(1.40, cfg)
        lo, _ = zones["ladder1"]
        assert lo == pytest.approx(1.35)  # would be 1.33 unclipped, anchor_low wins

    def test_different_ladders_have_different_window_widths(self):
        cfg = make_cfg()
        zones = pl.compute_ladder_zones(1.40, cfg)
        lo1, _ = zones["ladder1"]
        lo5, _ = zones["ladder5"]
        assert lo5 < lo1  # ladder5's wider window_pct reaches further down


class TestIsStuckIdle:
    def test_idle_above_current_price_is_stuck(self):
        lvl = GridLevel(index=0, buy_price=1.40, sell_price=1.414, base_qty=10, state="idle")
        assert pl._is_stuck_idle(lvl, current_price=1.35, min_notional=0.01) is True

    def test_idle_below_current_price_not_stuck(self):
        lvl = GridLevel(index=0, buy_price=1.30, sell_price=1.313, base_qty=10, state="idle")
        assert pl._is_stuck_idle(lvl, current_price=1.35, min_notional=0.01) is False

    def test_non_idle_never_stuck(self):
        lvl = GridLevel(index=0, buy_price=1.40, sell_price=1.414, base_qty=10, state="buy_open")
        assert pl._is_stuck_idle(lvl, current_price=1.35, min_notional=0.01) is False

    def test_decayed_notional_below_minimum_is_stuck(self):
        lvl = GridLevel(index=0, buy_price=1.30, sell_price=1.313, base_qty=0.001, state="idle")
        assert pl._is_stuck_idle(lvl, current_price=1.35, min_notional=0.01) is True


class TestExactLadderMatch:
    def test_exact_match(self):
        cfg = make_cfg()
        assert pl._exact_ladder_match(0.03, cfg) == "ladder3"

    def test_legacy_offset_matches_nothing(self):
        # A legacy/adopted position's ~5% gap -- close to ladder5's 5% but
        # NOT exact, so this must return None (routes to the even-split
        # path), not "ladder5".
        cfg = make_cfg()
        assert pl._exact_ladder_match(0.048, cfg) is None

    def test_within_tolerance_still_counts_as_exact(self):
        cfg = make_cfg()
        assert pl._exact_ladder_match(0.0301, cfg) == "ladder3"  # within OFFSET_TOLERANCE_PCT of 0.03

    def test_boundary_outside_tolerance_matches_nothing(self):
        cfg = make_cfg()
        assert pl._exact_ladder_match(0.024, cfg) is None  # 0.4pp off ladder2's 2%, outside tolerance


class TestDistributeProfitLadderAware:
    def test_exact_match_routes_fully_to_owning_ladder(self):
        cfg = make_cfg()
        state = BotState()
        name = pl.distribute_profit_ladder_aware(state, profit=1.0, sold_offset_pct=0.01, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        assert name == "ladder1"
        assert state.ladder_reserved["ladder1"] == pytest.approx(0.5)
        assert state.ladder_reserved_contributed["ladder1"] == pytest.approx(0.5)
        assert state.pending_reinvest_by_band[pl._band_key(0.01)] == pytest.approx(0.5)
        assert state.total_profit_realized == pytest.approx(1.0)

    def test_exact_match_other_ladders_untouched(self):
        cfg = make_cfg()
        state = BotState()
        pl.distribute_profit_ladder_aware(state, profit=1.0, sold_offset_pct=0.01, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        assert state.ladder_reserved.get("ladder2", 0.0) == 0.0
        assert state.ladder_reserved.get("ladder3", 0.0) == 0.0

    def test_legacy_position_splits_evenly_across_all_five(self):
        # Reproduces the real scenario found on XRP: a legacy position whose
        # gap (~5%) is closest to ladder5 but not an exact match must NOT
        # have its profit funneled entirely into ladder5 -- it splits
        # evenly across all 5 instead.
        cfg = make_cfg()
        state = BotState()
        name = pl.distribute_profit_ladder_aware(state, profit=5.0, sold_offset_pct=0.048, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        assert name == "legacy-even-split"
        for n in range(1, 6):
            ladder = f"ladder{n}"
            assert state.ladder_reserved[ladder] == pytest.approx(0.5)  # 5.0/5=1.0 share, 50% reserved
            assert state.ladder_reserved_contributed[ladder] == pytest.approx(0.5)
        assert state.total_profit_realized == pytest.approx(5.0)

    def test_legacy_position_queues_pending_reinvest_for_every_ladder(self):
        cfg = make_cfg()
        state = BotState()
        pl.distribute_profit_ladder_aware(state, profit=5.0, sold_offset_pct=0.048, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        for _, offset_pct, _, _ in pl.ladder_specs(cfg):
            assert state.pending_reinvest_by_band[pl._band_key(offset_pct)] == pytest.approx(0.5)

    def test_zero_or_negative_profit_is_noop(self):
        cfg = make_cfg()
        state = BotState()
        pl.distribute_profit_ladder_aware(state, profit=0.0, sold_offset_pct=0.01, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        assert state.ladder_reserved == {}
        assert state.total_profit_realized == 0.0


class TestIsLegacyOffset:
    def test_exact_match_is_not_legacy(self):
        cfg = make_cfg()
        assert pl.is_legacy_offset(0.03, cfg) is False

    def test_no_match_is_legacy(self):
        cfg = make_cfg()
        assert pl.is_legacy_offset(0.048, cfg) is True


class TestLiquidateLegacyPosition:
    def test_principal_and_distribute_share_merge_evenly_into_pending(self):
        cfg = make_cfg()
        state = BotState()
        pl.liquidate_legacy_position(state, profit=1.0, principal=100.0, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        # profit=1.0 -> reserve 0.5, distribute 0.5 -> merge_amount = 100.5, /5 = 20.1 each
        for _, offset_pct, _, _ in pl.ladder_specs(cfg):
            assert state.pending_reinvest_by_band[pl._band_key(offset_pct)] == pytest.approx(20.1)

    def test_only_profit_is_reserved_not_principal(self):
        cfg = make_cfg()
        state = BotState()
        pl.liquidate_legacy_position(state, profit=1.0, principal=100.0, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        for n in range(1, 6):
            assert state.ladder_reserved[f"ladder{n}"] == pytest.approx(0.1)  # 1.0 profit * 0.5 / 5, NOT principal

    def test_total_profit_realized_tracks_profit_only(self):
        cfg = make_cfg()
        state = BotState()
        pl.liquidate_legacy_position(state, profit=1.0, principal=100.0, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        assert state.total_profit_realized == pytest.approx(1.0)

    def test_zero_profit_still_merges_principal(self):
        cfg = make_cfg()
        state = BotState()
        pl.liquidate_legacy_position(state, profit=0.0, principal=50.0, cfg=cfg, trader=FakeTrader(), filters=FakeFilters())
        for _, offset_pct, _, _ in pl.ladder_specs(cfg):
            assert state.pending_reinvest_by_band[pl._band_key(offset_pct)] == pytest.approx(10.0)
        assert state.ladder_reserved == {}


class TestDrawFromLadderReserve:
    def test_caps_at_floor_pct(self):
        cfg = make_cfg(reserved_draw_floor_pct=0.5)
        state = BotState(ladder_reserved={"ladder1": 10.0})
        draw = pl._draw_from_ladder_reserve(state, cfg, "ladder1")
        assert draw == pytest.approx(5.0)
        assert state.ladder_reserved["ladder1"] == pytest.approx(5.0)

    def test_never_touches_a_different_ladders_pool(self):
        cfg = make_cfg()
        state = BotState(ladder_reserved={"ladder1": 10.0, "ladder2": 50.0})
        pl._draw_from_ladder_reserve(state, cfg, "ladder1")
        assert state.ladder_reserved["ladder2"] == 50.0  # untouched

    def test_empty_pool_draws_nothing(self):
        cfg = make_cfg()
        state = BotState(ladder_reserved={"ladder1": 0.0})
        assert pl._draw_from_ladder_reserve(state, cfg, "ladder1") == 0.0

    def test_max_amount_caps_below_floor_limit(self):
        cfg = make_cfg(reserved_draw_floor_pct=0.5)
        state = BotState(ladder_reserved={"ladder1": 10.0})
        draw = pl._draw_from_ladder_reserve(state, cfg, "ladder1", max_amount=2.0)
        assert draw == pytest.approx(2.0)
        assert state.ladder_reserved["ladder1"] == pytest.approx(8.0)

    def test_records_cumulative_drawn(self):
        cfg = make_cfg(reserved_draw_floor_pct=0.5)
        state = BotState(ladder_reserved={"ladder1": 10.0})
        pl._draw_from_ladder_reserve(state, cfg, "ladder1")
        pl._draw_from_ladder_reserve(state, cfg, "ladder1")
        assert state.ladder_reserved_drawn["ladder1"] == pytest.approx(5.0 + 2.5)


class TestReshapeLadder:
    def test_builds_fresh_levels_filling_the_window(self):
        cfg = make_cfg()
        state = BotState(levels=[], ladder_reserved={"ladder1": 5.0})
        trader, filters = FakeTrader(), FakeFilters()
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n > 0
        assert len(state.levels) == n
        for lvl in state.levels:
            assert lvl.buy_price < 1.40
            assert lvl.sell_price == pytest.approx(lvl.buy_price * 1.01)
            assert lvl.state == "idle"

    def test_draws_from_its_own_dry_reserve_to_reseed(self):
        cfg = make_cfg()
        state = BotState(levels=[], ladder_reserved={"ladder1": 10.0})
        trader, filters = FakeTrader(), FakeFilters()
        pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                           trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert state.ladder_reserved["ladder1"] < 10.0  # drew from itself to seed the empty ladder
        assert sum(l.target_qty() * l.buy_price for l in state.levels) > 0

    def test_never_draws_from_a_different_ladders_reserve(self):
        cfg = make_cfg()
        state = BotState(levels=[], ladder_reserved={"ladder1": 10.0, "ladder2": 999.0})
        trader, filters = FakeTrader(), FakeFilters()
        pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                           trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert state.ladder_reserved["ladder2"] == 999.0

    def test_held_sell_open_levels_never_touched(self):
        cfg = make_cfg()
        held = GridLevel(index=0, buy_price=1.35, sell_price=1.3635, base_qty=10, state="sell_open")
        state = BotState(levels=[held], ladder_reserved={"ladder1": 5.0})
        trader, filters = FakeTrader(), FakeFilters()
        pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                           trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert held in state.levels
        assert held.state == "sell_open"
        assert held.buy_price == 1.35  # untouched

    def test_one_shot_levels_never_touched(self):
        cfg = make_cfg()
        bulk = GridLevel(index=0, buy_price=1.35, sell_price=1.3635, base_qty=10, state="idle", one_shot=True)
        state = BotState(levels=[bulk], ladder_reserved={"ladder1": 5.0})
        trader, filters = FakeTrader(), FakeFilters()
        pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                           trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert bulk in state.levels
        assert bulk.buy_price == 1.35

    def test_cancel_failure_leaves_level_untouched(self):
        """A level whose cancel fails (already filled for real) must be
        left completely alone -- same safety property as DASH's version."""
        cfg = make_cfg()
        stuck = GridLevel(index=0, buy_price=1.10, sell_price=1.111, base_qty=10,
                           state="buy_open", buy_order_id="order-1")  # outside the new window -> mismatched
        state = BotState(levels=[stuck], ladder_reserved={"ladder1": 5.0})
        trader = FakeTrader(fail_cancel_ids={"order-1"})
        filters = FakeFilters()
        pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                           trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert stuck in state.levels
        assert stuck.buy_price == 1.10
        assert stuck.state == "buy_open"

    def test_noop_when_window_already_fully_covered_and_reserve_empty(self):
        cfg = make_cfg()
        points = pl.build_ladder_points(1.33, 1.40, 0.01)
        existing = [GridLevel(index=i, buy_price=p, sell_price=p * 1.01, base_qty=5, state="idle")
                    for i, p in enumerate(points)]
        state = BotState(levels=list(existing), ladder_reserved={"ladder1": 0.0})
        trader, filters = FakeTrader(), FakeFilters()
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n == 0
        assert len(state.levels) == len(existing)

    def test_draw_too_thin_to_build_grows_existing_survivor_instead_of_vanishing(self):
        # Reproduces the class of bug found live on DASH/HBAR 2026-10-08: a
        # reserve draw small enough that no NEW price point can clear
        # min_notional must not be silently discarded -- with a real
        # survivor already in the window (whose own notional comfortably
        # clears min_notional, so it isn't itself flagged stuck), the draw
        # should grow that survivor instead.
        cfg = make_cfg(reserved_topup_threshold_usd=100.0)
        survivor = GridLevel(index=0, buy_price=1.35, sell_price=1.35 * 1.01, base_qty=20.0, state="idle")
        state = BotState(levels=[survivor], ladder_reserved={"ladder1": 5.0})
        trader = FakeTrader()
        filters = FakeFilters(min_notional=1.0)
        before_qty = survivor.target_qty()
        before_reserve = state.ladder_reserved["ladder1"]
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n == 0
        assert len(state.levels) == 1  # no new level fabricated
        assert survivor.target_qty() > before_qty  # the draw was applied to it, not lost
        assert state.ladder_reserved["ladder1"] < before_reserve  # something was genuinely drawn

    def test_draw_too_thin_to_build_and_no_survivor_returns_capital_to_reserve(self):
        # Same too-thin-to-build scenario but with NO existing level to
        # grow either -- the only safe outcome is giving the drawn capital
        # back to the ladder's own reserve, not discarding it.
        cfg = make_cfg(reserved_topup_threshold_usd=100.0)
        state = BotState(levels=[], ladder_reserved={"ladder1": 5.0})
        trader = FakeTrader()
        filters = FakeFilters(min_notional=1000.0)  # impossibly high -- nothing can ever clear this
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n == 0
        assert state.levels == []
        assert state.ladder_reserved["ladder1"] == pytest.approx(5.0)  # drawn, then fully returned -- not lost
        assert state.ladder_reserved_drawn.get("ladder1", 0.0) == pytest.approx(0.0)  # draw-then-return must not inflate the all-time counter

    def test_completely_dry_bypasses_retention_floor_when_only_full_reserve_is_enough(self):
        # The normal floor-capped draw (reserved_draw_floor_pct) can leave a
        # completely dry ladder permanently stuck at zero rungs even with
        # real money sitting in its own reserve, because a dry ladder earns
        # no profit of its own to ever grow past that point -- found live
        # 2026-10-12 across XRP/UNI/ZEC/HBAR (manually fixed each time by
        # injecting fresh capital). When the FULL reserve -- not just the
        # floor-capped share -- would be enough to place one real order,
        # escaping zero rungs takes priority over the normal retention floor.
        cfg = make_cfg(reserved_topup_threshold_usd=100.0, reserved_draw_floor_pct=0.5)
        filters = FakeFilters(min_notional=1.0)
        needed_min = pl._min_capital_for_one_level(lo=1.33, hi=1.40, step_pct=0.01,
                                                     reference_price=cfg.reference_price,
                                                     dca_exponent=cfg.dca_exponent,
                                                     min_notional=filters.min_notional)
        reserve = needed_min + 0.50  # 50% of this is NOT enough; 100% is
        assert reserve * 0.5 < needed_min  # sanity: confirms the scenario is real
        state = BotState(levels=[], ladder_reserved={"ladder1": reserve})
        trader = FakeTrader()
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n > 0  # at least one real rung got built, not left stuck at zero
        assert all(lvl.state == "idle" for lvl in state.levels)
        assert state.ladder_reserved.get("ladder1", 0.0) < reserve  # the bypass draw actually happened

    def test_completely_dry_still_respects_floor_when_capped_draw_is_already_enough(self):
        # In the common, healthy case -- the floor-capped draw alone already
        # clears min_notional -- the bypass must NOT fire. The retention
        # floor only gets overridden as a last resort, never routinely.
        cfg = make_cfg(reserved_topup_threshold_usd=100.0, reserved_draw_floor_pct=0.5)
        filters = FakeFilters(min_notional=1.0)
        needed_min = pl._min_capital_for_one_level(lo=1.33, hi=1.40, step_pct=0.01,
                                                     reference_price=cfg.reference_price,
                                                     dca_exponent=cfg.dca_exponent,
                                                     min_notional=filters.min_notional)
        reserve = needed_min * 3  # comfortably more than enough even at 50%
        state = BotState(levels=[], ladder_reserved={"ladder1": reserve})
        trader = FakeTrader()
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n > 0
        assert state.ladder_reserved.get("ladder1", 0.0) >= reserve * 0.5 - 0.01

    def test_completely_dry_gracefully_stays_dry_when_even_full_reserve_isnt_enough(self):
        # If even 100% of the reserve would not clear min_notional, there is
        # nothing the bootstrap override can do -- falls back to the existing
        # graceful behavior (capital returned, never lost, ladder stays dry
        # until real outside capital arrives).
        cfg = make_cfg(reserved_topup_threshold_usd=100.0, reserved_draw_floor_pct=0.5)
        filters = FakeFilters(min_notional=1.0)
        needed_min = pl._min_capital_for_one_level(lo=1.33, hi=1.40, step_pct=0.01,
                                                     reference_price=cfg.reference_price,
                                                     dca_exponent=cfg.dca_exponent,
                                                     min_notional=filters.min_notional)
        reserve = needed_min * 0.1  # nowhere near enough even at 100%
        state = BotState(levels=[], ladder_reserved={"ladder1": reserve})
        trader = FakeTrader()
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n == 0
        assert state.ladder_reserved.get("ladder1", 0.0) == pytest.approx(reserve)

    def test_mismatched_level_collected_alone_is_not_discarded(self):
        # Reproduces the bug found live 2026-10-12: cancelling a single
        # out-of-window level (no reserve draw, no future-buy pool involved)
        # whose collected capital then cannot clear min_notional and has no
        # survivor to grow was being silently discarded -- not refunded to
        # ladder_reserved, not logged, just gone from the ladder's own books
        # even though the real money was never actually lost (the order never
        # filled, so the exchange released it back to free balance on cancel).
        cfg = make_cfg(reserved_topup_threshold_usd=100.0)
        stale = GridLevel(index=0, buy_price=1.00, sell_price=1.00 * 1.01, base_qty=20.0,
                           state="buy_open", buy_order_id="abc")
        state = BotState(levels=[stale], ladder_reserved={"ladder1": 0.0})
        trader = FakeTrader()
        filters = FakeFilters(min_notional=1000.0)  # impossibly high -- nothing can ever clear this
        n = pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert n == 0
        assert state.levels == []  # the stale out-of-window level was cancelled and removed
        assert state.ladder_reserved["ladder1"] == pytest.approx(20.0)  # collected capital landed in reserve, not lost
        assert "abc" in trader.cancelled

    def test_repeated_draw_and_return_does_not_inflate_drawn_counter(self):
        # Reproduces the bug found live on HBAR 2026-10-08: a ladder stuck in
        # a draw-then-return loop (same thin-capital conditions every cycle)
        # must not accumulate a growing "all-time drawn" figure across many
        # cycles, since no real capital is ever actually lost.
        cfg = make_cfg(reserved_topup_threshold_usd=100.0)
        state = BotState(levels=[], ladder_reserved={"ladder1": 5.0})
        trader = FakeTrader()
        filters = FakeFilters(min_notional=1000.0)
        for _ in range(20):
            pl.reshape_ladder(state, "ladder1", lo=1.33, hi=1.40, step_pct=0.01, offset_pct=0.01,
                               trader=trader, filters=filters, cfg=cfg, current_price=1.40)
        assert state.ladder_reserved["ladder1"] == pytest.approx(5.0)
        assert state.ladder_reserved_drawn.get("ladder1", 0.0) == pytest.approx(0.0)


class TestMaybeReshapeLadders:
    def test_disabled_flag_is_a_full_noop(self):
        cfg = make_cfg(pct_ladder_enabled=False)
        state = BotState(levels=[], ladder_reserved={f"ladder{n}": 100.0 for n in range(1, 6)})
        trader, filters = FakeTrader(), FakeFilters()
        n = pl.maybe_reshape_ladders(state, 1.40, trader, filters, cfg)
        assert n == 0
        assert state.levels == []

    def test_enabled_builds_all_five_ladders(self):
        cfg = make_cfg()
        state = BotState(levels=[], ladder_reserved={f"ladder{n}": 10.0 for n in range(1, 6)})
        trader, filters = FakeTrader(), FakeFilters()
        pl.maybe_reshape_ladders(state, 1.40, trader, filters, cfg)
        gaps = {round(lvl.sell_price / lvl.buy_price - 1.0, 4) for lvl in state.levels}
        for pct in (0.01, 0.02, 0.03, 0.04, 0.05):
            assert round(pct, 4) in gaps


class TestMaybeFundSellFromReserve:
    def test_lists_reserve_when_ladder_has_zero_sells(self):
        cfg = make_cfg(taker_avoidance_buffer=0.005)
        state = BotState(levels=[], coin_reserve_qty={"ladder1": 10.0},
                          coin_reserve_cost_basis={"ladder1": 1.35})
        trader, filters = FakeTrader(price=1.36), FakeFilters(min_notional=1.0)
        listed = pl.maybe_fund_sell_from_reserve(state, 1.36, trader, filters, cfg)
        assert listed == 1
        assert len(state.levels) == 1
        lvl = state.levels[0]
        assert lvl.state == "sell_open"
        assert lvl.from_coin_reserve is True
        assert lvl.coin_reserve_ladder == "ladder1"
        assert lvl.buy_price == 1.35
        assert lvl.qty == 10.0
        assert state.coin_reserve_qty["ladder1"] == 0.0
        assert len(trader.placed_sells) == 1

    def test_noop_when_ladder_already_has_a_live_sell(self):
        cfg = make_cfg()
        existing = GridLevel(index=1, buy_price=1.30, sell_price=1.313, base_qty=5.0,
                              qty=5.0, state="sell_open")  # ladder1's own 1% gap
        state = BotState(levels=[existing], coin_reserve_qty={"ladder1": 10.0},
                          coin_reserve_cost_basis={"ladder1": 1.35})
        trader, filters = FakeTrader(price=1.36), FakeFilters(min_notional=1.0)
        listed = pl.maybe_fund_sell_from_reserve(state, 1.36, trader, filters, cfg)
        assert listed == 0
        assert len(state.levels) == 1  # nothing added
        assert state.coin_reserve_qty["ladder1"] == 10.0  # untouched

    def test_noop_when_reserve_empty(self):
        cfg = make_cfg()
        state = BotState(levels=[], coin_reserve_qty={"ladder1": 0.0})
        trader, filters = FakeTrader(), FakeFilters()
        listed = pl.maybe_fund_sell_from_reserve(state, 1.36, trader, filters, cfg)
        assert listed == 0
        assert state.levels == []

    def test_taker_avoidance_floor_applies_when_price_rose_since_seeding(self):
        # cost_basis=1.00, offset 1% -> natural sell price 1.01, but current
        # price has since risen to 1.20 -- selling at 1.01 would cross the
        # spread as a taker. Must price above fresh market instead.
        cfg = make_cfg(taker_avoidance_buffer=0.01)
        state = BotState(levels=[], coin_reserve_qty={"ladder1": 10.0},
                          coin_reserve_cost_basis={"ladder1": 1.00})
        trader, filters = FakeTrader(price=1.20), FakeFilters(min_notional=1.0)
        pl.maybe_fund_sell_from_reserve(state, 1.20, trader, filters, cfg)
        lvl = state.levels[0]
        assert lvl.sell_price >= 1.20 + 0.01
        assert lvl.sell_price > 1.00 * 1.01  # natural offset price would have been too low

    def test_placement_failure_leaves_reserve_untouched(self):
        cfg = make_cfg()
        state = BotState(levels=[], coin_reserve_qty={"ladder1": 10.0},
                          coin_reserve_cost_basis={"ladder1": 1.35})
        trader, filters = FakeTrader(price=1.36, fail_sell=True), FakeFilters(min_notional=1.0)
        listed = pl.maybe_fund_sell_from_reserve(state, 1.36, trader, filters, cfg)
        assert listed == 0
        assert state.levels == []
        assert state.coin_reserve_qty["ladder1"] == 10.0  # not lost

    def test_too_thin_to_clear_min_notional_leaves_reserve_untouched(self):
        cfg = make_cfg()
        state = BotState(levels=[], coin_reserve_qty={"ladder1": 0.1},
                          coin_reserve_cost_basis={"ladder1": 1.35})
        trader, filters = FakeTrader(price=1.36), FakeFilters(min_notional=1000.0)
        listed = pl.maybe_fund_sell_from_reserve(state, 1.36, trader, filters, cfg)
        assert listed == 0
        assert state.coin_reserve_qty["ladder1"] == 0.1

    def test_handles_all_five_ladders_independently(self):
        cfg = make_cfg()
        state = BotState(levels=[], coin_reserve_qty={f"ladder{n}": 10.0 for n in range(1, 6)},
                          coin_reserve_cost_basis={f"ladder{n}": 1.35 for n in range(1, 6)})
        trader, filters = FakeTrader(price=1.36), FakeFilters(min_notional=1.0)
        listed = pl.maybe_fund_sell_from_reserve(state, 1.36, trader, filters, cfg)
        assert listed == 5
        assert len(state.levels) == 5
        assert {lvl.coin_reserve_ladder for lvl in state.levels} == {f"ladder{n}" for n in range(1, 6)}


class TestReplenishCoinReserve:
    def test_rebuys_from_principal(self):
        cfg = make_cfg()
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1.0)
        refilled = pl.replenish_coin_reserve(state, "ladder2", 20.0, trader, filters, cfg)
        assert refilled is True
        assert state.coin_reserve_qty["ladder2"] > 0
        assert state.coin_reserve_cost_basis["ladder2"] == pytest.approx(1.40 * 1.002)
        assert len(trader.placed_buys) == 1

    def test_zero_or_negative_principal_is_noop(self):
        cfg = make_cfg()
        state = BotState()
        trader, filters = FakeTrader(), FakeFilters()
        assert pl.replenish_coin_reserve(state, "ladder2", 0.0, trader, filters, cfg) is False
        assert state.coin_reserve_qty == {}

    def test_too_thin_returns_false_without_buying(self):
        cfg = make_cfg()
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1000.0)
        refilled = pl.replenish_coin_reserve(state, "ladder2", 20.0, trader, filters, cfg)
        assert refilled is False
        assert trader.placed_buys == []
        assert state.coin_reserve_qty == {}

    def test_buy_failure_returns_false(self):
        cfg = make_cfg()
        state = BotState()
        trader, filters = FakeTrader(price=1.40, fail_buy=True), FakeFilters(min_notional=1.0)
        refilled = pl.replenish_coin_reserve(state, "ladder2", 20.0, trader, filters, cfg)
        assert refilled is False
        assert state.coin_reserve_qty == {}


class TestLiquidateCoinReservePosition:
    def test_profit_splits_and_principal_replenishes(self):
        cfg = make_cfg()
        state = BotState()
        lvl = GridLevel(index=1, buy_price=1.35, sell_price=1.3635, base_qty=10.0, qty=10.0,
                         state="sell_open", from_coin_reserve=True, coin_reserve_ladder="ladder1")
        profit = (1.3635 - 1.35) * 10.0  # 0.135
        trader, filters = FakeTrader(price=1.36), FakeFilters(min_notional=1.0)
        refilled = pl.liquidate_coin_reserve_position(state, lvl, profit, trader, filters, cfg)
        assert refilled is True
        assert state.ladder_reserved["ladder1"] == pytest.approx(profit * 0.5)
        assert state.total_profit_realized == pytest.approx(profit)
        assert state.coin_reserve_qty["ladder1"] > 0  # replenished
        # distribute share (profit * 0.5) queued for ladder1's own band
        assert state.pending_reinvest_by_band[pl._band_key(0.01)] == pytest.approx(profit * 0.5)

    def test_replenishment_failure_sends_principal_to_pending_reinvest(self):
        cfg = make_cfg()
        state = BotState()
        lvl = GridLevel(index=1, buy_price=1.35, sell_price=1.3635, base_qty=10.0, qty=10.0,
                         state="sell_open", from_coin_reserve=True, coin_reserve_ladder="ladder1")
        profit = (1.3635 - 1.35) * 10.0
        principal = 1.35 * 10.0
        trader, filters = FakeTrader(price=1.36, fail_buy=True), FakeFilters(min_notional=1.0)
        refilled = pl.liquidate_coin_reserve_position(state, lvl, profit, trader, filters, cfg)
        assert refilled is False
        assert state.coin_reserve_qty == {}  # nothing to replenish with
        expected_band = state.pending_reinvest_by_band[pl._band_key(0.01)]
        assert expected_band == pytest.approx(profit * 0.5 + principal)  # distribute share + unreplenished principal

    def test_ladder_identity_comes_from_the_level_not_the_gap(self):
        # Simulates the exact scenario the taker-avoidance floor can cause:
        # sell_price pushed wider than the ladder's own 1% offset would
        # naturally produce, so the gap alone would misroute this via
        # is_legacy_offset/distribute_profit_ladder_aware if used instead.
        cfg = make_cfg()
        state = BotState()
        lvl = GridLevel(index=1, buy_price=1.00, sell_price=1.20, base_qty=10.0, qty=10.0,
                         state="sell_open", from_coin_reserve=True, coin_reserve_ladder="ladder1")
        profit = (1.20 - 1.00) * 10.0
        trader, filters = FakeTrader(price=1.19), FakeFilters(min_notional=1.0)
        pl.liquidate_coin_reserve_position(state, lvl, profit, trader, filters, cfg)
        # Must land on ladder1 (the recorded owner), not be split evenly
        # across all 5 as a "legacy" position would be.
        assert state.ladder_reserved.get("ladder1", 0.0) == pytest.approx(profit * 0.5)
        assert state.ladder_reserved.get("ladder2", 0.0) == 0.0


class TestApplyCoinGrowthSplit:
    def test_zero_growth_pct_is_all_buy_side(self):
        cfg = make_cfg(coin_growth_pct=0.0)
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1.0)
        pl._apply_coin_growth_split(state, "ladder2", 10.0, trader, filters, cfg)
        assert state.pending_reinvest_by_band[pl._band_key(0.02)] == pytest.approx(10.0)
        assert state.coin_reserve_qty.get("ladder2", 0.0) == 0.0

    def test_full_growth_pct_is_all_coin(self):
        cfg = make_cfg(coin_growth_pct=1.0)
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1.0)
        pl._apply_coin_growth_split(state, "ladder2", 10.0, trader, filters, cfg)
        assert state.pending_reinvest_by_band.get(pl._band_key(0.02), 0.0) == pytest.approx(0.0)
        assert state.coin_reserve_qty["ladder2"] > 0

    def test_partial_split_goes_both_ways(self):
        cfg = make_cfg(coin_growth_pct=0.20)
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1.0)
        pl._apply_coin_growth_split(state, "ladder2", 10.0, trader, filters, cfg)
        assert state.pending_reinvest_by_band[pl._band_key(0.02)] == pytest.approx(8.0)
        assert state.coin_reserve_qty["ladder2"] * 1.4028 == pytest.approx(2.0, rel=0.01)  # ~$2 bought

    def test_growth_grows_an_already_nonzero_reserve_with_weighted_avg_cost_basis(self):
        # The exact scenario coin_growth_pct introduces that didn't exist
        # before: replenish_coin_reserve called while the reserve is
        # ALREADY nonzero, repeatedly, as small profit events trickle in.
        cfg = make_cfg(coin_growth_pct=1.0)
        state = BotState(coin_reserve_qty={"ladder2": 10.0}, coin_reserve_cost_basis={"ladder2": 1.00})
        trader, filters = FakeTrader(price=2.00), FakeFilters(min_notional=1.0)  # buy at a very different price
        pl._apply_coin_growth_split(state, "ladder2", 20.0, trader, filters, cfg)
        # qty bought ~= 20 / (2.00*1.002) ~= 9.98
        new_qty = state.coin_reserve_qty["ladder2"]
        assert new_qty > 10.0  # grew, didn't just overwrite
        # weighted average cost basis must sit between 1.00 and ~2.004, not just reset to ~2.004
        assert 1.00 < state.coin_reserve_cost_basis["ladder2"] < 2.01

    def test_failed_growth_buy_falls_back_entirely_to_buy_side(self):
        cfg = make_cfg(coin_growth_pct=1.0)
        state = BotState()
        trader, filters = FakeTrader(price=1.40, fail_buy=True), FakeFilters(min_notional=1.0)
        pl._apply_coin_growth_split(state, "ladder2", 10.0, trader, filters, cfg)
        assert state.coin_reserve_qty.get("ladder2", 0.0) == 0.0
        assert state.pending_reinvest_by_band[pl._band_key(0.02)] == pytest.approx(10.0)  # nothing lost

    def test_too_thin_growth_amount_falls_back_to_buy_side(self):
        cfg = make_cfg(coin_growth_pct=1.0)
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1000.0)
        pl._apply_coin_growth_split(state, "ladder2", 10.0, trader, filters, cfg)
        assert state.coin_reserve_qty.get("ladder2", 0.0) == 0.0
        assert state.pending_reinvest_by_band[pl._band_key(0.02)] == pytest.approx(10.0)


class TestReinvestBothDirectionsIntegration:
    def test_distribute_profit_ladder_aware_respects_coin_growth_pct(self):
        cfg = make_cfg(coin_growth_pct=0.20)
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1.0)
        pl.distribute_profit_ladder_aware(state, profit=10.0, sold_offset_pct=0.01, cfg=cfg,
                                           trader=trader, filters=filters)
        # profit 10 -> reserve 5 (unaffected), distribute 5 -> 20% coin growth (1.0), 80% buy-side (4.0)
        assert state.ladder_reserved["ladder1"] == pytest.approx(5.0)
        assert state.pending_reinvest_by_band[pl._band_key(0.01)] == pytest.approx(4.0)
        assert state.coin_reserve_qty["ladder1"] > 0

    def test_liquidate_legacy_position_principal_never_grows_coin_only_profit_does(self):
        # profit=50 so each ladder's distribute share (50*0.5/5=5.0) clears
        # min_notional=1.0 as a coin buy -- a too-small profit would fall
        # back to buy-side via the min_notional guard, masking this test.
        cfg = make_cfg(coin_growth_pct=1.0)  # 100% of PROFIT's reinvest share to coin
        state = BotState()
        trader, filters = FakeTrader(price=1.40), FakeFilters(min_notional=1.0)
        pl.liquidate_legacy_position(state, profit=50.0, principal=100.0, cfg=cfg,
                                      trader=trader, filters=filters)
        # Principal (100) must land fully in buy-side pending-reinvest, split
        # evenly, regardless of coin_growth_pct -- only profit's reinvest
        # share is eligible. With coin_growth_pct=1.0, ALL of the distribute
        # share (25) should have gone to coin instead, so pending == principal exactly.
        total_pending = sum(state.pending_reinvest_by_band.values())
        assert total_pending == pytest.approx(100.0)
        # Every ladder must have grown its coin reserve from its profit share.
        assert sum(state.coin_reserve_qty.values()) > 0
        assert all(q > 0 for q in state.coin_reserve_qty.values())

    def test_liquidate_coin_reserve_position_profit_share_can_further_grow_reserve(self):
        # buy_price/sell_price set far apart (not a realistic 1% ladder gap)
        # purely so profit's distribute share clears min_notional and
        # exercises the second buy -- a too-small profit would fall back to
        # buy-side via the min_notional guard, masking this test.
        cfg = make_cfg(coin_growth_pct=1.0)
        state = BotState()
        lvl = GridLevel(index=1, buy_price=1.00, sell_price=2.00, base_qty=10.0, qty=10.0,
                         state="sell_open", from_coin_reserve=True, coin_reserve_ladder="ladder1")
        profit = (2.00 - 1.00) * 10.0  # 10.0 -> distribute share 5.0, well above min_notional
        trader, filters = FakeTrader(price=1.99), FakeFilters(min_notional=1.0)
        pl.liquidate_coin_reserve_position(state, lvl, profit, trader, filters, cfg)
        # principal (10.0) replenishes via one buy, profit's distribute share
        # (5.0) grows it FURTHER via a second buy -- two buys total.
        assert len(trader.placed_buys) == 2
        assert state.coin_reserve_qty["ladder1"] > 0
