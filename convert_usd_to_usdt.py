"""One-time step for ZEC's build: converts a chosen amount of the
account's free USD into USDT via the USDTUSD pair, so the capital the
user deposited (as USD, not USDT) is in the right currency for ZECUSDT
before build_zec_pct_ladders.py runs. Independent of the ZEC bot's own
config -- this only ever touches USDTUSD, via a raw python-binance
Client. Reverse direction of uni-grid-bot's convert_usdt_to_usd.py.

Takes the USD amount to convert as a required argument -- always reads
the real free USD balance first and refuses to convert more than what's
actually free. USD on this account is SHARED with XRP and UNI, so this
deliberately never assumes the whole free balance belongs to this
conversion -- only convert the amount actually earmarked for ZEC.

Places a real limit BUY of USDT priced at max(1.0, best ask) on USDTUSD
(buying USDT with USD) -- USDTUSD normally trades almost exactly at 1.00,
so this should fill close to immediately; it is not a market order, so a
brief delay or partial fill is possible if the book has moved.

Plan-only by default. Pass --execute to actually place the order.

    python convert_usd_to_usdt.py 200          # dry run
    python convert_usd_to_usdt.py 200 --execute
"""
import argparse
import math
import os
from decimal import Decimal

from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

SYMBOL = "USDTUSD"


def _decimals_from_tick(tick_size: float) -> int:
    d = Decimal(str(tick_size)).normalize()
    exponent = d.as_tuple().exponent
    return max(0, -exponent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("amount_usd", type=float, help="How much free USD to convert to USDT.")
    parser.add_argument("--execute", action="store_true",
                         help="Actually place the order. Omit for a plan-only dry run.")
    args = parser.parse_args()

    if args.amount_usd <= 0:
        raise SystemExit("amount_usd must be positive.")

    api_key = os.getenv("BINANCE_US_API_KEY")
    api_secret = os.getenv("BINANCE_US_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_US_API_KEY / BINANCE_US_API_SECRET not set in .env")

    client = Client(api_key, api_secret, tld="us")

    balance = client.get_asset_balance(asset="USD")
    free_usd = float(balance["free"]) if balance else 0.0
    print(f"Free USD balance: {free_usd:.4f} (SHARED with XRP and UNI)")

    if args.amount_usd > free_usd:
        raise SystemExit(f"Requested {args.amount_usd} USD but only {free_usd:.4f} is free. "
                          f"Re-run with an amount at or below that.")

    info = client.get_symbol_info(SYMBOL)
    if info is None:
        raise SystemExit(f"{SYMBOL} not found on Binance.US.")
    tick_size = None
    step_size = None
    for f in info["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick_size = float(f["tickSize"])
        elif f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
    price_decimals = _decimals_from_tick(tick_size)
    qty_decimals = _decimals_from_tick(step_size)

    ticker = client.get_orderbook_ticker(symbol=SYMBOL)
    best_ask = float(ticker["askPrice"])
    # Crosses the spread on purpose (pay a touch above the best ask) so
    # this fills as a real, immediate buy rather than resting -- same
    # technique used for every other seed/convert script in this project.
    price = math.ceil(max(1.0, best_ask) / tick_size) * tick_size
    qty = math.floor(args.amount_usd / price / step_size) * step_size
    price_str = f"{price:.{price_decimals}f}"
    qty_str = f"{qty:.{qty_decimals}f}"

    print(f"Best ask on {SYMBOL}: {best_ask}")
    print(f"Plan: BUY {qty_str} USDT @ {price_str} on {SYMBOL} -> costs ~${qty * price:.4f} USD")

    if not args.execute:
        print("\nPLAN ONLY -- nothing changed. Re-run with --execute to apply.")
        return

    print("\nEXECUTING...")
    order = client.order_limit_buy(symbol=SYMBOL, quantity=qty_str, price=price_str)
    print(f"Placed order {order['orderId']}: BUY {qty_str} USDT @ {price_str} on {SYMBOL}")
    print("Check it filled (should be near-instant) via account_status.py or the exchange UI, "
          "then proceed to build_zec_pct_ladders.py.")


if __name__ == "__main__":
    main()
