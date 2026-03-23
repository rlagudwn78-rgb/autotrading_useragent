"""
Agent 전용 거래소 클라이언트 (standalone)
ccxt 기반 통합 거래소 클라이언트 - app 의존성 없는 독립 버전
"""

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List

import ccxt.async_support as ccxt

logger = logging.getLogger("exchange_client")


class ExchangeError(Exception):
    """거래소 관련 오류"""
    pass


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str       # "LONG" or "SHORT"
    qty: Decimal
    entry_price: Decimal
    leverage: int
    unrealized_pnl: Decimal = Decimal("0")
    stop_loss: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None


class AgentExchangeClient:
    """ccxt 기반 통합 거래소 클라이언트 (Agent 전용)"""

    def __init__(self, exchange_id: str, api_key: str, api_secret: str, api_passphrase: str = ""):
        self.exchange_id = exchange_id.lower()
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase

        exchange_class = getattr(ccxt, self.exchange_id)
        exchange_config = {
            "apiKey": api_key,
            "secret": api_secret,
            "options": {
                "defaultType": "future",
                # Bybit: /v5/asset/coin/query-info 엔드포인트가 CloudFront 지역 차단됨
                # 선물 거래에 불필요한 currency 조회를 비활성화
                **({"fetchCurrencies": False} if exchange_id.lower() == "bybit" else {}),
            },
        }
        if api_passphrase:
            exchange_config["password"] = api_passphrase
        self.exchange: ccxt.Exchange = exchange_class(exchange_config)

        self._markets_loaded = False
        self._instrument_cache: Dict[str, Dict] = {}

    # 거래소별 심볼 별칭
    _EXCHANGE_SYMBOL_MAP: Dict[str, Dict[str, str]] = {
        "okx": {"XAUTUSDT": "XAUUSDT"},
        "binance": {"XAUTUSDT": "XAUUSDT"},
        "bitget": {"XAUTUSDT": "XAUUSDT"},
    }

    def _normalize_symbol(self, symbol: str) -> str:
        return self._EXCHANGE_SYMBOL_MAP.get(self.exchange_id, {}).get(symbol, symbol)

    def _to_ccxt_symbol(self, symbol: str) -> str:
        """'BTCUSDT' → 'BTC/USDT:USDT'"""
        symbol = self._normalize_symbol(symbol)
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT:USDT"
        return symbol

    def _new_exchange(self) -> ccxt.Exchange:
        """신규 ccxt 인스턴스 생성 — Bitget 단방향 SL/TP용 (캐시된 posMode 오판 방지)"""
        exchange_class = getattr(ccxt, self.exchange_id)
        config: dict = {
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "options": {"defaultType": "future"},
        }
        if self._api_passphrase:
            config["password"] = self._api_passphrase
        return exchange_class(config)

    async def _ensure_markets(self):
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    def get_instrument_info(self, symbol: str) -> Dict:
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        if self._markets_loaded:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            try:
                market = self.exchange.market(ccxt_symbol)
                precision = market.get("precision", {})
                limits = market.get("limits", {})
                amount_limits = limits.get("amount", {})
                cost_limits = limits.get("cost", {})

                qty_step_raw = precision.get("amount")
                tick_size_raw = precision.get("price")

                qty_step = Decimal(str(qty_step_raw)) if qty_step_raw else Decimal("0.001")
                tick_size = Decimal(str(tick_size_raw)) if tick_size_raw else Decimal("0.01")
                min_qty = Decimal(str(amount_limits.get("min") or "0.001"))
                min_amt = Decimal(str(cost_limits.get("min") or "5"))

                info = {
                    "qtyStep": qty_step,
                    "tickSize": tick_size,
                    "minOrderQty": min_qty,
                    "minOrderAmt": min_amt,
                }
                self._instrument_cache[symbol] = info
                return info
            except Exception as e:
                logger.warning(f"Failed to get market info for {symbol}: {e}")
        return {
            "qtyStep": Decimal("0.001"),
            "tickSize": Decimal("0.01"),
            "minOrderQty": Decimal("0.001"),
            "minOrderAmt": Decimal("5"),
        }

    def round_quantity(self, symbol: str, qty: Decimal) -> Decimal:
        info = self.get_instrument_info(symbol)
        qty_step = info["qtyStep"]
        return (qty / qty_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * qty_step

    @staticmethod
    def _safe_tick_for_price(price: Decimal) -> Decimal:
        """가격 기반 tick size 추정 (ccxt 데이터 없을 때 fallback용)"""
        if price < Decimal("0.0001"):   return Decimal("0.00000001")  # SHIB 등
        elif price < Decimal("0.01"):   return Decimal("0.000001")
        elif price < Decimal("0.1"):    return Decimal("0.00001")     # DOGE
        elif price < Decimal("10"):     return Decimal("0.0001")
        elif price < Decimal("100"):    return Decimal("0.001")
        elif price < Decimal("10000"):  return Decimal("0.01")        # ETH
        else:                           return Decimal("0.1")          # BTC

    def round_price(self, symbol: str, price: Decimal) -> Decimal:
        info = self.get_instrument_info(symbol)
        tick = info["tickSize"]
        # 안전 검사: tick이 가격의 10% 초과 → 잘못된 tick (fallback 오염)
        if tick > price * Decimal("0.1"):
            safe_tick = self._safe_tick_for_price(price)
            logger.warning(
                f"[{symbol}] tick={tick} seems wrong for price={price}, "
                f"using estimated tick={safe_tick}"
            )
            tick = safe_tick
        return (price / tick).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick

    # ===== 잔고 조회 =====

    async def get_balance(self) -> Optional[Decimal]:
        try:
            await self._ensure_markets()
            balance = await self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            free = usdt.get("free", 0) or 0
            return Decimal(str(free))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

    async def get_account_uid(self) -> Optional[str]:
        """API key에 연결된 거래소 계정 UID 조회 (레퍼럴 검증용)"""
        try:
            if self.exchange_id == "bybit":
                resp = await self.exchange.private_get_v5_user_query_api()
                return str(resp["result"]["userID"])
            elif self.exchange_id == "okx":
                resp = await self.exchange.private_get_account_config()
                return str(resp["data"][0]["uid"])
            elif self.exchange_id == "bitget":
                resp = await self.exchange.private_spot_get_v2_spot_account_info()
                return str(resp["data"]["userId"])
            elif self.exchange_id == "bingx":
                resp = await self.exchange.private_get_openapi_account_v1_uid()
                return str(resp["data"]["uid"])
            return None
        except Exception as e:
            logger.error(f"get_account_uid failed [{self.exchange_id}]: {e}")
            return None

    # ===== 현재가 조회 =====

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        try:
            await self._ensure_markets()
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            ticker = await self.exchange.fetch_ticker(ccxt_symbol)
            last = ticker.get("last")
            return Decimal(str(last)) if last else None
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None

    # ===== 포지션 조회 =====

    async def get_position(self, symbol: str) -> Optional[Position]:
        try:
            await self._ensure_markets()
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            positions = await self.exchange.fetch_positions([ccxt_symbol])
            for pos in positions:
                contracts = Decimal(str(pos.get("contracts") or 0))
                if contracts <= 0:
                    continue
                side = "LONG" if pos["side"] == "long" else "SHORT"
                sl_raw = pos.get("stopLossPrice")
                mark_raw = pos.get("markPrice")
                return Position(
                    symbol=symbol,
                    side=side,
                    qty=contracts,
                    entry_price=Decimal(str(pos["entryPrice"])),
                    leverage=int(pos.get("leverage") or 1),
                    unrealized_pnl=Decimal(str(pos.get("unrealizedPnl") or 0)),
                    stop_loss=Decimal(str(sl_raw)) if sl_raw else None,
                    mark_price=Decimal(str(mark_raw)) if mark_raw else None,
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            raise ExchangeError(f"get_position failed for {symbol}: {e}")

    async def get_all_positions(self) -> List[Dict]:
        try:
            await self._ensure_markets()
            positions = await self.exchange.fetch_positions()
            return [p for p in positions if Decimal(str(p.get("contracts") or 0)) > 0]
        except Exception as e:
            logger.error(f"Failed to get all positions: {e}")
            return []

    # ===== 포지션 모드 / 레버리지 =====

    async def switch_to_one_way_mode(self, symbol: str) -> bool:
        try:
            await self._ensure_markets()
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            if self.exchange_id == "bybit":
                await self.exchange.set_position_mode(False, ccxt_symbol)
            elif hasattr(self.exchange, "set_position_mode"):
                await self.exchange.set_position_mode(False, ccxt_symbol)
            logger.info(f"One-way mode set for {symbol}")
            return True
        except Exception as e:
            if "not modified" in str(e).lower() or "already" in str(e).lower():
                return True
            logger.error(f"Failed to switch to one-way mode: {e}")
            return False

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            await self._ensure_markets()
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            await self.exchange.set_leverage(leverage, ccxt_symbol)
            logger.info(f"Leverage set: {leverage}x for {symbol}")
            return True
        except Exception as e:
            if "110043" in str(e):
                return True
            logger.error(f"Failed to set leverage: {e}")
            return False

    # ===== 주문 실행 =====

    async def place_market_order(self, symbol: str, side: str, qty: Decimal) -> Optional[str]:
        try:
            await self._ensure_markets()
            rounded_qty = self.round_quantity(symbol, qty)
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            ccxt_side = side.lower()
            order = await self.exchange.create_order(
                ccxt_symbol, "market", ccxt_side, float(rounded_qty)
            )
            order_id = order["id"]
            logger.info(f"Market order placed: {side} {rounded_qty} {symbol} (ID: {order_id})")
            return order_id
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise ExchangeError(f"place_market_order failed for {symbol}: {e}")

    async def place_limit_order(self, symbol: str, side: str, qty: Decimal, price: Decimal) -> Optional[str]:
        try:
            await self._ensure_markets()
            rounded_qty = self.round_quantity(symbol, qty)
            rounded_price = self.round_price(symbol, price)
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            order = await self.exchange.create_order(
                ccxt_symbol, "limit", side.lower(), float(rounded_qty), float(rounded_price)
            )
            order_id = order["id"]
            logger.info(f"Limit order placed: {side} {rounded_qty} {symbol} @ {rounded_price} (ID: {order_id})")
            return order_id
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise ExchangeError(f"place_limit_order failed for {symbol}: {e}")

    async def place_tp_order(self, symbol: str, side: str, qty: Decimal, price: Decimal) -> Optional[str]:
        try:
            await self._ensure_markets()
            rounded_qty = self.round_quantity(symbol, qty)
            rounded_price = self.round_price(symbol, price)
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            if self.exchange_id == "bitget":
                # 단방향 모드: 신규 인스턴스로 holdSide 오판 방지
                ex = self._new_exchange()
                try:
                    await ex.load_markets()
                    order = await ex.create_order(
                        ccxt_symbol, "limit", side.lower(), float(rounded_qty), float(rounded_price),
                        {"reduceOnly": True, "timeInForce": "GTC"}
                    )
                finally:
                    await ex.close()
            else:
                order = await self.exchange.create_order(
                    ccxt_symbol, "limit", side.lower(), float(rounded_qty), float(rounded_price),
                    {"reduceOnly": True, "timeInForce": "GTC"}
                )
            order_id = order["id"]
            logger.info(f"TP order placed: {side} {rounded_qty} {symbol} @ {rounded_price} (ID: {order_id})")
            return order_id
        except Exception as e:
            logger.error(f"Failed to place TP order: {e}")
            raise ExchangeError(f"place_tp_order failed for {symbol}: {e}")

    async def set_stop_loss(self, symbol: str, stop_loss_price: Decimal) -> bool:
        try:
            await self._ensure_markets()
            rounded_sl = self.round_price(symbol, stop_loss_price)
            if self.exchange_id == "bybit":
                position = await self.get_position(symbol)
                if not position:
                    logger.warning(f"No position found to set SL for {symbol}")
                    return False
                if position.side == "SHORT" and rounded_sl <= position.entry_price:
                    raise ExchangeError(
                        f"set_stop_loss failed for {symbol}: SHORT 포지션의 SL({rounded_sl})은 "
                        f"진입가({position.entry_price})보다 높아야 합니다"
                    )
                if position.side == "LONG" and rounded_sl >= position.entry_price:
                    raise ExchangeError(
                        f"set_stop_loss failed for {symbol}: LONG 포지션의 SL({rounded_sl})은 "
                        f"진입가({position.entry_price})보다 낮아야 합니다"
                    )
                await self.exchange.private_post_v5_position_trading_stop({
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": str(rounded_sl),
                    "slTriggerBy": "LastPrice",
                    "positionIdx": 0,
                })
            elif self.exchange_id == "bitget":
                # 단방향 모드: place-pos-tpsl 엔드포인트 사용 (holdSide 불필요)
                await self.exchange.private_mix_post_v2_mix_order_place_pos_tpsl({
                    "symbol": symbol,
                    "productType": "USDT-FUTURES",
                    "marginMode": "crossed",
                    "marginCoin": "USDT",
                    "stopLossTriggerPrice": str(rounded_sl),
                    "stopLossTriggerType": "fill_price",
                    "stopLossExecutePrice": "0",
                })
            else:
                ccxt_symbol = self._to_ccxt_symbol(symbol)
                position = await self.get_position(symbol)
                if not position:
                    logger.warning(f"No position found to set SL for {symbol}")
                    return False
                close_side = "sell" if position.side == "LONG" else "buy"
                await self.exchange.create_order(
                    ccxt_symbol, "stop_market", close_side, float(position.qty), None,
                    {"stopPrice": float(rounded_sl), "reduceOnly": True}
                )
            logger.info(f"Stop loss set: {symbol} @ {rounded_sl}")
            return True
        except Exception as e:
            err_str = str(e)
            if "34040" in err_str or "not modified" in err_str.lower():
                logger.info(f"Stop loss already set at same price for {symbol} @ {stop_loss_price}")
                return True
            logger.error(f"Failed to set stop loss for {symbol}: {e}")
            raise ExchangeError(f"set_stop_loss failed for {symbol}: {e}")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            await self._ensure_markets()
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            await self.exchange.cancel_order(order_id, ccxt_symbol)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            if "110001" in str(e):
                return True
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ExchangeError(f"cancel_order failed for {order_id}: {e}")

    async def cancel_all_orders(self, symbol: str) -> bool:
        try:
            await self._ensure_markets()
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            await self.exchange.cancel_all_orders(ccxt_symbol)

            if self.exchange_id == "bitget":
                # ccxt cancel_all_orders는 일반 주문만 취소 — plan(stop/SL) 주문은 별도 처리
                try:
                    plan_orders = await self.exchange.fetch_open_orders(
                        ccxt_symbol, params={"isPlan": "plan"}
                    )
                    for order in plan_orders:
                        try:
                            await self.exchange.cancel_order(
                                order["id"], ccxt_symbol, params={"isPlan": "plan"}
                            )
                        except Exception as ce:
                            logger.warning(f"Failed to cancel plan order {order['id']}: {ce}")
                    if plan_orders:
                        logger.info(f"Cancelled {len(plan_orders)} plan orders for {symbol}")
                except Exception as pe:
                    logger.warning(f"Failed to fetch/cancel plan orders for {symbol}: {pe}")

            logger.info(f"All open orders cancelled for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {symbol}: {e}")
            return False

    async def close_position(self, symbol: str, side: str) -> bool:
        try:
            await self._ensure_markets()
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            close_side = "sell" if side == "LONG" else "buy"
            position = await self.get_position(symbol)
            if not position:
                logger.warning(f"No position to close for {symbol}")
                return False
            order = await self.exchange.create_order(
                ccxt_symbol, "market", close_side, float(position.qty),
                None, {"reduceOnly": True}
            )
            logger.info(f"Position closed: {side} {symbol} (ID: {order['id']})")
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            raise ExchangeError(f"close_position failed for {symbol}: {e}")

    def _to_okx_inst_id(self, symbol: str) -> str:
        """'BTCUSDT' → 'BTC-USDT-SWAP'"""
        symbol = self._normalize_symbol(symbol)
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}-USDT-SWAP"
        return symbol

    def _to_bingx_symbol(self, symbol: str) -> str:
        """'BTCUSDT' → 'BTC-USDT'"""
        symbol = self._normalize_symbol(symbol)
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}-USDT"
        return symbol

    async def get_closed_pnl(self, symbol: str, limit: int = 1) -> Optional[Dict]:
        """마지막 청산 포지션의 PnL 정보 조회 (다거래소 지원)"""
        try:
            if self.exchange_id == "bybit":
                result = await self.exchange.private_get_v5_position_closed_pnl({
                    "category": "linear",
                    "symbol": symbol,
                    "limit": limit,
                })
                rows = result.get("result", {}).get("list", [])
                return rows[0] if rows else None

            elif self.exchange_id == "okx":
                result = await self.exchange.private_get_account_positions_history({
                    "instId": self._to_okx_inst_id(symbol),
                    "limit": limit,
                })
                rows = result.get("data", [])
                if not rows:
                    return None
                row = rows[0]
                return {
                    "avgExitPrice": row.get("closeAvgPx"),
                    "closedPnl": row.get("realizedPnl"),
                    "createdTime": row.get("uTime"),
                }

            elif self.exchange_id == "bitget":
                result = await self.exchange.private_get_api_v2_mix_position_history_position({
                    "symbol": symbol,
                    "productType": "USDT-FUTURES",
                    "limit": limit,
                })
                rows = result.get("data", {}).get("list", [])
                if not rows:
                    return None
                row = rows[0]
                return {
                    "avgExitPrice": row.get("closeAvgPrice"),
                    "closedPnl": row.get("totalPnl"),
                    "createdTime": row.get("utime"),
                }

            elif self.exchange_id == "bingx":
                result = await self.exchange.swap_v1_private_get_trade_position_history({
                    "symbol": self._to_bingx_symbol(symbol),
                    "pageSize": limit,
                })
                rows = result.get("data", {}).get("positionHistory", [])
                if not rows:
                    return None
                row = rows[0]
                return {
                    "avgExitPrice": row.get("avgClosePrice"),
                    "closedPnl": row.get("realisedProfit"),
                    "createdTime": row.get("updateTime"),
                }

            return None
        except Exception as e:
            logger.warning(f"get_closed_pnl failed for {symbol}: {e}")
            return None

    async def get_recent_fills(self, symbol: str, limit: int = 5) -> List[Dict]:
        """최근 체결 내역 조회 (실제 체결가 확인용)"""
        try:
            if self.exchange_id == "bybit":
                result = await self.exchange.private_get_v5_execution_list({
                    "category": "linear",
                    "symbol": symbol,
                    "limit": limit,
                })
                return result.get("result", {}).get("list", [])
            else:
                ccxt_symbol = self._to_ccxt_symbol(symbol)
                trades = await self.exchange.fetch_my_trades(ccxt_symbol, limit=limit)
                return [
                    {
                        "execPrice": str(t["price"]),
                        "execQty": str(t["amount"]),
                        "execTime": str(int(t["timestamp"])),
                        # ccxt "limit"/"market" → capitalize → "Limit"/"Market" (Bybit raw 포맷과 통일)
                        "orderType": (t.get("type") or "").capitalize(),
                        "execFee": str(t.get("fee", {}).get("cost") or 0),
                    }
                    for t in trades
                ]
        except Exception as e:
            logger.warning(f"get_recent_fills failed for {symbol}: {e}")
            return []

    async def close(self):
        try:
            await self.exchange.close()
        except Exception:
            pass
        finally:
            try:
                self.exchange.apiKey = ""
                self.exchange.secret = ""
                self.exchange.password = ""
            except Exception:
                pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
