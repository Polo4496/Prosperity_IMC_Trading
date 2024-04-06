from datamodel import OrderDepth, UserId, TradingState, Order
import numpy as np
import pandas as pd


class Trader:
    # Memory
    assets = ['AMETHYSTS', 'STARFRUIT']
    last_prices = {
        asset: [] for asset in assets
    }
    last_fair_prices = {
        asset: [] for asset in assets
    }
    last_spreads = {
        asset: [] for asset in assets
    }
    # Parameters
    limit_position = {
        'AMETHYSTS': [-20, 20],
        'STARFRUIT': [-20, 20]
    }
    last_prices_storage = {
        'AMETHYSTS': 60,
        'STARFRUIT': 10
    }
    last_spreads_storage = {
        'AMETHYSTS': 5,
        'STARFRUIT': 5
    }
    fair_price_params = {
        'AMETHYSTS': (10000, 60, 'mean', None),
        'STARFRUIT': (None, 5, 'ewm', 10)
    }
    delta_spread_params = {
        'AMETHYSTS': 4,
        'STARFRUIT': 2
    }

    def get_fair_price(self, product):
        fair_price = None
        last_prices = self.last_prices[product]
        default, window, method, span = self.fair_price_params[product]
        if default is not None and len(last_prices) < window:
            fair_price = default
        elif method == 'mean':
            fair_price = np.mean(last_prices[-window:])
        elif method == 'ewm':
            fair_price = pd.DataFrame(last_prices[-window:]).ewm(span=span).mean().values[-1][0]

        return fair_price

    def update_last_prices(self, product, mid_price):
        if mid_price is None:
            mid_price = self.last_prices[product][-1]
        self.last_prices[product].append(mid_price)
        if len(self.last_prices[product]) > self.last_prices_storage[product]:
            self.last_prices[product].pop(0)

    def update_last_fair_prices(self, product, fair_price):
        self.last_fair_prices[product].append(fair_price)
        if len(self.last_fair_prices[product]) > self.last_prices_storage[product]:
            self.last_fair_prices[product].pop(0)

    def update_last_spreads(self, product, spread):
        if spread is None:
            spread = self.last_spreads[product][-1]
        self.last_spreads[product].append(spread)
        if len(self.last_spreads[product]) > self.last_spreads_storage[product]:
            self.last_spreads[product].pop(0)

    def compute_trend(self, product, window, fair_price=True):
        trend = 0
        last_prices = self.last_fair_prices[product] if fair_price else self.last_prices[product]
        if len(last_prices) >= window:
            trend = np.mean(np.diff(last_prices[-window:]))

        return trend

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []

            # Get Bid & Ask
            best_ask, best_bid = 0, 0
            best_ask_amount, best_bid_amount = 0, 0
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

            # Update Parameters
            mid_price = (best_ask + best_bid) / 2 if best_ask != 0 and best_bid != 0 else None
            self.update_last_prices(product, mid_price)

            spread = best_ask - best_bid if best_ask != 0 and best_bid != 0 else None
            self.update_last_spreads(product, spread)

            # Get Product Parameters
            current_position = state.position[product] if product in state.position.keys() else 0
            min_position = self.limit_position[product][0]
            max_position = self.limit_position[product][1]
            delta_spread = self.delta_spread_params[product]
            fair_price = self.get_fair_price(product)
            self.update_last_fair_prices(product, fair_price)

            shift_position, delta_ask, delta_bid = 0, 0, 0
            if product == 'AMETHYSTS':
                if current_position > 15:
                    shift_position = -1
                elif current_position < -15:
                    shift_position = 1
                continue
            elif product == 'STARFRUIT':
                trend = self.compute_trend(product, window=10, fair_price=True)
                delta_ask = -np.ceil(trend * 10) if trend < 0 else 0
                delta_bid = np.floor(trend * 10) if trend > 0 else 0

            # Market Taking Orders
            pos_buy, pos_sell = 0, 0
            if best_ask != 0 and best_ask <= fair_price - delta_ask:
                pos_buy += int(np.minimum(-best_ask_amount, max_position - current_position))
                if pos_buy != 0:
                    orders.append(Order(product, best_ask, pos_buy))
            if best_bid != 0 and best_bid >= fair_price - delta_bid:
                pos_sell += int(np.maximum(-best_bid_amount, min_position - current_position))
                if pos_sell != 0:
                    orders.append(Order(product, best_bid, pos_sell))

            # Market Making Orders
            '''if max_position - pos_buy - current_position > 0:
                pos_buy = max_position - pos_buy - current_position
                orders.append(Order(product, round(fair_price - delta_ask - delta_spread + shift_position), pos_buy))
            if min_position - pos_sell - current_position < 0:
                pos_sell = min_position - pos_sell - current_position
                orders.append(Order(product, round(fair_price + delta_bid + delta_spread + shift_position), pos_sell))'''

            result[product] = orders

        trader_data = ""
        conversions = 1

        return result, conversions, trader_data
