from datamodel import OrderDepth, UserId, TradingState, Order
import numpy as np
import pandas as pd


class Trader:
    # Memory
    assets = ['AMETHYSTS', 'STARFRUIT']
    mid_prices = {
        asset: [] for asset in assets
    }
    fair_prices = {
        asset: [] for asset in assets
    }
    avg_prices = {
        asset: [] for asset in assets
    }
    spreads = {
        asset: [] for asset in assets
    }
    # Parameters
    limit_position = {
        'AMETHYSTS': [-20, 20],
        'STARFRUIT': [-20, 20]
    }
    mid_prices_storage = {
        'AMETHYSTS': 100,
        'STARFRUIT': 5
    }
    fair_prices_storage = {
        'AMETHYSTS': 0,
        'STARFRUIT': 0
    }
    avg_prices_storage = {
        'AMETHYSTS': 0,
        'STARFRUIT': 150
    }
    spreads_storage = {
        'AMETHYSTS': 0,
        'STARFRUIT': 0
    }

    fair_price_params = {
        'AMETHYSTS': (10000, 100, 'mean', None),
        'STARFRUIT': (None, 5, 'ewm', 20)
    }
    delta_spread_params = {
        'AMETHYSTS': 4,
        'STARFRUIT': 2.5
    }

    def get_fair_price(self, product):
        fair_price = None
        last_prices = self.mid_prices[product]
        default, window, method, span = self.fair_price_params[product]
        if default is not None and len(last_prices) < window:
            fair_price = default
        elif method == 'mean':
            fair_price = np.mean(last_prices[-window:])
        elif method == 'ewm':
            fair_price = pd.DataFrame(last_prices[-window:]).ewm(span=span).mean().values[-1][0]

        return fair_price

    def update_mid_prices(self, product, mid_price):
        if mid_price is None:
            mid_price = self.mid_prices[product][-1]
        self.mid_prices[product].append(mid_price)
        if len(self.mid_prices[product]) > self.mid_prices_storage[product]:
            self.mid_prices[product].pop(0)

    def update_fair_prices(self, product, fair_price):
        self.fair_prices[product].append(fair_price)
        if len(self.fair_prices[product]) > self.fair_prices_storage[product]:
            self.fair_prices[product].pop(0)

    def update_avg_prices(self, product, window):
        if len(self.mid_prices[product]) >= window:
            avg_price = np.mean(self.mid_prices[product][-window:])
        else:
            avg_price = np.mean(self.mid_prices[product])
        self.avg_prices[product].append(avg_price)
        if len(self.avg_prices[product]) > self.avg_prices_storage[product]:
            self.avg_prices[product].pop(0)

    def update_spreads(self, product, spread):
        if spread is None:
            spread = self.spreads[product][-1]
        self.spreads[product].append(spread)
        if len(self.spreads[product]) > self.spreads_storage[product]:
            self.spreads[product].pop(0)

    def compute_zscore(self, product, window, mode='avg_price'):
        z_score = 0
        if mode == 'avg_price':
            last_prices = self.avg_prices[product]
        elif mode == 'fair_price':
            last_prices = self.fair_prices[product]
        elif mode == 'mid_price':
            last_prices = self.mid_prices[product]
        else:
            raise Exception('Mode not defined.')

        if len(last_prices) >= window:
            last_prices = last_prices[-window:]
            z_score = (last_prices[-1] - np.min(last_prices)) / (np.max(last_prices) - np.min(last_prices))

        return z_score

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
            self.update_mid_prices(product, mid_price)

            spread = best_ask - best_bid if best_ask != 0 and best_bid != 0 else None
            self.update_spreads(product, spread)

            # Get Product Parameters
            current_position = state.position[product] if product in state.position.keys() else 0
            min_position = self.limit_position[product][0]
            max_position = self.limit_position[product][1]
            delta_spread = self.delta_spread_params[product]
            fair_price = self.get_fair_price(product)
            delta_buy, delta_sell = 0, 0
            shift_mm, shift_buy, shift_sell = 0, 0, 0

            if product == 'AMETHYSTS':
                if current_position > 0:
                    shift_buy = -1
                elif current_position < 0:
                    shift_sell = 1

                if current_position > 15:
                    shift_mm = -1
                elif current_position < -15:
                    shift_mm = 1

            elif product == 'STARFRUIT':
                self.update_avg_prices(product, window=5)
                z_score = self.compute_zscore(product, window=150)
                z_score = 2 * (z_score - 0.5)
                delta_buy = -z_score if z_score < -0.5 else 0
                delta_sell = z_score if z_score > 0.5 else 0

                if current_position > 10:
                    shift_buy = -0.25
                elif current_position < -10:
                    shift_sell = 0.25

                if current_position > 15:
                    shift_mm = -0.5
                elif current_position < -15:
                    shift_mm = 0.5

            # Market Taking Orders
            pos_buy, pos_sell = 0, 0
            if best_ask != 0 and best_ask <= fair_price - delta_buy + shift_buy:
                pos_buy += int(np.minimum(-best_ask_amount, max_position - current_position))
                if pos_buy != 0:
                    orders.append(Order(product, best_ask, pos_buy))
            if best_bid != 0 and best_bid >= fair_price + delta_sell + shift_sell:
                pos_sell += int(np.maximum(-best_bid_amount, min_position - current_position))
                if pos_sell != 0:
                    orders.append(Order(product, best_bid, pos_sell))

            # Market Making Orders
            if max_position - pos_buy - current_position > 0:
                pos_buy = max_position - pos_buy - current_position
                orders.append(Order(product, round(fair_price - delta_spread + shift_mm), pos_buy))
            if min_position - pos_sell - current_position < 0:
                pos_sell = min_position - pos_sell - current_position
                orders.append(Order(product, round(fair_price + delta_spread + shift_mm), pos_sell))

            result[product] = orders

        trader_data = ""
        conversions = 1

        return result, conversions, trader_data
