from datamodel import OrderDepth, UserId, TradingState, Order
import numpy as np
import pandas as pd
import jsonpickle


class Trader:
    # Memory
    assets = ['AMETHYSTS', 'STARFRUIT', 'ORCHIDS']
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
    z_score_starfruit = []
    # Parameters
    limit_position = {
        'AMETHYSTS': [-20, 20],
        'STARFRUIT': [-20, 20],
        'ORCHIDS': [-100, 100]
    }
    mid_prices_storage = {
        'AMETHYSTS': 100,
        'STARFRUIT': 5,
        'ORCHIDS': 0
    }
    fair_prices_storage = {
        'AMETHYSTS': 0,
        'STARFRUIT': 0,
        'ORCHIDS': 0
    }
    avg_prices_storage = {
        'AMETHYSTS': 0,
        'STARFRUIT': 150,
        'ORCHIDS': 0
    }
    spreads_storage = {
        'AMETHYSTS': 0,
        'STARFRUIT': 0,
        'ORCHIDS': 0
    }
    fair_price_params = {
        'AMETHYSTS': (10000, 100, 'mean', None),
        'STARFRUIT': (None, 5, 'ewm', 20),
        'ORCHIDS': (None, 0, 'mean', None)
    }
    delta_spread_params = {
        'AMETHYSTS': 4,
        'STARFRUIT': 2.5,
        'ORCHIDS': 0
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
        z_score = 0.5
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

        if product == 'STARFRUIT':
            self.z_score_starfruit.append(z_score)
            if len(self.z_score_starfruit) > 100:
                self.z_score_starfruit.pop(0)
            z_score = np.mean(self.z_score_starfruit)

        return z_score

    def shift_market_making(self, current_position, position_threshold, shift):
        shift_mm = 0
        if current_position > position_threshold:
            shift_mm = -shift
        elif current_position < -position_threshold:
            shift_mm = shift
        return shift_mm

    def shift_market_taking(self, current_position, position_threshold, shift):
        shift_buy, shift_sell = 0, 0
        if current_position > position_threshold:
            shift_buy = -shift
        elif current_position < -position_threshold:
            shift_sell = shift
        return shift_buy, shift_sell

    def run(self, state: TradingState):
        # Check Class Variables
        if len(self.mid_prices[self.assets[0]]) == 0 and state.traderData != '':
            trader_data = jsonpickle.decode(state.traderData)
            self.mid_prices = trader_data['mid_prices']
            self.fair_prices = trader_data['fair_prices']
            self.avg_prices = trader_data['avg_prices']
            self.spreads = trader_data['spreads']
            self.z_score_starfruit = trader_data['z_score_starfruit']
        # Process
        result = {}
        conversions = 0
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []

            if product not in self.assets:
                continue

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
                # Shift Fair Price
                shift_buy, shift_sell = self.shift_market_taking(current_position, 0, 1)
                shift_mm = self.shift_market_making(current_position, 15, 1)
            elif product == 'STARFRUIT':
                self.update_avg_prices(product, window=5)
                z_score = self.compute_zscore(product, window=150)
                z_score = 2 * (z_score - 0.5)
                delta_buy = -z_score if z_score < -0.5 else 0
                delta_sell = z_score if z_score > 0.5 else 0
                # Shift Fair Price
                shift_buy, shift_sell = self.shift_market_taking(current_position, 10, 0.25)
                shift_mm = self.shift_market_making(current_position, 15, 0.5)
            elif product == 'ORCHIDS':
                ask_price = state.observations.conversionObservations[product].askPrice
                bid_price = state.observations.conversionObservations[product].bidPrice
                import_tariff = state.observations.conversionObservations[product].importTariff
                export_tariff = state.observations.conversionObservations[product].exportTariff
                transport_fees = state.observations.conversionObservations[product].transportFees

                if 'ORCHIDS' in state.own_trades and np.sum([trade.quantity for trade in state.own_trades[product]]) < 70:
                    self.delta_spread_params[product] = np.maximum(self.delta_spread_params[product]-0.01, -0.25)
                else:
                    self.delta_spread_params[product] = 0

                tot_volume_ask = 0
                if (best_bid - ask_price) - import_tariff - transport_fees > 0:
                    orders.append(Order(product, best_bid, -best_bid_amount))
                    tot_volume_ask = -best_bid_amount

                tot_volume_bid = 0
                if (bid_price - best_ask) - export_tariff - transport_fees - 0.1 > 0:
                    orders.append(Order(product, best_ask, best_ask_amount))
                    tot_volume_bid = best_ask_amount

                min_ask = np.ceil(ask_price + import_tariff + transport_fees + delta_spread)
                min_ask = int(np.maximum(min_ask, np.ceil(bid_price) - 1))
                orders.append(Order("ORCHIDS", min_ask, min_position - tot_volume_ask))
                min_bid = np.floor(bid_price - export_tariff - transport_fees - 0.1 - delta_spread)
                min_bid = int(np.minimum(min_bid, np.floor(ask_price) + 1))
                orders.append(Order("ORCHIDS", min_bid, max_position - tot_volume_bid))

                if current_position != 0:
                    conversions = -current_position

            # Market Orders
            if product == 'AMETHYSTS' or product == 'STARFRUIT':
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

        trader_data = jsonpickle.encode({
            'mid_prices': self.mid_prices,
            'fair_prices': self.fair_prices,
            'avg_prices': self.avg_prices,
            'spreads': self.spreads,
            'z_score_starfruit': self.z_score_starfruit
        })

        return result, conversions, trader_data
