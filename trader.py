from datamodel import OrderDepth, UserId, TradingState, Order
import numpy as np
import pandas as pd
import statistics
import jsonpickle
norm = statistics.NormalDist()
import time 


def f(sigma, S, K, r, T, C):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * np.sqrt(T)) * norm.cdf(d2) - C


def f_prime(sigma, S, K, r, T, C):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def newton_method_standard(x0, num_iterations, **kwargs):
    x = x0
    for i in range(num_iterations):
        x = x - f(x, **kwargs) / f_prime(x, **kwargs)
    return x


class Trader:
    # Memory
    assets = ['AMETHYSTS', 'STARFRUIT', 'ORCHIDS', 'COCONUT', 'COCONUT_COUPON']
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
    diff_basket = []
    diff_volatility = []
    sigma_smooth = []
    # Parameters
    limit_position = {
        'AMETHYSTS': [-20, 20],
        'STARFRUIT': [-20, 20],
        'ORCHIDS': [-100, 100],
        'GIFT_BASKET': [-60, 60],
        'COCONUT': [-300, 300],
        'COCONUT_COUPON': [-600, 600]
    }
    mid_prices_storage = {
        'AMETHYSTS': 100,
        'STARFRUIT': 5,
        'ORCHIDS': 0,
        'COCONUT': 1000,
        'COCONUT_COUPON': 1
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
        'ORCHIDS': 0,
        'COCONUT': 0,
        'COCONUT_COUPON': 0
    }
    fair_price_params = {
        'AMETHYSTS': (10000, 100, 'mean', None),
        'STARFRUIT': (None, 5, 'ewm', 20),
        'ORCHIDS': (None, 0, 'mean', None),
        'COCONUT': (None, 0, 'mean', None),
        'COCONUT_COUPON': (None, 0, 'mean', None)
    }
    delta_spread_params = {
        'AMETHYSTS': 4,
        'STARFRUIT': 2.5,
        'ORCHIDS': 0,
        'COCONUT': 0,
        'COCONUT_COUPON': 0
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
            self.diff_basket = trader_data['diff_basket']
            self.diff_volatility = trader_data['diff_volatility']
            self.sigma_smooth = trader_data['sigma_smooth']

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
            elif product == 'COCONUT' or product == 'COCONUT_COUPON':
                continue

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

        # Basket Process
        '''if 'GIFT_BASKET' in state.order_depths:
            mid_prices_single = {}
            bid_volumes_single = {}
            ask_volumes_single = {}
            best_bids_single = {}
            best_asks_single = {}
            for product in ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]:
                order_depth = state.order_depths[product]
                best_ask, best_bid = 0, 0
                best_ask_amount, best_bid_amount = 0, 0
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                mid_price = (best_ask + best_bid) / 2 if best_ask != 0 and best_bid != 0 else None
                mid_prices_single[product] = mid_price
                bid_volumes_single[product] = best_bid_amount
                ask_volumes_single[product] = best_ask_amount
                best_bids_single[product] = best_bid
                best_asks_single[product] = best_ask

            mid_price_etf = 4 * mid_prices_single['CHOCOLATE'] + 6 * mid_prices_single['STRAWBERRIES'] + mid_prices_single['ROSES']
            mid_price_basket = mid_prices_single["GIFT_BASKET"]
            best_bid_basket = best_bids_single["GIFT_BASKET"]
            best_ask_basket = best_asks_single["GIFT_BASKET"]
            bid_volume_basket = bid_volumes_single["GIFT_BASKET"]
            ask_volume_basket = ask_volumes_single["GIFT_BASKET"]

            etf_basket_diff = mid_price_etf - mid_price_basket
            self.diff_basket.append(etf_basket_diff)
            size = len(self.diff_basket)
            std = 39.53
            mu = -379.49
            if size > 60:
                if size > 400:
                    self.diff_basket.pop(0)
                std = np.std(self.diff_basket)
            signal = (etf_basket_diff - mu) / std

            product = 'GIFT_BASKET'
            current_position = state.position[product] if product in list(state.position.keys()) else 0
            min_position = self.limit_position[product][0]
            max_position = self.limit_position[product][1]

            volume = 0
            alpha_open = 1.5
            alpha_close = -0.5
            if signal > alpha_open:
                # long basket
                volume = int(np.minimum(-ask_volume_basket, max_position - current_position))
            elif signal < -alpha_open:
                # short basket
                volume = int(np.maximum(-bid_volume_basket, min_position - current_position))
            elif signal < alpha_close and current_position > 0:
                # close long position basket
                volume = int(np.maximum(-bid_volume_basket, -current_position))
            elif signal > -alpha_close and current_position < 0:
                # close short position basket
                volume = int(np.minimum(-ask_volume_basket, -current_position))

            orders = []
            if volume > 0:
                orders.append(Order(product, int(best_ask_basket), volume))
            elif volume < 0:
                orders.append(Order(product, int(best_bid_basket), volume))
            result[product] = orders'''

        # Coconut Process
        if 'COCONUT' in state.order_depths:
            mid_price = self.mid_prices['COCONUT']
            if len(mid_price) > 100:
                returns = np.diff(np.log(mid_price))
                u = 1 + np.median(returns[returns > 0])
                d = 1 + np.median(returns[returns < 0])
            else:
                u = 1.0000997810967889
                d = 0.9999002216138848

            S, K, T, delta_t = mid_price[-1], 10000, 1, 1/10000
            sigma = np.log(u / d) / (2 * np.sqrt(delta_t)) * np.sqrt(252)
            r = (1 + (np.log(np.sqrt(u * d)))) ** 252 - 1
            d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1)

            mid_price_coupon = self.mid_prices['COCONUT_COUPON']
            sigma_implied = newton_method_standard(sigma, 50, S=S, K=K, r=r, T=T, C=mid_price_coupon[-1])

            diff_sigma = (sigma_implied - sigma < 0) * 2 - 1
            self.diff_volatility.append(diff_sigma)
            if len(self.diff_volatility) > 60:
                self.diff_volatility.pop(0)
            sigma_smooth = np.round(np.mean(self.diff_volatility))
            if sigma_smooth == 0:
                sigma_smooth = self.sigma_smooth
            else:
                self.sigma_smooth = sigma_smooth

            target_position_coupon = np.maximum(np.minimum(np.round(600 * (0.5 / delta)) * sigma_smooth, 600), -600)
            target_position_coconut = -np.round((target_position_coupon * delta) * np.abs(sigma_smooth))
            current_position_coupon = state.position['COCONUT_COUPON'] if 'COCONUT_COUPON' in list(state.position.keys()) else 0
            current_position_coconut = state.position['COCONUT'] if 'COCONUT' in list(state.position.keys()) else 0

            volume = int(target_position_coupon - current_position_coupon)
            if volume != 0:
                orders = []
                order_depth = state.order_depths['COCONUT_COUPON']
                best_ask, best_bid = 0, 0
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                if volume > 0:
                    orders.append(Order('COCONUT_COUPON', int(best_ask), volume))
                else:
                    orders.append(Order('COCONUT_COUPON', int(best_bid), volume))
                result['COCONUT_COUPON'] = orders

            volume = int(target_position_coconut - current_position_coconut)
            if volume != 0:
                orders = []
                order_depth = state.order_depths['COCONUT']
                best_ask, best_bid = 0, 0
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                if volume > 0:
                    orders.append(Order('COCONUT', int(best_ask), volume))
                else:
                    orders.append(Order('COCONUT', int(best_bid), volume))
                result['COCONUT'] = orders
                
        trader_data = jsonpickle.encode({
            'mid_prices': self.mid_prices,
            'fair_prices': self.fair_prices,
            'avg_prices': self.avg_prices,
            'spreads': self.spreads,
            'z_score_starfruit': self.z_score_starfruit,
            'diff_basket': self.diff_basket,
            'diff_volatility': self.diff_volatility,
            'sigma_smooth': self.sigma_smooth
        })

        return result, conversions, trader_data