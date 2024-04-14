from datamodel import OrderDepth, UserId, TradingState, Order, Symbol
import numpy as np
import pandas as pd

from statistics import NormalDist
def f(x, mean, sigma):
    dist = NormalDist(0,1)
    return dist.cdf((x-mean+1)/sigma) - dist.cdf((x-mean)/sigma)

def compute_orders_sell(symbol:Symbol,min_bid:int, volume:int, sigma=0.05)->list[Order]:
    remaining_volume = volume
    orders = []
    for i in range(0, 1000):
        bid = min_bid + i 
        amount = int(np.ceil(volume*2*f(bid, min_bid,sigma)))
        if amount==0 and remaining_volume>0:
            if remaining_volume>1:
                print((sigma,min_bid,volume, remaining_volume),"BIIIGG")
            amount=remaining_volume
        amount = np.minimum(amount,remaining_volume)
        remaining_volume-=amount
        orders.append(Order(symbol,int(bid),int(-amount)))
        if remaining_volume<0:
            print((sigma,min_bid,volume),"NEGATIVE")
        if remaining_volume==0:
            break
    if remaining_volume>0:
        print((sigma,min_bid,volume),"Remaining")
    return orders


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
        'STARFRUIT': 2.5,
        "ORCHIDS": 1
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
        order_depth = state.order_depths["ORCHIDS"]
        orders = []

        current_position = state.position["ORCHIDS"] if "ORCHIDS" in state.position.keys() else 0

        conversions = 0 
        
        print(state.own_trades)

        volume = 0
        
        for i in range(len(list(order_depth.buy_orders.items()))):
            bid, bid_amount = list(order_depth.buy_orders.items())[i]
            
            if (bid - state.observations.conversionObservations["ORCHIDS"].askPrice) - state.observations.conversionObservations["ORCHIDS"].importTariff - state.observations.conversionObservations["ORCHIDS"].transportFees > 0:
                print("TIMESTAMP: ",state.timestamp)
                orders.append(Order("ORCHIDS",bid,-bid_amount))
                volume+=bid_amount

        min_bid = int(np.ceil(state.observations.conversionObservations["ORCHIDS"].askPrice + state.observations.conversionObservations["ORCHIDS"].importTariff + state.observations.conversionObservations["ORCHIDS"].transportFees + 1.2))
        
        for order in compute_orders_sell("ORCHIDS",min_bid, 100-volume,sigma=0.7):
            # print(order.price,order.quantity, order)
            orders.append(order)
        
        if current_position <0:
            conversions= -current_position  

        result["ORCHIDS"] = orders

        trader_data = ""

        return result, conversions, trader_data
