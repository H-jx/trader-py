from typing import List, Dict
import datetime
from enum import Enum

class Signal(Enum):
    BUY = 'BUY'
    SELL = 'SELL'

class Trade:
    def __init__(self, close: float, volume: float, action: str, timestamp: int = 0, time: str = '', profit: float = 0.0):
        self.timestamp = timestamp
        self.time = time
        self.close = close
        self.volume = volume
        self.action = action
        self.profit = profit

class Backtest:
    def __init__(self, trade_volume: float, balance: float, position: float):
        self.transact_fee_rate = {
            "makerFeeRate": 0.00044,
            "takerFeeRate": 0.00044,
        }
        self.trade_volume = trade_volume
        self.trades: List[Trade] = []
        self.current_data = {
            'balance': balance,
            'position': position,
            'last_price': 0,
            'max_drawdown': 0,
            'buy_count': 0,
            'sell_count': 0,
        }
        self.init_data = {
            'balance': balance,
            'position': position,
            'start_price': 0,
        }
    def reset(self) -> None:
        self.current_data = {
            'balance': self.init_data['balance'],
            'position': self.init_data['position'],
            'last_price': 0,
            'max_drawdown': 0,
            'buy_count': 0,
            'sell_count': 0,
        }
        self.trades = []

    def mock_trade(self, action: str, close: float, trade_volume: float, timestamp: int) -> None:
        if self.init_data['start_price'] == 0:
            self.init_data['start_price'] = close

        self.current_data['last_price'] = close

        if action:
            price = close
            trade_volume = self.trade_volume
            cost = trade_volume * price
            if action == Signal.BUY and self.current_data['balance'] >= cost:
                self.current_data['balance'] -= cost + self.transact_fee_rate['makerFeeRate'] * cost
                self.current_data['position'] += trade_volume
                self.current_data['buy_count'] += 1
 
            elif action == Signal.SELL:
                trade_volume = trade_volume if self.current_data['position'] >= trade_volume else self.current_data['position']
                self.current_data['balance'] += cost - self.transact_fee_rate['makerFeeRate'] * cost
                self.current_data['position'] -= trade_volume
                self.current_data['sell_count'] += 1

            self.update_profit()
            self.trades.append(Trade(
                time=datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'), 
                close=price, 
                volume=trade_volume, 
                action=action, 
                profit=self.get_profit()[1]
            ))
      

    def update_profit(self) -> List[float]:
        current_asset = self.current_data['balance'] + self.current_data['position'] * self.current_data['last_price']
        initial_asset = self.init_data['balance'] + self.init_data['position'] * self.init_data['start_price']

        profit = round((current_asset - initial_asset), 4)
        profit_rate = round((profit / initial_asset) * 100, 4)

        if profit_rate < self.current_data['max_drawdown']:
            self.current_data['max_drawdown'] = profit_rate
        
        return [profit, profit_rate]
    
    def get_profit(self):
        profit, profit_rate = self.update_profit()
        return [profit, profit_rate]
    
    def get_results(self):
        profit, profit_rate = self.get_profit()

        return {
            'max_drawdown': self.current_data['max_drawdown'],
            'profit': profit,
            'profit_rate': profit_rate,
            'buy_count': self.current_data['buy_count'],
            'sell_count': self.current_data['sell_count'],
            'trade_count': self._get_trade_count(),
        }

    def get_trades(self):
        return self.trades
    def _get_trade_count(self) -> int:
        count = 0
        if (len(self.trades) < 2):
            return count
        now = self.trades[-1].timestamp
        for trade in self.trades:
            time_diff = now - trade.timestamp
            if time_diff <= 1000 * 60 * 30:
                count += 1
        return count