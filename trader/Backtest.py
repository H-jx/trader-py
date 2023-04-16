from typing import List, Dict
import datetime
import pandas as pd


class Trade:
    def __init__(self, close: float, open: float, high: float, volume: float, action: str, time: pd.Timestamp, profit: float = 0.0):
        self.time = str(time)
        self.timestamp = int(time.timestamp() * 1000)
        self.close = close
        self.high = high
        self.low = min(close, open)
        self.open = open
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
        self.init_data = {
            'balance': balance,
            'position': position,
            'start_price': 0,
        }
        self.reset()
    def reset(self) -> None:
        self.current_data: Dict[str, float] = {
            'balance': self.init_data['balance'],
            'position': self.init_data['position'],
            'last_price': 120,
            'max_drawdown': 0,
            'buy_count': 0,
            'sell_count': 0,
            'profit': 0,
            'profit_rate': 0,
            'current_asset': self.init_data['balance'],
            'previous_asset': self.init_data['balance'],
        }

        self.trades = []

    def mock_trade(self, action: str, open: float, close: float, high: float, trade_volume: float, time: pd.Timestamp) -> None:
        if self.init_data['start_price'] == 0:
            self.init_data['start_price'] = close

        self.current_data['last_price'] = close

        if action:
            price = close
            cost = trade_volume * price
      
            if action == "BUY" and self.current_data['balance'] >= cost:
                self.current_data['balance'] -= (cost + self.transact_fee_rate['makerFeeRate'] * cost)
                self.current_data['position'] += trade_volume
                self.current_data['buy_count'] += 1
               
            elif action == "SELL" and self.current_data['position'] > 0.001:
                if self.current_data['position'] < trade_volume:
                    trade_volume = self.current_data['position']
                cost = trade_volume * price
                self.current_data['balance'] += (cost - self.transact_fee_rate['makerFeeRate'] * cost)
                self.current_data['position'] -= trade_volume
                self.current_data['sell_count'] += 1

            self.update_profit()

        self.trades.append(Trade(
            time=time, 
            close=close,
            high=high,
            open=open,
            volume=trade_volume, 
            action=action, 
            profit=self.get_profit()[1]
        ))
            # self.trades.append({
            #     'time': time,
            #     'timestamp': int(time.timestamp() * 1000),
            #     'close': close,
            #     'high': close,
            #     'low': close,
            #     'open': close,
            #     'volume': trade_volume,
            #     'action': action,
            #     'profit': self.get_profit()[1]
            # })
      

    def update_profit(self) -> List[float]:
        current_asset = self.current_data['balance'] + self.current_data['position'] * self.current_data['last_price']
        initial_asset = self.init_data['balance'] + self.init_data['position'] * self.init_data['start_price']
        self.current_data['previous_asset'] = self.current_data['current_asset']
        self.current_data['current_asset'] = current_asset
        self.current_data['profit'] = round((current_asset - initial_asset), 6)
        self.current_data['profit_rate'] = round((self.current_data['profit'] / initial_asset), 6)

        if self.current_data['profit_rate'] < self.current_data['max_drawdown']:
            self.current_data['max_drawdown'] = self.current_data['profit_rate']
        
        return [self.current_data['profit'], self.current_data['profit_rate']]
    
    def get_profit(self):
        return [self.current_data['profit'], self.current_data['profit_rate']]
    
    def get_results(self) -> Dict[str, float]:
        profit, profit_rate = self.get_profit()
        return {
            'max_drawdown': self.current_data['max_drawdown'],
            'profit': profit,
            'profit_rate': profit_rate,
            'buy_count': self.current_data['buy_count'],
            'sell_count': self.current_data['sell_count'],
            'trade_count': self._get_trade_count(),
            'current_asset': self.current_data['current_asset'],
            'previous_asset': self.current_data['previous_asset'],
        }

    def get_trades(self):
        return [trade.__dict__ for trade in self.trades]
    def _get_trade_count(self) -> int:
        count = 0
        if (len(self.trades) < 2):
            return count
        now = self.trades[-1].timestamp
        for trade in self.trades:
            time_diff = now - trade.timestamp
            if time_diff <= 1000 * 60 * 60 and trade.action:
                count += 1

        if count > 10:
            count = 10
        return count
    

backtest = Backtest(trade_volume = 0.1, balance= 1600, position = 0)

# mock_trade中 time 设定具体时间, 不要now   

# backtest.mock_trade(action="BUY", open=120, close=1000, high=120, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 00))
# backtest.mock_trade(action="SELL", open=60, close=60, high=120, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 5))
# backtest.mock_trade(action="", open=120, close=1200, high=1200, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 10))
# backtest.mock_trade(action="SELL", open=1200, close=1200, high=1200, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 15))
# backtest.mock_trade(action="BUY", open=1200, close=1200, high=1200, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 30))
# backtest.mock_trade(action="", open=1200, close=1200, high=1200, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 35))
# backtest.mock_trade(action="", open=1200, close=1200, high=1200, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 40))
# backtest.mock_trade(action="", open=1200, close=1200, high=1200, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 45))
# backtest.mock_trade(action="", open=1200, close=1200, high=1200, trade_volume=1, time=pd.Timestamp(2023, 4, 15, 15, 50))
# print(backtest.get_results())