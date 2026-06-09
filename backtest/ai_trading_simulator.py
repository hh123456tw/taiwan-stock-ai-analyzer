#\!/usr/bin/env python3
"""
AI交易模擬器 - 整合AI預測與模擬下單系統
基於現有AI模型預測結果進行交易模擬與績效評估
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple

class AITradingSimulator:
    def __init__(self, initial_capital=1000000):
        """
        初始化AI交易模擬器
        
        Args:
            initial_capital: 初始資金 (預設100萬)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # 持股部位
        self.trade_history = []  # 交易紀錄
        self.daily_portfolio = []  # 每日投資組合價值
        
        # 交易成本設定
        self.commission_rate = 0.001425  # 手續費 0.1425%
        self.tax_rate = 0.003  # 證交稅 0.3% (僅賣出)
        self.min_commission = 20  # 最低手續費20元
        
        # 資料庫路徑
        self.db_path = '/mnt/nvme/taiwan-stock-ai-analyzer-main/predictions.db'
        
    def calculate_trading_cost(self, amount: float, is_buy: bool = True) -> float:
        """計算交易成本"""
        commission = max(amount * self.commission_rate, self.min_commission)
        tax = amount * self.tax_rate if not is_buy else 0
        return commission + tax
        
    def get_ai_predictions(self, start_date: str, end_date: str) -> pd.DataFrame:
        """從資料庫獲取AI預測資料"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT prediction_date, symbol, predicted_direction, confidence_score,
               predicted_price, actual_price, model_type
        FROM predictions 
        WHERE prediction_date BETWEEN ? AND ?
        AND confidence_score > 0.6
        ORDER BY prediction_date, confidence_score DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df
        
    def generate_trading_signals(self, predictions_df: pd.DataFrame) -> List[Dict]:
        """根據AI預測生成交易信號"""
        signals = []
        
        for _, row in predictions_df.iterrows():
            signal = {
                'date': row['prediction_date'],
                'symbol': row['symbol'],
                'action': 'BUY' if row['predicted_direction'] == 'UP' else 'SELL',
                'confidence': row['confidence_score'],
                'predicted_price': row['predicted_price'],
                'actual_price': row['actual_price'],
                'model': row['model_type']
            }
            signals.append(signal)
            
        return signals
        
    def execute_trade(self, signal: Dict, current_price: float) -> bool:
        """執行交易"""
        symbol = signal['symbol']
        action = signal['action']
        confidence = signal['confidence']
        
        # 根據信心度決定投資金額 (5%-20%資金)
        position_size = min(0.05 + (confidence - 0.6) * 0.375, 0.2)
        investment_amount = self.current_capital * position_size
        
        if action == 'BUY':
            # 買入邏輯
            shares = int(investment_amount // (current_price * 1000)) * 1000  # 整張買入
            if shares > 0:
                total_cost = shares * current_price + self.calculate_trading_cost(shares * current_price)
                
                if total_cost <= self.current_capital:
                    self.current_capital -= total_cost
                    if symbol in self.positions:
                        self.positions[symbol]['shares'] += shares
                        self.positions[symbol]['avg_cost'] = (
                            self.positions[symbol]['avg_cost'] * self.positions[symbol]['shares'] + 
                            shares * current_price
                        ) / (self.positions[symbol]['shares'] + shares)
                    else:
                        self.positions[symbol] = {
                            'shares': shares,
                            'avg_cost': current_price,
                            'entry_date': signal['date']
                        }
                    
                    # 記錄交易
                    trade = {
                        'date': signal['date'],
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'amount': shares * current_price,
                        'cost': total_cost,
                        'confidence': confidence,
                        'model': signal['model']
                    }
                    self.trade_history.append(trade)
                    return True
                    
        elif action == 'SELL' and symbol in self.positions:
            # 賣出邏輯
            shares = self.positions[symbol]['shares']
            if shares > 0:
                revenue = shares * current_price
                trading_cost = self.calculate_trading_cost(revenue, is_buy=False)
                net_revenue = revenue - trading_cost
                
                self.current_capital += net_revenue
                
                # 記錄交易
                trade = {
                    'date': signal['date'],
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': current_price,
                    'amount': revenue,
                    'net_revenue': net_revenue,
                    'profit': net_revenue - (shares * self.positions[symbol]['avg_cost']),
                    'confidence': confidence,
                    'model': signal['model']
                }
                self.trade_history.append(trade)
                
                # 清除持股
                del self.positions[symbol]
                return True
                
        return False
        
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """執行回測"""
        print(f"開始執行AI交易模擬回測: {start_date} 到 {end_date}")
        
        # 獲取AI預測資料
        predictions_df = self.get_ai_predictions(start_date, end_date)
        if predictions_df.empty:
            print("沒有找到預測資料")
            return {}
            
        print(f"找到 {len(predictions_df)} 筆AI預測資料")
        
        # 生成交易信號
        signals = self.generate_trading_signals(predictions_df)
        
        # 執行交易
        successful_trades = 0
        for signal in signals:
            if signal['actual_price'] and signal['actual_price'] > 0:
                if self.execute_trade(signal, signal['actual_price']):
                    successful_trades += 1
                    
        # 計算最終績效
        final_portfolio_value = self.current_capital
        for symbol, position in self.positions.items():
            # 用最後價格估算持股價值
            final_portfolio_value += position['shares'] * position['avg_cost']  # 簡化處理
            
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': (final_portfolio_value - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.trade_history),
            'successful_trades': successful_trades,
            'win_rate': successful_trades / len(signals) if signals else 0,
            'remaining_positions': len(self.positions),
            'cash_remaining': self.current_capital
        }
        
        return results
        
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """生成回測報告"""
        report = f"""
=== AI交易模擬回測報告 ===
測試期間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

績效總覽:
- 初始資金: {results['initial_capital']:,} 元
- 最終價值: {results['final_portfolio_value']:,} 元  
- 總報酬率: {results['total_return']:.2%}
- 剩餘現金: {results['cash_remaining']:,} 元

交易統計:
- 總交易次數: {results['total_trades']}
- 成功交易: {results['successful_trades']}
- 勝率: {results['win_rate']:.2%}
- 剩餘持股: {results['remaining_positions']} 檔

交易明細:
"""
        
        # 添加交易明細
        for trade in self.trade_history[-10:]:  # 顯示最後10筆交易
            report += f"- {trade['date']} {trade['action']} {trade['symbol']} {trade['shares']}股 @{trade['price']:.2f} (信心:{trade['confidence']:.1%})\n"
            
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
        return report

def main():
    """主執行函數"""
    simulator = AITradingSimulator(initial_capital=1000000)
    
    # 設定回測期間 (最近30天)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 執行回測
    results = simulator.run_backtest(start_date, end_date)
    
    if results:
        # 生成報告
        report = simulator.generate_report(results)
        print(report)
        
        # 保存報告
        os.makedirs('/mnt/nvme/taiwan-stock-ai-analyzer-main/backtest/reports', exist_ok=True)
        report_path = f'/mnt/nvme/taiwan-stock-ai-analyzer-main/backtest/reports/ai_trading_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        simulator.generate_report(results, report_path)
        print(f"\n報告已保存至: {report_path}")
    else:
        print("回測執行失敗")

if __name__ == "__main__":
    main()
