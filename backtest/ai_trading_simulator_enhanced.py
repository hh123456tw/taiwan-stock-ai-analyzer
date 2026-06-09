#!/usr/bin/env python3
"""
AI交易模擬器 - 增強版
增加詳細決策記錄和透明度機制
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
        """初始化AI交易模擬器"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.decision_log = []  # 新增：詳細決策記錄
        self.signal_log = []    # 新增：信號生成記錄
        
        # 交易成本設定
        self.commission_rate = 0.001425
        self.tax_rate = 0.003
        self.min_commission = 20
        
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
        SELECT prediction_date, symbol, predicted_price, actual_price,
               model_name, confidence, 
               CASE 
                   WHEN actual_price IS NOT NULL AND predicted_price IS NOT NULL THEN
                       CASE WHEN actual_price > predicted_price THEN 'UP' ELSE 'DOWN' END
                   ELSE NULL
               END as predicted_direction
        FROM predictions 
        WHERE prediction_date BETWEEN ? AND ?
        AND confidence > 0.6
        AND actual_price IS NOT NULL
        ORDER BY prediction_date, confidence DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df
        
    def generate_trading_signals(self, predictions_df: pd.DataFrame) -> List[Dict]:
        """根據AI預測生成交易信號"""
        signals = []
        
        for _, row in predictions_df.iterrows():
            # 基於預測價格與實際價格的關係生成信號
            if row['predicted_price'] and row['actual_price']:
                predicted_change = (row['predicted_price'] - row['actual_price']) / row['actual_price']
                predicted_change_pct = predicted_change * 100
                
                decision_reason = ""
                action = None
                
                # 如果預測上漲超過2%，生成買入信號
                if predicted_change > 0.02:
                    action = 'BUY'
                    decision_reason = f"AI預測上漲{predicted_change_pct:.2f}% (預測價:{row['predicted_price']:.2f} vs 實際價:{row['actual_price']:.2f}), 超過2%買入閾值, 模型:{row['model_name']}, 信心度:{row['confidence']:.2f}"
                # 如果預測下跌超過2%，生成賣出信號  
                elif predicted_change < -0.02:
                    action = 'SELL'
                    decision_reason = f"AI預測下跌{abs(predicted_change_pct):.2f}% (預測價:{row['predicted_price']:.2f} vs 實際價:{row['actual_price']:.2f}), 超過2%賣出閾值, 模型:{row['model_name']}, 信心度:{row['confidence']:.2f}"
                else:
                    # 記錄被過濾的信號
                    self.signal_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'date': row['prediction_date'],
                        'symbol': row['symbol'],
                        'action': 'FILTERED',
                        'reason': f"預測變化{predicted_change_pct:.2f}%不達2%閾值，已過濾 (模型:{row['model_name']}, 信心度:{row['confidence']:.2f})",
                        'predicted_price': row['predicted_price'],
                        'actual_price': row['actual_price'],
                        'confidence': row['confidence'],
                        'model': row['model_name'],
                        'predicted_change_pct': predicted_change_pct
                    })
                    continue  # 跳過變動太小的預測
                    
                signal = {
                    'date': row['prediction_date'],
                    'symbol': row['symbol'],
                    'action': action,
                    'confidence': row['confidence'] if row['confidence'] else 0.6,
                    'predicted_price': row['predicted_price'],
                    'actual_price': row['actual_price'],
                    'model': row['model_name'],
                    'decision_reason': decision_reason,  # 新增：決策原因
                    'predicted_change_pct': predicted_change_pct  # 新增：預測變化百分比
                }
                signals.append(signal)
                
                # 記錄信號生成日誌
                self.signal_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'date': row['prediction_date'],
                    'symbol': row['symbol'],
                    'action': action,
                    'reason': decision_reason,
                    'predicted_price': row['predicted_price'],
                    'actual_price': row['actual_price'],
                    'confidence': row['confidence'],
                    'model': row['model_name'],
                    'predicted_change_pct': predicted_change_pct
                })
                
        return signals
        
    def execute_trade(self, signal: Dict, current_price: float) -> bool:
        """執行交易"""
        symbol = signal['symbol']
        action = signal['action']
        confidence = signal['confidence']
        
        # 根據信心度決定投資金額
        position_size = min(0.05 + (confidence - 0.6) * 0.25, 0.15)
        investment_amount = self.current_capital * position_size
        
        # 記錄詳細決策過程
        decision_detail = {
            'timestamp': datetime.now().isoformat(),
            'date': signal['date'],
            'symbol': symbol,
            'action': action,
            'signal_reason': signal.get('decision_reason', ''),
            'confidence': confidence,
            'model': signal.get('model', ''),
            'position_size_pct': position_size * 100,
            'investment_amount': investment_amount,
            'current_capital': self.current_capital,
            'predicted_change': signal.get('predicted_change_pct', 0)
        }
        
        if action == 'BUY':
            shares = int(investment_amount // (current_price * 1000)) * 1000
            if shares > 0:
                total_cost = shares * current_price + self.calculate_trading_cost(shares * current_price)
                
                if total_cost <= self.current_capital:
                    decision_detail['shares'] = shares
                    decision_detail['share_price'] = current_price
                    decision_detail['total_cost'] = total_cost
                    decision_detail['execution_reason'] = f"資金充足({self.current_capital:,.0f}元 >= {total_cost:,.0f}元)，執行買入{shares}股"
                    decision_detail['execution_status'] = 'SUCCESS'
                    
                    self.current_capital -= total_cost
                    if symbol in self.positions:
                        old_shares = self.positions[symbol]['shares']
                        old_cost = self.positions[symbol]['avg_cost']
                        self.positions[symbol]['shares'] += shares
                        self.positions[symbol]['avg_cost'] = (old_cost * old_shares + shares * current_price) / (old_shares + shares)
                        decision_detail['position_action'] = f"加碼持股：原{old_shares}股@{old_cost:.2f} + 新{shares}股@{current_price:.2f}"
                    else:
                        self.positions[symbol] = {
                            'shares': shares,
                            'avg_cost': current_price,
                            'entry_date': signal['date']
                        }
                        decision_detail['position_action'] = f"新建持股：{shares}股@{current_price:.2f}"
                    
                    trade = {
                        'date': signal['date'],
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'amount': shares * current_price,
                        'cost': total_cost,
                        'confidence': confidence,
                        'model': signal.get('model', ''),
                        'decision_reason': signal.get('decision_reason', ''),
                        'position_size_pct': position_size * 100
                    }
                    self.trade_history.append(trade)
                    self.decision_log.append(decision_detail)
                    return True
                else:
                    decision_detail['execution_reason'] = f"資金不足({self.current_capital:,.0f}元 < {total_cost:,.0f}元)，取消買入"
                    decision_detail['execution_status'] = 'CANCELLED_INSUFFICIENT_FUNDS'
                    self.decision_log.append(decision_detail)
            else:
                decision_detail['execution_reason'] = f"計算股數為0({investment_amount:,.0f}元 / {current_price:.2f}元)，取消買入"
                decision_detail['execution_status'] = 'CANCELLED_ZERO_SHARES'
                self.decision_log.append(decision_detail)
                
        elif action == 'SELL' and symbol in self.positions:
            shares = self.positions[symbol]['shares']
            if shares > 0:
                revenue = shares * current_price
                trading_cost = self.calculate_trading_cost(revenue, is_buy=False)
                net_revenue = revenue - trading_cost
                profit = net_revenue - (shares * self.positions[symbol]['avg_cost'])
                
                decision_detail['shares'] = shares
                decision_detail['share_price'] = current_price
                decision_detail['revenue'] = revenue
                decision_detail['trading_cost'] = trading_cost
                decision_detail['net_revenue'] = net_revenue
                decision_detail['profit'] = profit
                decision_detail['avg_cost'] = self.positions[symbol]['avg_cost']
                decision_detail['execution_reason'] = f"賣出{shares}股@{current_price:.2f}，獲利{profit:,.0f}元"
                decision_detail['execution_status'] = 'SUCCESS'
                decision_detail['position_action'] = f"清空持股：賣出{shares}股@{current_price:.2f} (買入成本@{self.positions[symbol]['avg_cost']:.2f})"
                
                self.current_capital += net_revenue
                
                trade = {
                    'date': signal['date'],
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': current_price,
                    'amount': revenue,
                    'net_revenue': net_revenue,
                    'profit': profit,
                    'confidence': confidence,
                    'model': signal.get('model', ''),
                    'decision_reason': signal.get('decision_reason', ''),
                    'avg_cost': self.positions[symbol]['avg_cost']
                }
                self.trade_history.append(trade)
                self.decision_log.append(decision_detail)
                del self.positions[symbol]
                return True
        elif action == 'SELL' and symbol not in self.positions:
            decision_detail['execution_reason'] = f"無{symbol}持股，無法賣出"
            decision_detail['execution_status'] = 'CANCELLED_NO_POSITION'
            self.decision_log.append(decision_detail)
                
        if 'execution_status' not in decision_detail:
            decision_detail['execution_reason'] = "未知原因執行失敗"
            decision_detail['execution_status'] = 'FAILED'
            self.decision_log.append(decision_detail)
        return False
        
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """執行回測"""
        print(f"🤖 開始執行AI交易模擬回測: {start_date} 到 {end_date}")
        
        predictions_df = self.get_ai_predictions(start_date, end_date)
        if predictions_df.empty:
            print("❌ 沒有找到符合條件的預測資料")
            return {}
            
        print(f"📊 找到 {len(predictions_df)} 筆AI預測資料")
        
        signals = self.generate_trading_signals(predictions_df)
        print(f"📈 生成 {len(signals)} 個交易信號")
        print(f"🔍 過濾 {len([s for s in self.signal_log if s['action'] == 'FILTERED'])} 個不符閾值的信號")
        
        if not signals:
            print("⚠️ 沒有生成任何交易信號")
            return {}
        
        successful_trades = 0
        for signal in signals:
            if signal['actual_price'] and signal['actual_price'] > 0:
                if self.execute_trade(signal, signal['actual_price']):
                    successful_trades += 1
                    print(f"✅ 執行交易: {signal['action']} {signal['symbol']} - {signal.get('decision_reason', '')[:50]}...")
                else:
                    print(f"❌ 交易失敗: {signal['action']} {signal['symbol']}")
                    
        # 計算最終績效
        final_portfolio_value = self.current_capital
        for symbol, position in self.positions.items():
            # 使用持股成本估算
            final_portfolio_value += position['shares'] * position['avg_cost']
            
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': (final_portfolio_value - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.trade_history),
            'successful_trades': successful_trades,
            'win_rate': successful_trades / len(signals) if signals else 0,
            'remaining_positions': len(self.positions),
            'cash_remaining': self.current_capital,
            'signals_generated': len(signals),
            'signals_filtered': len([s for s in self.signal_log if s['action'] == 'FILTERED']),
            'decisions_logged': len(self.decision_log)
        }
        
        print(f"📊 回測完成: 總報酬率 {results['total_return']:.2%}, 執行 {results['total_trades']} 筆交易")
        return results
        
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """生成回測報告"""
        if not results:
            return "無回測結果可供報告"
            
        report = f"""
=== 🤖 AI交易模擬回測報告 ===
📅 測試期間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💰 績效總覽:
- 初始資金: {results['initial_capital']:,} 元
- 最終價值: {results['final_portfolio_value']:,} 元  
- 總報酬率: {results['total_return']:.2%}
- 剩餘現金: {results['cash_remaining']:,} 元

📊 交易統計:
- 生成信號: {results['signals_generated']}
- 總交易次數: {results['total_trades']}
- 成功交易: {results['successful_trades']}
- 勝率: {results['win_rate']:.2%}
- 剩餘持股: {results['remaining_positions']} 檔

🔍 信號分析統計:
- 總預測數: {len(self.signal_log)}
- 已過濾信號: {results.get('signals_filtered', 0)}
- 有效信號: {results['signals_generated']}
- 決策記錄: {results.get('decisions_logged', 0)} 筆

🎯 決策執行統計:
- 成功執行: {len([d for d in self.decision_log if d.get('execution_status') == 'SUCCESS'])}
- 資金不足取消: {len([d for d in self.decision_log if d.get('execution_status') == 'CANCELLED_INSUFFICIENT_FUNDS'])}
- 無持股取消: {len([d for d in self.decision_log if d.get('execution_status') == 'CANCELLED_NO_POSITION'])}

📈 最近交易明細:
"""
        
        for trade in self.trade_history[-5:]:
            reason = trade.get('decision_reason', '未記錄原因')[:100]
            position_size = trade.get('position_size_pct', 0)
            report += f"- {trade['date']} {trade['action']} {trade['symbol']} {trade['shares']}股 @{trade['price']:.2f}\n"
            report += f"  📝 決策原因: {reason}\n"
            report += f"  💰 投資比例: {position_size:.1f}%, 信心度: {trade['confidence']:.2f}\n"
            if trade['action'] == 'SELL' and 'profit' in trade:
                report += f"  💵 獲利: {trade['profit']:,.0f}元\n"
            report += "\n"
            
        # 新增詳細決策日誌
        report += "\n=== 📋 詳細決策記錄 ===\n"
        for i, decision in enumerate(self.decision_log[-10:], 1):
            status_emoji = "✅" if decision.get('execution_status') == 'SUCCESS' else "❌"
            report += f"{i}. {status_emoji} [{decision['date']}] {decision['symbol']} - {decision['action']}\n"
            report += f"   🔍 信號原因: {decision.get('signal_reason', 'N/A')[:80]}...\n"
            report += f"   ⚡ 執行結果: {decision.get('execution_reason', 'N/A')}\n"
            if 'profit' in decision:
                profit_emoji = "💚" if decision['profit'] > 0 else "💔"
                report += f"   {profit_emoji} 獲利: {decision['profit']:,.0f}元\n"
            report += f"   📊 狀態: {decision.get('execution_status', 'N/A')}\n\n"
            
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
        return report
    
    def get_decision_summary(self) -> Dict:
        """獲取決策摘要"""
        return {
            'total_signals': len(self.signal_log),
            'filtered_signals': len([s for s in self.signal_log if s['action'] == 'FILTERED']),
            'valid_signals': len([s for s in self.signal_log if s['action'] != 'FILTERED']),
            'successful_executions': len([d for d in self.decision_log if d.get('execution_status') == 'SUCCESS']),
            'failed_executions': len([d for d in self.decision_log if d.get('execution_status') != 'SUCCESS']),
            'latest_decisions': self.decision_log[-5:] if self.decision_log else []
        }

def main():
    """主執行函數"""
    print("🚀 啟動AI交易模擬器 - 增強版")
    simulator = AITradingSimulator(initial_capital=1000000)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    results = simulator.run_backtest(start_date, end_date)
    
    if results:
        report = simulator.generate_report(results)
        print(report)
        
        # 保存報告
        os.makedirs('/mnt/nvme/taiwan-stock-ai-analyzer-main/backtest/reports', exist_ok=True)
        report_path = f'/mnt/nvme/taiwan-stock-ai-analyzer-main/backtest/reports/ai_trading_report_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        simulator.generate_report(results, report_path)
        print(f"📁 報告已保存至: {report_path}")
        
        # 保存決策日誌
        decision_log_path = f'/mnt/nvme/taiwan-stock-ai-analyzer-main/backtest/reports/decision_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(decision_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'signal_log': simulator.signal_log,
                'decision_log': simulator.decision_log,
                'summary': simulator.get_decision_summary()
            }, f, ensure_ascii=False, indent=2)
        print(f"📋 決策日誌已保存至: {decision_log_path}")
    else:
        print("❌ 回測執行失敗或無結果")

if __name__ == "__main__":
    main()
