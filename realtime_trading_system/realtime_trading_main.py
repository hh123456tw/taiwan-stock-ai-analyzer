#!/usr/bin/env python3
"""
台股AI實時自動交易系統 - 主執行程式
實現9:00-13:30自動模擬交易，完整記錄優化AI
"""

import sys
import os
import asyncio
import logging
import json
import sqlite3
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
import signal

# 添加項目路徑
SYSTEM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SYSTEM_ROOT)
sys.path.insert(0, os.path.join(SYSTEM_ROOT, 'realtime_trading_system'))

from realtime_trading_system.config.trading_config import TradingConfig
from realtime_trading_system.core.market_scanner_fixed import RealTimeMarketScanner
from realtime_trading_system.core.ai_decision_engine import AIDecisionEngine
from realtime_trading_system.telegram_notifier import notify_trade

class RealTimeTradingSystem:
    """實時自動交易系統主控制器"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.logger = self._setup_logger()
        self.is_running = False
        self.is_trading_hours = False
        
        # 核心組件
        self.market_scanner = None
        self.ai_engine = None
        
        # 交易狀態
        self.current_portfolio = {'cash': 400000, 'positions': {}, 'daily_pnl': 0}
        self.trading_log = []
        self.daily_stats = {'trades': 0, 'decisions': 0, 'pnl': 0}
        
        # 設置信號處理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_logger(self) -> logging.Logger:
        """設置主系統日誌"""
        logger = logging.getLogger('RealTimeTradingSystem')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            return logger
            
        # 確保日誌目錄存在
        log_dir = os.path.join(SYSTEM_ROOT, 'realtime_trading_system', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 文件日誌
        log_file = os.path.join(log_dir, f'trading_system_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台日誌
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式設置
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _signal_handler(self, signum, frame):
        """處理系統信號"""
        self.logger.info(f"收到信號 {signum}，準備優雅關閉系統...")
        self.is_running = False

    def _is_trading_time(self) -> bool:
        """檢查是否在交易時間內 (9:00-13:30)"""
        now = datetime.now().time()
        market_open = time(9, 0)   # 9:00
        market_close = time(13, 30)  # 13:30
        
        # 檢查是否為工作日
        weekday = datetime.now().weekday()
        if weekday >= 5:  # 週六(5) 週日(6)
            return False
            
        return market_open <= now <= market_close

    async def _initialize_components(self):
        """初始化系統組件"""
        try:
            self.logger.info("🚀 初始化實時交易系統組件...")
            
            # 初始化市場掃描器
            self.market_scanner = RealTimeMarketScanner()
            self.logger.info("✅ 市場掃描器初始化完成")
            
            # 初始化AI決策引擎
            self.ai_engine = AIDecisionEngine()
            self.logger.info("✅ AI決策引擎初始化完成")
            
            # 載入當前持倉狀況
            await self._load_portfolio_state()
            self.logger.info("✅ 持倉狀況載入完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 組件初始化失敗: {e}")
            return False

    async def _load_portfolio_state(self):
        """載入當前投資組合狀態"""
        try:
            # 從數據庫或文件載入持倉狀況
            portfolio_file = os.path.join(SYSTEM_ROOT, 'realtime_trading_system', 'data', 'portfolio.json')
            
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r', encoding='utf-8') as f:
                    self.current_portfolio = json.load(f)
                
                # 確保舊格式兼容性 - 為沒有purchase_dates的持股添加欄位
                for stock_code, position in self.current_portfolio['positions'].items():
                    if 'purchase_dates' not in position:
                        # 假設舊持股都是昨天或更早買入的，可以立即賣出
                        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
                        position['purchase_dates'] = {yesterday: position['quantity']}
                    else:
                        # 將日期字符串轉換為date對象供內部使用
                        new_dates = {}
                        for date_str, qty in position['purchase_dates'].items():
                            if isinstance(date_str, str):
                                new_dates[datetime.fromisoformat(date_str).date()] = qty
                            else:
                                new_dates[date_str] = qty
                        position['purchase_dates'] = new_dates
                
                self.logger.info(f"📊 載入投資組合: 現金 ${self.current_portfolio['cash']:,.0f}")
            else:
                # 初始化空投資組合
                self.current_portfolio = {
                    'cash': 400000,
                    'positions': {},
                    'daily_pnl': 0,
                    'total_pnl': 0
                }
                await self._save_portfolio_state()
                
        except Exception as e:
            self.logger.error(f"載入投資組合失敗: {e}")

    async def _save_portfolio_state(self):
        """保存投資組合狀態"""
        try:
            portfolio_file = os.path.join(SYSTEM_ROOT, 'realtime_trading_system', 'data', 'portfolio.json')
            os.makedirs(os.path.dirname(portfolio_file), exist_ok=True)
            
            # 創建可序列化的副本
            portfolio_copy = self.current_portfolio.copy()
            for stock_code, position in portfolio_copy['positions'].items():
                if 'purchase_dates' in position:
                    # 將date對象轉換為字符串
                    date_dict = {}
                    for date_obj, qty in position['purchase_dates'].items():
                        if hasattr(date_obj, 'isoformat'):
                            date_dict[date_obj.isoformat()] = qty
                        else:
                            date_dict[str(date_obj)] = qty
                    position['purchase_dates'] = date_dict
            
            with open(portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio_copy, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存投資組合失敗: {e}")

    async def _scan_and_decide(self):
        """掃描市場並做出交易決策"""
        try:
            self.logger.info("🔍 開始市場掃描...")
            
            # 1. 掃描市場機會
            candidates, market_overview = await self.market_scanner.scan_market()
            
            if not candidates:
                self.logger.info("📊 未發現符合條件的投資機會")
                return
                
            self.logger.info(f"📊 發現 {len(candidates)} 個潛在交易機會")
            
            # 2. AI決策分析
            # 確保portfolio有available_capital欄位
            portfolio_for_ai = self.current_portfolio.copy()
            portfolio_for_ai["available_capital"] = portfolio_for_ai.get("cash", 0)

            # 使用增強版AI決策引擎 (整合技術指標和市場情緒)
            decisions = self.ai_engine.make_enhanced_trading_decisions(candidates, portfolio_for_ai)
            
            if not decisions:
                self.logger.info("🤖 增強版AI決策引擎未產生交易信號")
                return
                
            # 3. 執行交易決策
            for decision in decisions:
                await self._execute_trade_decision(decision)
                
            # 4. 更新每日統計
            self.daily_stats['decisions'] += len(decisions)
            
            # 5. 記錄完整決策過程
            await self._log_trading_session({
                'timestamp': datetime.now().isoformat(),
                'candidates_found': len(candidates),
                'decisions_made': len(decisions),
                'portfolio_state': self.current_portfolio.copy(),
                'daily_stats': self.daily_stats.copy()
            })
            
        except Exception as e:
            self.logger.error(f"❌ 掃描決策過程出錯: {e}")

    async def _execute_trade_decision(self, decision: Dict):
        """執行交易決策"""
        try:
            stock_code = decision.get('stock_code', decision.get('symbol'))
            action = decision['action']  # BUY/SELL
            quantity = decision.get('quantity', decision.get('shares', 0))
            price = decision.get('price', decision.get('target_price', 0))
            confidence = decision['confidence']
            reasoning = decision.get('reasoning', decision.get('approval_reason', ''))
            
            # 增強版決策的額外資訊
            technical_score = decision.get('technical_score', 0.5)
            sentiment_score = decision.get('sentiment_score', 0.5)
            original_confidence = decision.get('original_confidence', confidence)
            enhancement_version = decision.get('enhancement_version', '1.0')
            
            # 計算交易金額
            trade_amount = quantity * price
            today = datetime.now().date()
            
            # 執行模擬交易
            if action == 'BUY':
                if self.current_portfolio['cash'] >= trade_amount:
                    # 買入
                    self.current_portfolio['cash'] -= trade_amount
                    if stock_code not in self.current_portfolio['positions']:
                        self.current_portfolio['positions'][stock_code] = {
                            'quantity': 0, 
                            'avg_price': 0,
                            'purchase_dates': {}  # 追蹤購買日期和數量 {date: quantity}
                        }
                    
                    # 更新持股
                    pos = self.current_portfolio['positions'][stock_code]
                    total_cost = pos['quantity'] * pos['avg_price'] + trade_amount
                    total_quantity = pos['quantity'] + quantity
                    pos['avg_price'] = total_cost / total_quantity if total_quantity > 0 else price
                    pos['quantity'] = total_quantity
                    
                    # 記錄購買日期和數量 (T+1規則)
                    if today not in pos['purchase_dates']:
                        pos['purchase_dates'][today] = 0
                    pos['purchase_dates'][today] += quantity
                    
                    self.daily_stats['trades'] += 1
                    
                    if enhancement_version == '2.0':
                        self.logger.info(
                            f"✅ 增強版買入: {stock_code} {quantity}股 @{price:.2f} "
                            f"信心度:{original_confidence:.3f}→{confidence:.3f} "
                            f"(技術:{technical_score:.3f}, 情緒:{sentiment_score:.3f}) - {reasoning[:80]}..."
                        )
                    else:
                        self.logger.info(
                            f"✅ 買入成功: {stock_code} {quantity}股 @{price:.2f} "
                            f"(信心度:{confidence:.3f}) - {reasoning[:100]}..."
                        )
                    
                    # 發送TG通知
                    try:
                        trade_data = {
                            'action': 'BUY',
                            'symbol': stock_code,
                            'company_name': stock_code,  # 可以改為完整公司名稱
                            'price': price,
                            'shares': quantity,
                            'investment_amount': trade_amount,
                            'confidence': confidence,
                            'decision_reason': reasoning[:200],
                            'approval_reason': reasoning,
                            'enhancement_version': enhancement_version,
                            'technical_score': technical_score,
                            'sentiment_score': sentiment_score,
                            'original_confidence': original_confidence
                        }
                        notify_trade(trade_data)
                    except Exception as e:
                        self.logger.warning(f"⚠️ TG通知發送失敗: {e}")
                else:
                    self.logger.warning(f"⚠️ 資金不足，無法買入 {stock_code}")
                    
            elif action == 'SELL':
                if stock_code in self.current_portfolio['positions']:
                    pos = self.current_portfolio['positions'][stock_code]
                    
                    # 檢查T+1規則 - 計算可賣出數量 (排除今日買入股票)
                    sellable_quantity = pos['quantity']
                    if today in pos['purchase_dates']:
                        sellable_quantity -= pos['purchase_dates'][today]
                    
                    # 檢查是否有足夠可賣出股票
                    if sellable_quantity >= quantity:
                        # 賣出 (符合T+1規則)
                        self.current_portfolio['cash'] += trade_amount
                        pos['quantity'] -= quantity
                        
                        # 從舊的持股中扣除 (FIFO原則)
                        remaining_to_sell = quantity
                        for date in sorted(pos['purchase_dates'].keys()):
                            if date == today:  # 今日買入不能賣
                                continue
                            if remaining_to_sell <= 0:
                                break
                            available = pos['purchase_dates'][date]
                            if available > 0:
                                sold_from_date = min(remaining_to_sell, available)
                                pos['purchase_dates'][date] -= sold_from_date
                                remaining_to_sell -= sold_from_date
                                if pos['purchase_dates'][date] == 0:
                                    del pos['purchase_dates'][date]
                        
                        # 計算損益
                        cost = quantity * pos['avg_price']
                        pnl = trade_amount - cost
                        self.current_portfolio['daily_pnl'] += pnl
                        self.current_portfolio['total_pnl'] += pnl
                        self.daily_stats['pnl'] += pnl
                        
                        # 如果全部賣出，移除持股
                        if pos['quantity'] == 0:
                            del self.current_portfolio['positions'][stock_code]
                            
                        self.daily_stats['trades'] += 1
                        
                        if enhancement_version == '2.0':
                            self.logger.info(
                                f"✅ 增強版賣出: {stock_code} {quantity}股 @{price:.2f} "
                                f"損益: ${pnl:+.0f} 信心度:{original_confidence:.3f}→{confidence:.3f} "
                                f"(技術:{technical_score:.3f}, 情緒:{sentiment_score:.3f}) - {reasoning[:60]}..."
                            )
                        else:
                            self.logger.info(
                                f"✅ 賣出成功: {stock_code} {quantity}股 @{price:.2f} "
                                f"損益: ${pnl:+.0f} (信心度:{confidence:.3f}) - {reasoning[:100]}..."
                            )
                    else:
                        # T+1規則限制
                        today_bought = pos['purchase_dates'].get(today, 0)
                        self.logger.warning(
                            f"⚠️ T+1規則限制: {stock_code} 可賣{sellable_quantity}股，今日買入{today_bought}股無法當日賣出 "
                            f"(要求賣出{quantity}股)"
                        )
                else:
                    self.logger.warning(f"⚠️ 無持股，無法賣出 {stock_code}")
            
            # 保存投資組合狀態
            await self._save_portfolio_state()
            
        except Exception as e:
            self.logger.error(f"❌ 執行交易決策失敗: {e}")

    async def _log_trading_session(self, session_data: Dict):
        """記錄完整的交易會話"""
        try:
            # 保存到JSON日誌
            log_file = os.path.join(
                SYSTEM_ROOT, 'realtime_trading_system', 'logs', 
                f'trading_sessions_{datetime.now().strftime("%Y%m%d")}.json'
            )
            
            # 讀取現有日誌
            sessions = []
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    sessions = json.load(f)
            
            # 添加新會話
            sessions.append(session_data)
            
            # 保存日誌
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"📝 交易會話已記錄到: {log_file}")
            
        except Exception as e:
            self.logger.error(f"記錄交易會話失敗: {e}")

    async def _daily_summary_report(self):
        """生成每日交易總結報告"""
        try:
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'trading_stats': self.daily_stats.copy(),
                'portfolio': self.current_portfolio.copy(),
                'report_time': datetime.now().isoformat()
            }
            
            # 保存報告
            report_file = os.path.join(
                SYSTEM_ROOT, 'realtime_trading_system', 'logs',
                f'daily_report_{datetime.now().strftime("%Y%m%d")}.json'
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 記錄到日誌
            self.logger.info(
                f"📊 每日交易總結: 交易{self.daily_stats['trades']}筆, "
                f"決策{self.daily_stats['decisions']}次, "
                f"損益${self.daily_stats['pnl']:+.0f}, "
                f"現金餘額${self.current_portfolio['cash']:,.0f}"
            )
            
        except Exception as e:
            self.logger.error(f"生成每日報告失敗: {e}")

    async def run(self):
        """主執行循環"""
        self.logger.info("🚀 台股AI實時自動交易系統啟動")
        
        # 初始化組件
        if not await self._initialize_components():
            self.logger.error("❌ 系統初始化失敗，退出")
            return
        
        self.is_running = True
        scan_interval = 300  # 5分鐘掃描一次
        last_scan_time = None
        
        while self.is_running:
            try:
                current_time = datetime.now()
                self.is_trading_hours = self._is_trading_time()
                
                if self.is_trading_hours:
                    # 交易時間內，執行掃描和決策
                    if not last_scan_time or (current_time - last_scan_time).seconds >= scan_interval:
                        await self._scan_and_decide()
                        last_scan_time = current_time
                        
                    # 每小時報告狀態
                    if current_time.minute == 0:
                        self.logger.info(
                            f"⏰ 交易中 - 現金: ${self.current_portfolio['cash']:,.0f}, "
                            f"持股: {len(self.current_portfolio['positions'])}, "
                            f"今日交易: {self.daily_stats['trades']}, "
                            f"今日損益: ${self.daily_stats['pnl']:+.0f}"
                        )
                        
                else:
                    # 非交易時間
                    if current_time.hour == 14 and current_time.minute == 0:
                        # 收盤後生成每日報告
                        await self._daily_summary_report()
                        
                        # 重置每日統計
                        self.daily_stats = {'trades': 0, 'decisions': 0, 'pnl': 0}
                        self.current_portfolio['daily_pnl'] = 0
                        
                    # 非交易時間較長間隔檢查
                    await asyncio.sleep(60)  # 1分鐘檢查一次
                    continue
                
                # 正常間隔
                await asyncio.sleep(30)  # 30秒檢查一次
                
            except KeyboardInterrupt:
                self.logger.info("👋 接收到中斷信號，系統準備關閉...")
                break
            except Exception as e:
                self.logger.error(f"❌ 主循環出現錯誤: {e}")
                await asyncio.sleep(60)  # 錯誤時暫停1分鐘
        
        # 系統關閉
        await self._daily_summary_report()
        self.logger.info("👋 台股AI實時自動交易系統已關閉")

async def main():
    """主函數"""
    system = RealTimeTradingSystem()
    await system.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n系統被用戶中斷")
    except Exception as e:
        print(f"系統執行錯誤: {e}")