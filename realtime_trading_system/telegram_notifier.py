#!/usr/bin/env python3
"""
實時交易系統 Telegram 通知模組
當AI做出交易決策時即時通知
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime

class TradingTelegramNotifier:
    """交易Telegram通知器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def notify_trade_execution(self, trade_data: Dict) -> bool:
        """通知交易執行"""
        try:
            # 動態導入避免依賴問題
            import sys
            sys.path.append('/mnt/nvme/taiwan-stock-ai-analyzer-main')
            from ai_trading_notification_bot import AITradingNotificationBot
            
            bot = AITradingNotificationBot()
            
            # 判斷交易類型
            action = trade_data.get('action', '').upper()
            
            if action == 'BUY':
                alert_type = 'buy'
            elif action == 'SELL':
                alert_type = 'sell'
            else:
                alert_type = 'decision'
            
            # 準備通知資料
            notification_data = {
                'symbol': trade_data.get('symbol', ''),
                'name': trade_data.get('company_name', ''),
                'price': trade_data.get('price', trade_data.get('target_price', 0)),
                'shares': trade_data.get('shares', trade_data.get('quantity', 0)),
                'amount': trade_data.get('investment_amount', 0),
                'confidence': trade_data.get('confidence', 0),
                'reason': trade_data.get('decision_reason', trade_data.get('approval_reason', ''))
            }
            
            # 對於賣出交易，加入損益資訊
            if action == 'SELL':
                notification_data['pnl'] = trade_data.get('pnl', 0)
            
            success = await bot.send_trading_alert(alert_type, notification_data)
            
            if success:
                self.logger.info(f"✅ 交易通知已發送: {action} {trade_data.get('symbol', '')}")
            else:
                self.logger.warning(f"⚠️ 交易通知發送失敗: {action} {trade_data.get('symbol', '')}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 發送交易通知時出錯: {e}")
            return False
    
    async def notify_market_scan_result(self, scan_data: Dict) -> bool:
        """通知市場掃描結果"""
        try:
            import sys
            sys.path.append('/mnt/nvme/taiwan-stock-ai-analyzer-main')
            from ai_trading_notification_bot import AITradingNotificationBot
            
            bot = AITradingNotificationBot()
            
            # 只在有重要決策時通知，避免太多訊息
            candidates = scan_data.get('candidates', 0)
            approved = scan_data.get('approved_decisions', 0)
            
            # 只有當找到候選但被拒絕時才通知（避免垃圾訊息）
            if candidates > 0 and approved == 0:
                notification_data = {
                    'candidates': candidates,
                    'approved': approved,
                    'rejection_reason': scan_data.get('rejection_reason', '未知'),
                    'market_sentiment': scan_data.get('market_sentiment', 'NEUTRAL'),
                    'available_cash': scan_data.get('available_cash', 0)
                }
                
                success = await bot.send_trading_alert('decision', notification_data)
                return success
            
            return True  # 不需要通知時也返回True
            
        except Exception as e:
            self.logger.error(f"❌ 發送掃描通知時出錯: {e}")
            return False
    
    def notify_sync(self, alert_type: str, data: Dict) -> bool:
        """同步版本的通知方法（用於非async環境）"""
        try:
            import threading
            
            # 在新線程中運行async代碼避免事件循環衝突
            result = [False]  # 使用列表保存結果
            
            def run_async():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    if alert_type == 'trade':
                        result[0] = loop.run_until_complete(self.notify_trade_execution(data))
                    elif alert_type == 'scan':
                        result[0] = loop.run_until_complete(self.notify_market_scan_result(data))
                    
                    loop.close()
                except Exception as e:
                    self.logger.error(f"❌ async執行出錯: {e}")
            
            # 在新線程中執行
            thread = threading.Thread(target=run_async)
            thread.start()
            thread.join(timeout=10)  # 10秒超時
            
            return result[0]
            
        except Exception as e:
            self.logger.error(f"❌ 同步通知出錯: {e}")
            return False

# 全局通知器實例
_notifier = None

def get_notifier() -> TradingTelegramNotifier:
    """獲取全局通知器實例"""
    global _notifier
    if _notifier is None:
        _notifier = TradingTelegramNotifier()
    return _notifier

def notify_trade(trade_data: Dict) -> bool:
    """便捷函數：通知交易"""
    notifier = get_notifier()
    return notifier.notify_sync('trade', trade_data)

def notify_scan(scan_data: Dict) -> bool:
    """便捷函數：通知掃描結果"""
    notifier = get_notifier()
    return notifier.notify_sync('scan', scan_data)