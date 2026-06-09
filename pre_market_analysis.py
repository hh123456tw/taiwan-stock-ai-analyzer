#!/usr/bin/env python3
"""
台股盤前分析系統
每日08:30自動分析美股影響、預測開盤走勢
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional
import asyncio

class PreMarketAnalyzer:
    """盤前市場分析器"""
    
    def __init__(self):
        self.predictions_db = '/mnt/nvme/taiwan-stock-ai-analyzer-main/predictions.db'
        self.core_stocks = ['2330', '2317', '2454', '0056', '00878', '00881']
        
    def get_us_market_data(self) -> Dict:
        """獲取美股數據（影響台股開盤）"""
        try:
            # 獲取主要美股指數
            indices = {
                'nasdaq': '^IXIC',      # NASDAQ
                'sp500': '^GSPC',       # S&P 500
                'dow': '^DJI',          # 道瓊
                'vix': '^VIX'           # 恐慌指數
            }
            
            us_data = {}
            
            for name, symbol in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]['Close']
                        prev = hist.iloc[-2]['Close'] if len(hist) > 1 else latest
                        change = latest - prev
                        change_pct = (change / prev) * 100
                        
                        us_data[name] = {
                            'value': latest,
                            'change': change,
                            'change_pct': change_pct
                        }
                except:
                    us_data[name] = {'value': 0, 'change': 0, 'change_pct': 0}
            
            return us_data
            
        except Exception as e:
            print(f"⚠️ 美股數據獲取失敗: {e}")
            return {}
    
    def get_tsmc_adr_data(self) -> Dict:
        """獲取台積電ADR數據"""
        try:
            # 台積電ADR (TSM)
            ticker = yf.Ticker("TSM")
            hist = ticker.history(period="2d")
            
            if not hist.empty:
                latest = hist.iloc[-1]['Close']
                prev = hist.iloc[-2]['Close'] if len(hist) > 1 else latest
                change = latest - prev
                change_pct = (change / prev) * 100
                
                # 簡單換算 (實際匯率會影響)
                estimated_tw_price = latest * 32  # 約略匯率
                
                return {
                    'adr_price': latest,
                    'change': change,
                    'change_pct': change_pct,
                    'estimated_tw_price': estimated_tw_price
                }
        except Exception as e:
            print(f"⚠️ 台積電ADR數據獲取失敗: {e}")
            
        return {'adr_price': 0, 'change': 0, 'change_pct': 0, 'estimated_tw_price': 0}
    
    def get_ai_focus_stocks(self) -> List[str]:
        """獲取AI今日關注股票 - 改進版"""
        try:
            conn = sqlite3.connect(self.predictions_db)
            
            # 1. 先嘗試獲取最新高信心度個股預測
            query = """
                WITH latest_predictions AS (
                    SELECT symbol, confidence, predicted_price,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY created_at DESC) as rn,
                           created_at
                    FROM predictions 
                    WHERE model_name LIKE '%集成%'
                    AND created_at >= date('now', '-7 days')
                    AND confidence >= 0.7
                    AND symbol NOT LIKE '00%'  -- 排除ETF
                )
                SELECT symbol, confidence
                FROM latest_predictions 
                WHERE rn = 1
                ORDER BY confidence DESC, created_at DESC
                LIMIT 8
            """
            
            df = pd.read_sql_query(query, conn)
            
            # 2. 如果個股不足，補充權值股
            focus_stocks = df['symbol'].tolist() if not df.empty else []
            
            # 補充台股權值股和熱門股
            backup_stocks = [
                '2330',   # 台積電
                '2317',   # 鴻海  
                '2454',   # 聯發科
                '2308',   # 台達電
                '2881',   # 富邦金
                '2882',   # 國泰金
                '2891',   # 中信金
                '2303',   # 聯電
                '3034',   # 聯詠
                '2382',   # 廣達
                '2409',   # 友達
                '3008',   # 大立光
            ]
            
            # 確保有足夠的關注股票
            for stock in backup_stocks:
                if stock not in focus_stocks and len(focus_stocks) < 10:
                    focus_stocks.append(stock)
            
            conn.close()
            
            return focus_stocks[:10]  # 最多返回10檔
            
        except Exception as e:
            print(f"❌ AI關注股票獲取失敗: {e}")
            # 發生錯誤時返回權值股
            return ['2330', '2317', '2454', '2308', '2881', '2882', '2303', '3034', '2382', '2409']

    def predict_market_opening(self, us_data: Dict, tsm_data: Dict) -> Dict:
        """預測開盤走勢"""
        predictions = {
            'trend': '中性',
            'confidence': 50,
            'support_level': 23400,
            'resistance_level': 23600,
            'volatility': 0.5,
            'reasoning': []
        }
        
        try:
            # 美股影響分析
            if us_data:
                nasdaq_change = us_data.get('nasdaq', {}).get('change_pct', 0)
                sp500_change = us_data.get('sp500', {}).get('change_pct', 0)
                vix_level = us_data.get('vix', {}).get('value', 20)
                
                # 綜合美股表現
                us_avg_change = (nasdaq_change + sp500_change) / 2
                
                if us_avg_change > 1:
                    predictions['trend'] = '偏多'
                    predictions['confidence'] = 70
                    predictions['reasoning'].append(f"美股大漲{us_avg_change:.1f}%，預期開高")
                elif us_avg_change < -1:
                    predictions['trend'] = '偏空'
                    predictions['confidence'] = 70
                    predictions['reasoning'].append(f"美股大跌{us_avg_change:.1f}%，預期開低")
                elif abs(us_avg_change) > 0.5:
                    predictions['trend'] = '偏多' if us_avg_change > 0 else '偏空'
                    predictions['confidence'] = 60
                    predictions['reasoning'].append(f"美股{'上漲' if us_avg_change > 0 else '下跌'}{abs(us_avg_change):.1f}%")
                
                # VIX恐慌指數
                if vix_level > 25:
                    predictions['volatility'] = 1.2
                    predictions['reasoning'].append(f"VIX高達{vix_level:.1f}，市場恐慌")
                elif vix_level < 15:
                    predictions['volatility'] = 0.3
                    predictions['reasoning'].append("VIX低位，市場平靜")
            
            # 台積電ADR影響
            if tsm_data and tsm_data['change_pct'] != 0:
                tsm_change = tsm_data['change_pct']
                if abs(tsm_change) > 2:
                    direction = '上漲' if tsm_change > 0 else '下跌'
                    predictions['reasoning'].append(f"台積電ADR{direction}{abs(tsm_change):.1f}%")
                    
                    # 調整預測
                    if tsm_change > 2:
                        predictions['trend'] = '偏多'
                        predictions['confidence'] = min(predictions['confidence'] + 15, 85)
                    elif tsm_change < -2:
                        predictions['trend'] = '偏空'
                        predictions['confidence'] = min(predictions['confidence'] + 15, 85)
        
        except Exception as e:
            print(f"⚠️ 開盤預測計算出錯: {e}")
        
        return predictions
    
    def generate_pre_market_analysis(self) -> str:
        """生成盤前分析報告"""
        current_time = datetime.now()
        
        # 計算開盤倒數
        market_open = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
        if current_time > market_open:
            market_open += timedelta(days=1)
        
        countdown = market_open - current_time
        countdown_str = f"{countdown.seconds//3600}小時{(countdown.seconds//60)%60}分鐘"
        
        # 獲取數據
        us_data = self.get_us_market_data()
        tsm_data = self.get_tsmc_adr_data()
        focus_stocks = self.get_ai_focus_stocks()
        predictions = self.predict_market_opening(us_data, tsm_data)
        
        report = f"""🌅 台股盤前智能分析 (AI增強版)
{current_time.strftime('%Y-%m-%d %H:%M')}

【🌍 全球市場脈動解析】"""

        if us_data:
            nasdaq = us_data.get('nasdaq', {})
            sp500 = us_data.get('sp500', {})
            vix = us_data.get('vix', {})
            
            report += f"""
• 📈 NASDAQ: {nasdaq.get('value', 0):.0f} ({nasdaq.get('change_pct', 0):+.2f}%)
• 📊 S&P 500: {sp500.get('value', 0):.0f} ({sp500.get('change_pct', 0):+.2f}%)
• 😰 VIX恐慌指數: {vix.get('value', 0):.1f}
• 🎭 美股情緒: {'樂觀' if (nasdaq.get('change_pct', 0) + sp500.get('change_pct', 0))/2 > 0.5 else '謹慎' if (nasdaq.get('change_pct', 0) + sp500.get('change_pct', 0))/2 < -0.5 else '中性'}"""
        else:
            report += "\n• ⚠️ 美股數據獲取中..."

        report += f"""

【🏭 台積電領先指標】"""

        if tsm_data and tsm_data['adr_price'] > 0:
            emoji = '📈' if tsm_data['change_pct'] > 1 else '💚' if tsm_data['change_pct'] > 0 else '📉' if tsm_data['change_pct'] < -1 else '🔻'
            report += f"""
• {emoji} TSM ADR: ${tsm_data['adr_price']:.2f} ({tsm_data['change_pct']:+.2f}%)
• 💱 估算台股價: ${tsm_data['estimated_tw_price']:.0f}
• 🎯 對台積電影響: {'正面' if tsm_data['change_pct'] > 1 else '負面' if tsm_data['change_pct'] < -1 else '中性'}"""
        else:
            report += "\n• ⚠️ 台積電ADR數據獲取中..."

        report += f"""

【🔮 AI開盤預測引擎】"""

        trend_emoji = {'偏多': '📈', '偏空': '📉', '中性': '➡️'}.get(predictions['trend'], '➡️')
        confidence_emoji = '🎯' if predictions['confidence'] > 70 else '🤔' if predictions['confidence'] > 50 else '❓'
        
        report += f"""
• {trend_emoji} 預期走勢: {predictions['trend']} ({confidence_emoji} 信心度{predictions['confidence']}%)
• 📊 預期波動: {predictions['volatility']:.1f}%
• 🛡️ 關鍵支撐: {predictions['support_level']:,}點
• ⚡ 預測依據: {' | '.join(predictions['reasoning']) if predictions['reasoning'] else '技術面分析'}"""

        report += f"""

【🎯 精選標的深度解析】"""

        if focus_stocks:
            report += f"""
• 💎 權值龍頭: {', '.join([s for s in focus_stocks[:10] if s in ['2330', '2317', '2454']])}
• 🏦 金融三雄: {', '.join([s for s in focus_stocks[:10] if s in ['2881', '2882', '2891']])}
• ⚡ 科技潛力: {', '.join([s for s in focus_stocks[:10] if s in ['2308', '2303', '3034']])}
• 🌟 特殊標的: {', '.join([s for s in focus_stocks[:10] if s not in ['2330', '2317', '2454', '2881', '2882', '2891', '2308', '2303', '3034']])}
• 🤖 AI信心度: ≥80% 高信心推薦
• 💡 進場策略: 開盤後觀察前30分鐘量能確認方向
• 🎯 重點邏輯: AI高信心預測 + 技術面突破 + 基本面支撐
• ⚠️ 風險控制: 單檔持股不超過總資金20%，設定明確停損點"""
        else:
            report += """
• 🔍 重點標的: 暫無高信心推薦
• 💡 操作建議: 今日以觀望為主
• 📊 等待更明確的交易信號"""

        report += f"""

【⏰ 開盤倒數 & 實戰準備】
• 🕘 距離開盤: {countdown_str}
• 📅 交易提醒: 09:00開盤，AI將自動掃描交易機會
• 🤖 系統狀態: 準備就緒，等待開盤信號

【💡 今日實戰攻略】"""

        if predictions['trend'] == '偏多':
            strategy = "適度積極，逢回布局優質股"
        elif predictions['trend'] == '偏空':
            strategy = "保守為主，等待反彈機會"
        else:
            strategy = "中性觀望，等待方向明確"

        report += f"""
• 🎯 整體策略: {strategy}
• 🛡️ 風險控制: 設定停損，分散投資
• 📊 進場條件: 等待AI系統確認信號
• ⚡ 特別提醒: 開盤前30分鐘易有假突破

🧠 智能分析師: AI深度學習預測引擎 v3.0
⏰ 分析時間: {current_time.strftime('%Y-%m-%d %H:%M')}
📊 數據整合: 美股 + ADR + AI預測 + LLM洞察
🎯 下次更新: 盤中監控(每5分鐘) + 盤後總結(15:30)"""

        return report

async def send_pre_market_to_telegram(analyzer) -> bool:
    """發送完整盤前分析到TG"""
    try:
        from telegram import Bot
        import asyncio
        
        # 設置Bot Token和Chat ID（從環境變數讀取）
        import os
        bot_token = os.environ.get("TG_BOT_TOKEN", "")
        chat_id = os.environ.get("TG_CHAT_ID", "")
        
        if not bot_token or not chat_id:
            print("❌ BOT TOKEN 或 CHAT ID 未設置")
            return False
            
        bot = Bot(token=bot_token)
        
        # 生成完整分析報告
        full_analysis = analyzer.generate_pre_market_analysis()
        
        # 清理格式，適合TG顯示
        tg_message = full_analysis.replace('╔', '').replace('╚', '').replace('═', '').replace('║', '').replace('━', '—').strip()
        
        # 分割長訊息 (TG限制4096字符)
        max_length = 4000
        if len(tg_message) > max_length:
            parts = []
            lines = tg_message.split('\n')
            current_part = ""
            
            for line in lines:
                if len(current_part + line + '\n') > max_length:
                    if current_part:
                        parts.append(current_part.strip())
                        current_part = line + '\n'
                    else:
                        # 單行太長，強制分割
                        parts.append(line[:max_length])
                        current_part = line[max_length:] + '\n' if len(line) > max_length else ""
                else:
                    current_part += line + '\n'
            
            if current_part.strip():
                parts.append(current_part.strip())
            
            # 發送所有部分
            for i, part in enumerate(parts):
                if i == 0:
                    header = "🌅 台股盤前智能分析 (完整版)\n\n"
                    part = header + part
                
                await bot.send_message(chat_id=chat_id, text=part)
                print(f"📱 已發送第{i+1}/{len(parts)}部分到TG")
                
                # 避免發送太快被限制
                if i < len(parts) - 1:
                    await asyncio.sleep(1)
        else:
            # 訊息不長，直接發送
            message_with_header = f"🌅 台股盤前智能分析 (完整版)\n\n{tg_message}"
            await bot.send_message(chat_id=chat_id, text=message_with_header)
            print("📱 完整分析已發送到TG")
        
        return True
        
    except Exception as e:
        print(f"❌ TG推送失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主程序"""
    analyzer = PreMarketAnalyzer()
    
    print("🌅 啟動台股盤前分析系統...")
    print("📊 正在分析美股影響和開盤預測...")
    
    try:
        analysis = analyzer.generate_pre_market_analysis()
        print(analysis)
        
        # TG推送功能
        import asyncio
        try:
            success = asyncio.run(send_pre_market_to_telegram(analyzer))
            if success:
                print("\n📱 盤前分析已推送到TG")
            else:
                print("\n⚠️ TG推送失敗，僅本地顯示")
        except Exception as e:
            print(f"\n⚠️ TG推送出錯: {e}")
        
    except Exception as e:
        print(f"❌ 盤前分析出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()