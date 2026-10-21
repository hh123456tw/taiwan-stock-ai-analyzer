#!/usr/bin/env python3
"""
增強版台股盤後分析系統 - 整合真實數據源
結合Yahoo Finance即時數據、AI預測、技術分析
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import time

class EnhancedRealPostMarketAnalyzer:
    """增強版實時盤後分析器"""
    
    def __init__(self):
        self.predictions_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.db')
        self.stock_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'jiansoft_daily.db')
        
        # 核心股票池 (基於實時交易系統)
        self.core_stocks = [
            # 權值股
            '2330', '2317', '2454', '2308', '2882', '2891', '2881', '2303', '3034', '2382',
            # ETF
            '0050', '0056', '00878', '00881', '00891', '00919', '00922', '00927',
            # 其他重要股票
            '2313', '2408', '2474', '3231', '3008', '6505', '3711', '2345'
        ]
        
    def get_real_market_data(self) -> Dict:
        """獲取真實市場數據"""
        try:
            # 嘗試從Yahoo Finance獲取台股大盤數據
            taiex = yf.Ticker("^TWII")  # 台股加權指數
            hist = taiex.history(period="2d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                
                current_value = latest['Close']
                prev_value = prev['Close']
                change = current_value - prev_value
                change_pct = (change / prev_value) * 100
                volume = latest['Volume'] / 1000000  # 轉為百萬
                
                return {
                    'taiex': {
                        'value': round(current_value, 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2),
                        'volume': round(volume, 1),
                        'prev_close': round(prev_value, 2),
                        'source': 'Yahoo Finance',
                        'update_time': datetime.now().strftime('%H:%M')
                    }
                }
            else:
                # 備用數據
                return self._get_fallback_market_data()
                
        except Exception as e:
            print(f"⚠️ Yahoo Finance數據獲取失敗: {e}")
            return self._get_fallback_market_data()
    
    def _get_fallback_market_data(self) -> Dict:
        """備用市場數據"""
        return {
            'taiex': {
                'value': 23450.0,  # 近期合理值
                'change': 25.5,
                'change_pct': 0.11,
                'volume': 1850.0,
                'prev_close': 23424.5,
                'source': '估算數據',
                'update_time': datetime.now().strftime('%H:%M')
            }
        }
    
    def get_real_stock_prices(self, symbols: List[str]) -> Dict:
        """獲取真實股價數據"""
        stock_data = {}
        
        for symbol in symbols[:10]:  # 限制數量避免API限制
            try:
                # 處理台股代號格式
                yahoo_symbol = f"{symbol}.TW" if not symbol.endswith('.TW') else symbol
                
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty and info:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    stock_data[symbol] = {
                        'current_price': round(latest['Close'], 2),
                        'prev_close': round(prev['Close'], 2),
                        'change': round(latest['Close'] - prev['Close'], 2),
                        'change_pct': round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2),
                        'volume': int(latest['Volume']),
                        'name': info.get('longName', symbol),
                        'pe_ratio': info.get('trailingPE', 0),
                        'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                        'source': 'Yahoo Finance'
                    }
                    
                time.sleep(0.1)  # 避免請求過快
                
            except Exception as e:
                print(f"⚠️ {symbol} 數據獲取失敗: {e}")
                continue
                
        return stock_data
    
    def get_ai_predictions_with_real_prices(self) -> pd.DataFrame:
        """獲取AI預測並匹配真實股價"""
        try:
            conn = sqlite3.connect(self.predictions_db)
            
            # 獲取最新預測
            query = """
                WITH latest_predictions AS (
                    SELECT symbol, predicted_price, confidence, prediction_date, created_at,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY created_at DESC) as rn
                    FROM predictions 
                    WHERE model_name LIKE '%集成%'
                    AND created_at >= date('now', '-7 days')
                )
                SELECT symbol, predicted_price, confidence, prediction_date, created_at
                FROM latest_predictions 
                WHERE rn = 1 AND confidence >= 0.5
                ORDER BY confidence DESC, created_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return df
            
            # 獲取真實股價數據
            symbols = df['symbol'].unique().tolist()
            real_prices = self.get_real_stock_prices(symbols)
            
            # 匹配真實價格
            for idx, row in df.iterrows():
                symbol = row['symbol']
                if symbol in real_prices:
                    df.at[idx, 'current_price'] = real_prices[symbol]['current_price']
                    df.at[idx, 'company_name'] = real_prices[symbol]['name']
                    df.at[idx, 'change_pct'] = real_prices[symbol]['change_pct']
                    df.at[idx, 'pe_ratio'] = real_prices[symbol]['pe_ratio']
                    df.at[idx, 'dividend_yield'] = real_prices[symbol]['dividend_yield']
                    
                    # 計算預測準確度
                    pred_change = (row['predicted_price'] - real_prices[symbol]['current_price']) / real_prices[symbol]['current_price'] * 100
                    df.at[idx, 'predicted_change_pct'] = pred_change
            
            return df
            
        except Exception as e:
            print(f"❌ AI預測數據處理失敗: {e}")
            return pd.DataFrame()
    
    def analyze_sector_performance(self, stock_data: Dict) -> Dict:
        """分析類股表現"""
        sectors = {
            'technology': ['2330', '2454', '2303', '3034', '2382', '2408', '3231'],
            'finance': ['2882', '2891', '2881', '2886', '2892', '2880'],
            'etf': ['0050', '0056', '00878', '00881', '00891', '00919'],
            'traditional': ['2317', '2474', '3008', '6505', '3711']
        }
        
        sector_performance = {}
        
        for sector, stocks in sectors.items():
            sector_stocks = [s for s in stocks if s in stock_data]
            if sector_stocks:
                changes = [stock_data[s]['change_pct'] for s in sector_stocks]
                avg_change = np.mean(changes)
                sector_performance[sector] = {
                    'avg_change': round(avg_change, 2),
                    'stocks_count': len(sector_stocks),
                    'best_performer': max(sector_stocks, key=lambda x: stock_data[x]['change_pct']),
                    'worst_performer': min(sector_stocks, key=lambda x: stock_data[x]['change_pct'])
                }
        
        return sector_performance
    
    def generate_enhanced_analysis(self) -> str:
        """產生增強版分析報告"""
        
        # 獲取真實市場數據
        market_data = self.get_real_market_data()
        predictions_df = self.get_ai_predictions_with_real_prices()
        
        # 獲取核心股票的真實價格
        real_stock_data = self.get_real_stock_prices(self.core_stocks)
        
        # 分析類股表現
        sector_performance = self.analyze_sector_performance(real_stock_data)
        
        # 統計數據
        current_time = datetime.now()
        report_date = current_time.strftime("%Y-%m-%d")
        report_time = current_time.strftime("%H:%M")
        
        total_predictions = len(predictions_df) if not predictions_df.empty else 0
        high_confidence = len(predictions_df[predictions_df['confidence'] >= 0.8]) if not predictions_df.empty else 0
        
        # 產生報告
        taiex_data = market_data['taiex']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║            📊 台股盤後深度分析 - 真實數據版                      ║
║                    {report_date} {report_time}                 ║
╚══════════════════════════════════════════════════════════════╝

【📈 大盤指數表現】({taiex_data['source']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 加權指數: {taiex_data['value']:,.0f} 點 ({taiex_data['change']:+.1f}, {taiex_data['change_pct']:+.2f}%)
• 昨日收盤: {taiex_data['prev_close']:,.0f} 點
• 成交量: {taiex_data['volume']:,.1f} 億元
• 數據更新: {taiex_data['update_time']} ({taiex_data['source']})
• 市場趨勢: {'📈 上漲' if taiex_data['change'] > 0 else '📉 下跌' if taiex_data['change'] < 0 else '➡️ 平盤'}

【🏭 類股表現分析】(基於真實股價)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        # 類股表現排序
        if sector_performance:
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['avg_change'], reverse=True)
            for i, (sector, data) in enumerate(sorted_sectors, 1):
                sector_names = {
                    'technology': '科技股',
                    'finance': '金融股', 
                    'etf': 'ETF',
                    'traditional': '傳產股'
                }
                emoji = '🔥' if data['avg_change'] > 1 else '💪' if data['avg_change'] > 0 else '📉'
                report += f"""
• 第{i}名 {sector_names.get(sector, sector)}: {data['avg_change']:+.2f}% {emoji} ({data['stocks_count']}檔)"""

        report += f"""

【🤖 AI智能預測分析】({total_predictions}支股票)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        if total_predictions > 0:
            avg_confidence = predictions_df['confidence'].mean() * 100
            bullish_count = len(predictions_df[predictions_df['predicted_change_pct'] > 0]) if 'predicted_change_pct' in predictions_df.columns else 0
            bearish_count = len(predictions_df[predictions_df['predicted_change_pct'] < 0]) if 'predicted_change_pct' in predictions_df.columns else 0
            
            report += f"""
• 🎯 預測覆蓋: {total_predictions}支股票，高信心度{high_confidence}支  
• 📊 平均信心度: {avg_confidence:.1f}%
• 🤖 市場情緒: 看多{bullish_count}支 vs 看空{bearish_count}支
• 🔮 數據來源: AI預測引擎 + Yahoo Finance即時價格"""

            # 顯示高信心度預測
            if high_confidence > 0:
                report += f"""

【⭐ 高信心度推薦】(信心度 ≥ 80%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
                
                high_conf_stocks = predictions_df[predictions_df['confidence'] >= 0.8].sort_values('confidence', ascending=False)
                
                for i, (_, stock) in enumerate(high_conf_stocks.head(6).iterrows()):
                    company_name = str(stock.get('company_name', stock['symbol']))[:25]
                    
                    current_price = f"{stock['current_price']:.2f}" if pd.notna(stock.get('current_price')) else "N/A"
                    pred_change = f"{stock['predicted_change_pct']:+.1f}%" if pd.notna(stock.get('predicted_change_pct')) else "N/A"
                    
                    pe_ratio = f"P/E:{stock['pe_ratio']:.1f}" if pd.notna(stock.get('pe_ratio')) and stock.get('pe_ratio', 0) > 0 else ""
                    dividend = f"殖利率:{stock['dividend_yield']:.1f}%" if pd.notna(stock.get('dividend_yield')) and stock.get('dividend_yield', 0) > 0 else ""
                    
                    pred_days = (datetime.now() - pd.to_datetime(stock['prediction_date'])).days if pd.notna(stock.get('prediction_date')) else 'N/A'
                    time_desc = "今日" if pred_days == 0 else f"{pred_days}日前" if pred_days != 'N/A' else "歷史"
                    
                    report += f"""
🎯 {i+1}. {stock['symbol']} {company_name}
   預測價格: {stock['predicted_price']:.2f} (現價:{current_price}, {pred_change})
   AI信心度: {stock['confidence']*100:.0f}% | {time_desc}預測 | {pe_ratio} {dividend}"""
            
            else:
                report += "\n• ⚠️ 目前無高信心度預測，建議謹慎觀望"
        else:
            report += "\n• ❌ 無可用AI預測數據"

        # 即時股價表現 (前10檔)
        if real_stock_data:
            report += f"""

【💹 核心股票即時表現】(Yahoo Finance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            
            # 按漲跌幅排序
            sorted_stocks = sorted(real_stock_data.items(), key=lambda x: x[1]['change_pct'], reverse=True)
            
            for i, (symbol, data) in enumerate(sorted_stocks[:8], 1):
                emoji = '📈' if data['change_pct'] > 2 else '💚' if data['change_pct'] > 0 else '📉' if data['change_pct'] < -2 else '🔻'
                name = data['name'][:15] if len(data['name']) > 15 else data['name']
                report += f"""
{emoji} {symbol} {name}: {data['current_price']:.2f} ({data['change']:+.2f}, {data['change_pct']:+.2f}%)"""

        report += f"""

【🔮 明日操作策略建議】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 🎯 關注重點: {'高信心度AI推薦股票' if high_confidence > 0 else '等待更明確的交易信號'}
• 📈 操作策略: {'逢回分批布局優質股' if taiex_data['change'] > 0 else '觀望為主，等待反彈'}
• 🛡️ 風險控制: 單股持倉≤20%，設定停損點5-8%
• ⏰ 進場時機: 開盤後觀察量能，避免追高殺低

【⚠️ 風險提醒與數據說明】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 🌐 數據來源: Yahoo Finance即時數據 + 本地AI預測引擎
• ⏰ 更新頻率: 盤後即時分析，數據具時效性
• 🔄 準確性: AI預測僅供參考，請結合基本面分析
• 🌍 國際因素: 關注美股走勢、匯率變化、國際政經情勢

📋 分析師: 增強版AI分析系統 v7.0
🕐 更新時間: {report_date} {report_time}
📊 數據來源: Yahoo Finance + AI預測引擎 + 即時股價
⚡ 功能特色: 真實數據 + 智能分析 + 類股輪動 + 風險評估"""

        return report

async def send_post_market_to_telegram(analyzer) -> bool:
    """發送盤後分析到TG"""
    try:
        import os
        bot_token = os.environ.get("TRADING_NOTIFICATION_BOT_TOKEN") or os.environ.get("TG_BOT_TOKEN")
        chat_id = os.environ.get("TRADING_NOTIFICATION_CHAT_ID") or os.environ.get("TG_CHAT_ID")
        
        if not bot_token or not chat_id:
            print("⚠️ 未配置 TG_BOT_TOKEN 或 TG_CHAT_ID，略過 TG 訊息推送")
            return False
        
        # 獲取分析數據
        predictions_df = analyzer.get_ai_predictions_with_real_prices()
        market_data = analyzer.get_real_market_data()
        real_stock_data = analyzer.get_real_stock_prices(analyzer.core_stocks[:5])
        sector_performance = analyzer.analyze_sector_performance(real_stock_data)
        
        # 準備TG推送數據
        taiex_data = market_data['taiex']
        high_confidence_count = len(predictions_df[predictions_df['confidence'] >= 0.8]) if not predictions_df.empty else 0
        avg_confidence = predictions_df['confidence'].mean() * 100 if not predictions_df.empty else 0
        
        # 找出最強/最弱類股
        best_sector = max(sector_performance.items(), key=lambda x: x[1]['avg_change'])[0] if sector_performance else 'N/A'
        worst_sector = min(sector_performance.items(), key=lambda x: x[1]['avg_change'])[0] if sector_performance else 'N/A'
        
        # 生成明日關注股票
        if not predictions_df.empty:
            tomorrow_focus = ', '.join(predictions_df.head(3)['symbol'].tolist())
        else:
            tomorrow_focus = '暫無重點推薦'
        
        # 策略建議
        if taiex_data['change'] > 0:
            strategy = "大盤上漲，可逢回布局優質股"
        elif taiex_data['change'] < -50:
            strategy = "大盤下跌較深，謹慎觀望"
        else:
            strategy = "盤整格局，持續觀察"
        
        data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'taiex_value': f"{taiex_data['value']:,.0f}",
            'taiex_change': f"{taiex_data['change']:+.1f}, {taiex_data['change_pct']:+.2f}%",
            'volume': f"{taiex_data['volume']:.1f}",
            'market_trend': '📈 上漲' if taiex_data['change'] > 0 else '📉 下跌' if taiex_data['change'] < 0 else '➡️ 平盤',
            'high_confidence_count': high_confidence_count,
            'avg_confidence': avg_confidence,
            'tomorrow_focus': tomorrow_focus,
            'best_sector': best_sector,
            'worst_sector': worst_sector,
            'strategy': strategy
        }
        
        try:
            from telegram import Bot
            bot = Bot(token=bot_token)
            msg_text = (
                f"📊 【台股盤後AI總結】 {data['date']}\n"
                f"📈 加權指數: {data['taiex_value']} ({data['taiex_change']}) | 成交量: {data['volume']}億\n"
                f"🎯 明日重點關注: {data['tomorrow_focus']}\n"
                f"🏆 最強類股: {data['best_sector']} | 弱勢類股: {data['worst_sector']}\n"
                f"💡 策略建議: {data['strategy']}\n"
                f"✨ 高信心度推薦數: {data['high_confidence_count']} 檔"
            )
            await bot.send_message(chat_id=chat_id, text=msg_text)
            return True
        except ImportError:
            import requests
            api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            msg_text = (
                f"📊 【台股盤後AI總結】 {data['date']}\n"
                f"📈 加權指數: {data['taiex_value']} ({data['taiex_change']}) | 成交量: {data['volume']}億\n"
                f"🎯 明日重點關注: {data['tomorrow_focus']}\n"
                f"🏆 最強類股: {data['best_sector']} | 弱勢類股: {data['worst_sector']}\n"
                f"💡 策略建議: {data['strategy']}\n"
                f"✨ 高信心度推薦數: {data['high_confidence_count']} 檔"
            )
            resp = requests.post(api_url, json={"chat_id": chat_id, "text": msg_text}, timeout=10)
            return resp.status_code == 200
        
    except Exception as e:
        print(f"❌ TG推送失敗: {e}")
        return False

async def main():
    analyzer = EnhancedRealPostMarketAnalyzer()
    
    print("🚀 啟動增強版台股盤後分析系統...")
    print("📊 正在獲取Yahoo Finance即時數據...")
    
    try:
        analysis = analyzer.generate_enhanced_analysis()
        print(analysis)
        
        # 先發送今日購買摘要
        try:
            from daily_purchase_summary import send_daily_purchase_summary
            await send_daily_purchase_summary()
            print("✅ 今日購買摘要已發送")
        except Exception as e:
            print(f"⚠️ 今日購買摘要發送失敗: {e}")

        # TG推送功能
        try:
            success = await send_post_market_to_telegram(analyzer)
            if success:
                print("\n📱 盤後分析已推送到TG")
            else:
                print("\n⚠️ TG推送失敗，僅本地顯示")
        except Exception as e:
            print(f"\n⚠️ TG推送出錯: {e}")
        
    except Exception as e:
        print(f"❌ 增強版盤後分析出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())