#!/usr/bin/env python3
"""
市場情緒分析器 - 客觀指標版
整合VIX、資金流向、融資融券等客觀市場數據
"""
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    """市場情緒分析器 - 基於客觀指標"""
    
    def __init__(self):
        """初始化市場情緒分析器"""
        self.vix_thresholds = {
            'extreme_fear': 40,    # 極度恐慌
            'fear': 30,           # 恐慌
            'neutral_high': 20,   # 中性偏高
            'neutral_low': 15,    # 中性偏低
            'complacency': 12     # 自滿
        }
        
        logger.info("✅ 市場情緒分析器初始化完成")
    
    def get_vix_sentiment(self) -> Dict:
        """獲取VIX恐慌指數情緒"""
        try:
            # 獲取VIX數據
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")
            
            if vix_data.empty:
                raise ValueError("無法獲取VIX數據")
            
            current_vix = vix_data['Close'].iloc[-1]
            prev_vix = vix_data['Close'].iloc[-2] if len(vix_data) > 1 else current_vix
            vix_change = current_vix - prev_vix
            vix_change_pct = (vix_change / prev_vix) * 100
            
            # 判斷情緒等級
            if current_vix >= self.vix_thresholds['extreme_fear']:
                sentiment = "極度恐慌"
                emoji = "😱"
                score = 0.1  # 極低分表示恐慌
                description = "市場極度恐慌，可能是抄底良機"
            elif current_vix >= self.vix_thresholds['fear']:
                sentiment = "恐慌"
                emoji = "😰"
                score = 0.25
                description = "市場恐慌情緒濃厚，風險偏好低"
            elif current_vix >= self.vix_thresholds['neutral_high']:
                sentiment = "中性偏空"
                emoji = "😐"
                score = 0.4
                description = "市場謹慎，波動性偏高"
            elif current_vix >= self.vix_thresholds['neutral_low']:
                sentiment = "中性"
                emoji = "😊"
                score = 0.6
                description = "市場情緒穩定，正常波動"
            elif current_vix >= self.vix_thresholds['complacency']:
                sentiment = "樂觀"
                emoji = "😃"
                score = 0.75
                description = "市場樂觀，低波動環境"
            else:
                sentiment = "過度樂觀"
                emoji = "🤑"
                score = 0.9
                description = "市場可能過度樂觀，注意回調風險"
            
            # 變化趨勢
            if vix_change_pct > 10:
                trend = "恐慌急升"
                trend_emoji = "📈💀"
            elif vix_change_pct > 5:
                trend = "恐慌上升"
                trend_emoji = "📈😰"
            elif vix_change_pct < -10:
                trend = "恐慌急降"
                trend_emoji = "📉😊"
            elif vix_change_pct < -5:
                trend = "恐慌下降"
                trend_emoji = "📉🙂"
            else:
                trend = "穩定"
                trend_emoji = "➡️"
            
            return {
                'current_vix': round(current_vix, 2),
                'vix_change': round(vix_change, 2),
                'vix_change_pct': round(vix_change_pct, 2),
                'sentiment': sentiment,
                'emoji': emoji,
                'score': score,
                'description': description,
                'trend': trend,
                'trend_emoji': trend_emoji,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"VIX情緒分析失敗: {e}")
            return {
                'current_vix': 20.0,
                'sentiment': "數據異常",
                'emoji': "❓",
                'score': 0.5,
                'description': "VIX數據暫時無法獲取",
                'error': str(e)
            }
    
    def get_market_breadth_sentiment(self) -> Dict:
        """獲取市場廣度情緒（漲跌家數比）"""
        try:
            # 獲取主要市場指數
            symbols = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'DIA': 'Dow Jones'
            }
            
            breadth_data = {}
            total_score = 0
            
            for symbol, name in symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    
                    if len(hist) >= 2:
                        today_close = hist['Close'].iloc[-1]
                        yesterday_close = hist['Close'].iloc[-2]
                        change_pct = ((today_close - yesterday_close) / yesterday_close) * 100
                        
                        breadth_data[symbol] = {
                            'name': name,
                            'change_pct': round(float(change_pct), 2),
                            'positive': bool(change_pct > 0)
                        }
                        
                        # 計算分數 (正面=1, 負面=0)
                        total_score += 1 if change_pct > 0 else 0
                        
                except Exception as e:
                    logger.warning(f"無法獲取 {symbol} 數據: {e}")
            
            if not breadth_data:
                raise ValueError("無法獲取任何市場廣度數據")
            
            # 計算整體情緒
            positive_ratio = total_score / len(breadth_data)
            
            if positive_ratio >= 0.8:
                sentiment = "強勢"
                emoji = "🚀"
                description = "多數指數上漲，市場情緒強勢"
            elif positive_ratio >= 0.6:
                sentiment = "偏多"
                emoji = "📈"
                description = "多數指數表現正面"
            elif positive_ratio >= 0.4:
                sentiment = "分歧"
                emoji = "⚖️"
                description = "指數表現分歧，方向不明"
            elif positive_ratio >= 0.2:
                sentiment = "偏空"
                emoji = "📉"
                description = "多數指數下跌"
            else:
                sentiment = "弱勢"
                emoji = "💀"
                description = "多數指數重挫，市場情緒悲觀"
            
            return {
                'positive_ratio': round(positive_ratio, 3),
                'sentiment': sentiment,
                'emoji': emoji,
                'description': description,
                'score': positive_ratio,
                'details': breadth_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"市場廣度分析失敗: {e}")
            return {
                'sentiment': "數據異常",
                'emoji': "❓",
                'score': 0.5,
                'description': "市場廣度數據暫時無法獲取",
                'error': str(e)
            }
    
    def get_crypto_sentiment(self) -> Dict:
        """獲取加密貨幣情緒（風險資產指標）"""
        try:
            # 比特幣作為風險情緒指標
            btc = yf.Ticker("BTC-USD")
            btc_data = btc.history(period="5d")
            
            if btc_data.empty:
                raise ValueError("無法獲取BTC數據")
            
            # 計算近期波動和趨勢
            recent_returns = btc_data['Close'].pct_change().dropna()
            current_price = btc_data['Close'].iloc[-1]
            volatility = recent_returns.std() * 100  # 轉為百分比
            
            # 5日趨勢
            if len(btc_data) >= 5:
                price_5d_ago = btc_data['Close'].iloc[-5]
                trend_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
            else:
                trend_5d = 0
            
            # 情緒判斷 (基於趨勢和波動性)
            if trend_5d > 10 and volatility < 5:
                sentiment = "強勢上漲"
                emoji = "🚀"
                score = 0.9
                description = "比特幣強勢上漲，風險偏好高"
            elif trend_5d > 5:
                sentiment = "上漲"
                emoji = "📈"
                score = 0.75
                description = "比特幣上漲，風險偏好回升"
            elif trend_5d > -5:
                sentiment = "震盪"
                emoji = "📊"
                score = 0.5
                description = "比特幣盤整，風險偏好中性"
            elif trend_5d > -10:
                sentiment = "下跌"
                emoji = "📉"
                score = 0.25
                description = "比特幣下跌，風險偏好降低"
            else:
                sentiment = "重挫"
                emoji = "💀"
                score = 0.1
                description = "比特幣重挫，風險偏好極低"
            
            return {
                'current_price': round(current_price, 0),
                'trend_5d': round(trend_5d, 2),
                'volatility': round(volatility, 2),
                'sentiment': sentiment,
                'emoji': emoji,
                'score': score,
                'description': description,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"加密貨幣情緒分析失敗: {e}")
            return {
                'sentiment': "數據異常",
                'emoji': "❓",
                'score': 0.5,
                'description': "加密貨幣數據暫時無法獲取",
                'error': str(e)
            }
    
    def get_bond_yield_sentiment(self) -> Dict:
        """獲取債券收益率情緒（資金流向指標）"""
        try:
            # 10年期美債收益率
            tny = yf.Ticker("^TNX")
            tny_data = tny.history(period="5d")
            
            if tny_data.empty:
                raise ValueError("無法獲取10年期美債數據")
            
            current_yield = tny_data['Close'].iloc[-1]
            
            # 收益率變化
            if len(tny_data) > 1:
                prev_yield = tny_data['Close'].iloc[-2]
                yield_change = current_yield - prev_yield
            else:
                yield_change = 0
            
            # 根據收益率水平和變化判斷情緒
            if current_yield > 5.0:  # 高收益率
                if yield_change > 0.1:
                    sentiment = "資金大量流出股市"
                    emoji = "🏃‍♂️💰"
                    score = 0.2
                else:
                    sentiment = "高收益環境"
                    emoji = "💰"
                    score = 0.3
            elif current_yield > 4.0:  # 中等收益率
                if yield_change > 0.1:
                    sentiment = "資金轉向債券"
                    emoji = "📈💵"
                    score = 0.4
                elif yield_change < -0.1:
                    sentiment = "資金回流股市"
                    emoji = "📉📈"
                    score = 0.7
                else:
                    sentiment = "資金流向中性"
                    emoji = "⚖️"
                    score = 0.5
            else:  # 低收益率
                sentiment = "低收益推升風險資產"
                emoji = "📈🚀"
                score = 0.8
            
            description = f"10年期美債收益率 {current_yield:.2f}% ({yield_change:+.2f}%)"
            
            return {
                'current_yield': round(current_yield, 3),
                'yield_change': round(yield_change, 3),
                'sentiment': sentiment,
                'emoji': emoji,
                'score': score,
                'description': description,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"債券收益率分析失敗: {e}")
            return {
                'sentiment': "數據異常",
                'emoji': "❓",
                'score': 0.5,
                'description': "債券收益率數據暫時無法獲取",
                'error': str(e)
            }
    
    def analyze_comprehensive_market_sentiment(self) -> Dict:
        """綜合市場情緒分析"""
        logger.info("開始綜合市場情緒分析...")
        
        # 獲取各項指標
        vix_sentiment = self.get_vix_sentiment()
        breadth_sentiment = self.get_market_breadth_sentiment()
        crypto_sentiment = self.get_crypto_sentiment()
        bond_sentiment = self.get_bond_yield_sentiment()
        
        # 計算權重平均分數
        indicators = [
            (vix_sentiment, 0.3),      # VIX權重30%
            (breadth_sentiment, 0.25), # 市場廣度25%
            (crypto_sentiment, 0.25),  # 加密貨幣25%
            (bond_sentiment, 0.2)      # 債券20%
        ]
        
        total_score = 0
        total_weight = 0
        valid_indicators = []
        
        for indicator, weight in indicators:
            if 'score' in indicator and 'error' not in indicator:
                total_score += indicator['score'] * weight
                total_weight += weight
                valid_indicators.append(indicator)
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.5  # 默認中性
        
        # 綜合情緒判斷
        if final_score >= 0.8:
            overall_sentiment = "極度樂觀"
            overall_emoji = "🤑"
            risk_level = "高"
            advice = "市場可能過熱，注意回調風險"
        elif final_score >= 0.65:
            overall_sentiment = "樂觀"
            overall_emoji = "😊"
            risk_level = "中等"
            advice = "市場情緒正面，可適度參與"
        elif final_score >= 0.45:
            overall_sentiment = "中性"
            overall_emoji = "😐"
            risk_level = "中等"
            advice = "市場情緒分歧，謹慎觀望"
        elif final_score >= 0.3:
            overall_sentiment = "悲觀"
            overall_emoji = "😰"
            risk_level = "中等"
            advice = "市場情緒偏空，等待機會"
        else:
            overall_sentiment = "極度悲觀"
            overall_emoji = "😱"
            risk_level = "低"
            advice = "極度恐慌可能是抄底良機"
        
        # 關鍵信號
        key_signals = []
        if 'current_vix' in vix_sentiment:
            if vix_sentiment['current_vix'] > 30:
                key_signals.append(f"VIX高達{vix_sentiment['current_vix']:.1f}，恐慌情緒濃厚")
        
        if 'positive_ratio' in breadth_sentiment:
            if breadth_sentiment['positive_ratio'] > 0.8:
                key_signals.append("多數指數上漲，市場廣度良好")
            elif breadth_sentiment['positive_ratio'] < 0.2:
                key_signals.append("多數指數下跌，市場廣度惡化")
        
        result = {
            'overall_sentiment': overall_sentiment,
            'overall_emoji': overall_emoji,
            'final_score': round(final_score, 3),
            'risk_level': risk_level,
            'advice': advice,
            'key_signals': key_signals,
            'indicators': {
                'vix': vix_sentiment,
                'market_breadth': breadth_sentiment,
                'crypto': crypto_sentiment,
                'bond_yield': bond_sentiment
            },
            'valid_indicators_count': len(valid_indicators),
            'analysis_time': datetime.now().isoformat()
        }
        
        logger.info(f"✅ 綜合市場情緒分析完成: {overall_sentiment} (評分: {final_score:.3f})")
        return result

def main():
    """測試函數"""
    analyzer = MarketSentimentAnalyzer()
    
    print("=== 市場情緒分析測試 ===")
    
    # 綜合分析
    result = analyzer.analyze_comprehensive_market_sentiment()
    
    print(f"\n📊 綜合市場情緒: {result['overall_sentiment']} {result['overall_emoji']}")
    print(f"📈 情緒評分: {result['final_score']:.3f}")
    print(f"⚠️ 風險等級: {result['risk_level']}")
    print(f"💡 建議: {result['advice']}")
    
    if result['key_signals']:
        print("\n🔍 關鍵信號:")
        for signal in result['key_signals']:
            print(f"  • {signal}")
    
    print("\n📋 詳細指標:")
    for name, indicator in result['indicators'].items():
        if 'error' not in indicator:
            emoji = indicator.get('emoji', '📊')
            sentiment = indicator.get('sentiment', '未知')
            description = indicator.get('description', '')
            print(f"  {name}: {sentiment} {emoji} - {description}")
        else:
            print(f"  {name}: 數據異常 - {indicator.get('error', '')}")
    
    # 詳細JSON輸出
    print(f"\n=== 完整JSON結果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()