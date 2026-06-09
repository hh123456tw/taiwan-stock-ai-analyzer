import requests
import json
import re
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import time

class MarketSentimentEngine:
    '''市場情緒分析引擎'''
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentiment_cache = {}
        self.cache_duration = 1800  # 30分鐘緩存
        
        # 市場情緒關鍵字
        self.positive_keywords = [
            '上漲', '漲停', '創新高', '突破', '買進', '看多', '利多', 
            '強勢', '反彈', '轉強', '量增', '獲利', '成長', '樂觀'
        ]
        
        self.negative_keywords = [
            '下跌', '跌停', '創新低', '破底', '賣出', '看空', '利空',
            '弱勢', '回檔', '轉弱', '量縮', '虧損', '衰退', '悲觀'
        ]
    
    def get_market_sentiment_score(self, symbol: str) -> float:
        '''獲取股票市場情緒評分 (0-1)'''
        try:
            # 檢查緩存
            cache_key = f'{symbol}_sentiment'
            if self._is_cache_valid(cache_key):
                return self.sentiment_cache[cache_key]['score']
            
            # 計算綜合情緒分數
            news_score = self._get_news_sentiment(symbol)
            market_score = self._get_market_momentum_score(symbol)
            volume_score = self._get_volume_sentiment(symbol)
            
            # 綜合權重: 新聞30% + 市場動量40% + 成交量30%
            final_score = (news_score * 0.3) + (market_score * 0.4) + (volume_score * 0.3)
            final_score = max(0.0, min(1.0, final_score))
            
            # 更新緩存
            self.sentiment_cache[cache_key] = {
                'score': final_score,
                'timestamp': datetime.now(),
                'components': {
                    'news': news_score,
                    'market': market_score,
                    'volume': volume_score
                }
            }
            
            self.logger.info(f'📈 {symbol} 情緒分析: 綜合={final_score:.3f} (新聞={news_score:.3f}, 市場={market_score:.3f}, 量能={volume_score:.3f})')
            return final_score
            
        except Exception as e:
            self.logger.warning(f'情緒分析失敗 {symbol}: {e}')
            return 0.5  # 返回中性評分
    
    def _get_news_sentiment(self, symbol: str) -> float:
        '''分析新聞情緒 (模擬實現)'''
        try:
            # 實際應用中可整合新聞API或網路爬蟲
            # 這裡使用簡化邏輯
            
            # 模擬新聞情緒分析結果
            import random
            random.seed(hash(symbol + datetime.now().strftime('%Y%m%d')))
            
            # 大盤股傾向較中性
            if symbol in ['2330', '2317', '2454']:
                base_score = 0.45 + (random.random() * 0.1)  # 0.45-0.55
            # ETF傾向穩定
            elif symbol.startswith('00'):
                base_score = 0.48 + (random.random() * 0.04)  # 0.48-0.52
            else:
                base_score = 0.4 + (random.random() * 0.2)   # 0.4-0.6
            
            return base_score
            
        except:
            return 0.5
    
    def _get_market_momentum_score(self, symbol: str) -> float:
        '''計算市場動量情緒'''
        try:
            import yfinance as yf
            
            # 獲取近期價格數據
            stock = yf.Ticker(f'{symbol}.TW')
            hist = stock.history(period='1mo')
            
            if len(hist) < 10:
                return 0.5
            
            # 計算價格動量指標
            current_price = hist['Close'].iloc[-1]
            ma5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma10 = hist['Close'].rolling(10).mean().iloc[-1]
            ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            
            # 動量評分
            score = 0.5
            
            # 價格相對移動平均線位置
            if current_price > ma5 > ma10 > ma20:
                score += 0.3  # 強勢上升
            elif current_price > ma5 > ma10:
                score += 0.2  # 上升趨勢
            elif current_price > ma10:
                score += 0.1  # 輕微上升
            elif current_price < ma5 < ma10 < ma20:
                score -= 0.3  # 強勢下降
            elif current_price < ma5 < ma10:
                score -= 0.2  # 下降趨勢
            elif current_price < ma10:
                score -= 0.1  # 輕微下降
            
            # 近期漲跌幅
            recent_change = (current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]
            if recent_change > 0.05:    # 5日漲幅>5%
                score += 0.15
            elif recent_change > 0.02:  # 5日漲幅>2%
                score += 0.08
            elif recent_change < -0.05: # 5日跌幅>5%
                score -= 0.15
            elif recent_change < -0.02: # 5日跌幅>2%
                score -= 0.08
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f'市場動量計算失敗 {symbol}: {e}')
            return 0.5
    
    def _get_volume_sentiment(self, symbol: str) -> float:
        '''分析成交量情緒'''
        try:
            import yfinance as yf
            
            stock = yf.Ticker(f'{symbol}.TW')
            hist = stock.history(period='1mo')
            
            if len(hist) < 10:
                return 0.5
            
            # 計算成交量相關指標
            current_volume = hist['Volume'].iloc[-1]
            avg_volume_10 = hist['Volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = hist['Volume'].rolling(20).mean().iloc[-1]
            
            # 價格變化
            price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
            
            score = 0.5
            
            # 量價關係分析
            if current_volume > avg_volume_20 * 1.5:  # 大量
                if price_change > 0:
                    score += 0.2  # 大量上漲，正面
                else:
                    score -= 0.15  # 大量下跌，負面
            elif current_volume > avg_volume_10 * 1.2:  # 溫和放量
                if price_change > 0:
                    score += 0.1  # 溫和放量上漲
                else:
                    score -= 0.08  # 溫和放量下跌
            elif current_volume < avg_volume_20 * 0.7:  # 量縮
                if price_change > 0:
                    score += 0.05  # 縮量上漲，相對正面
                else:
                    score -= 0.05  # 縮量下跌，持續觀察
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f'成交量分析失敗 {symbol}: {e}')
            return 0.5
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        '''檢查緩存是否有效'''
        if cache_key not in self.sentiment_cache:
            return False
        
        cache_time = self.sentiment_cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def get_market_overview_sentiment(self) -> Dict[str, float]:
        '''獲取整體市場情緒'''
        # 重要指標股票
        key_stocks = ['2330', '0050', '2317', '2454', '2412']
        
        market_sentiment = {}
        total_score = 0
        
        for stock in key_stocks:
            score = self.get_market_sentiment_score(stock)
            market_sentiment[stock] = score
            total_score += score
        
        market_sentiment['overall'] = total_score / len(key_stocks)
        
        return market_sentiment

# 測試功能
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    sentiment_engine = MarketSentimentEngine()
    
    print('🧪 測試市場情緒分析')
    print('=' * 50)
    
    # 測試個股情緒
    test_stocks = ['2330', '0050', '2454']
    
    for stock in test_stocks:
        score = sentiment_engine.get_market_sentiment_score(stock)
        print(f'{stock}: 情緒評分 {score:.3f}')
    
    # 測試市場總覽
    market_overview = sentiment_engine.get_market_overview_sentiment()
    print(f'\n📊 市場總體情緒: {market_overview["overall"]:.3f}')
    
    print('✅ 市場情緒分析測試完成')
