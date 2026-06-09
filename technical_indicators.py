import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import logging

class SimpleTechnicalIndicators:
    '''簡化版技術指標計算器'''
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_technical_score(self, symbol):
        '''獲取股票技術評分'''
        try:
            # 獲取股票數據
            stock = yf.Ticker(f'{symbol}.TW')
            hist = stock.history(period='2mo')
            
            if len(hist) < 20:
                return 0.5  # 數據不足，返回中性評分
            
            close_prices = hist['Close']
            current_price = close_prices.iloc[-1]
            
            # 簡單RSI計算
            rsi = self.calculate_simple_rsi(close_prices)
            
            # 簡單MACD計算
            macd_score = self.calculate_simple_macd(close_prices)
            
            # 價格趨勢評分
            trend_score = self.calculate_trend_score(close_prices)
            
            # 綜合評分
            final_score = (rsi * 0.3) + (macd_score * 0.4) + (trend_score * 0.3)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.warning(f'計算 {symbol} 技術指標失敗: {e}')
            return 0.5
    
    def calculate_simple_rsi(self, prices, period=14):
        '''簡化RSI計算'''
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # RSI評分轉換
            if current_rsi < 30:
                return 0.7  # 超賣，偏買進
            elif current_rsi > 70:
                return 0.3  # 超買，偏賣出
            else:
                return 0.5  # 中性
                
        except:
            return 0.5
    
    def calculate_simple_macd(self, prices):
        '''簡化MACD計算'''
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            # MACD評分
            if current_macd > current_signal:
                return 0.6  # 黃金交叉，偏看漲
            else:
                return 0.4  # 死亡交叉，偏看跌
                
        except:
            return 0.5
    
    def calculate_trend_score(self, prices):
        '''計算價格趨勢評分'''
        try:
            # 計算多個時間週期的移動平均
            ma5 = prices.rolling(window=5).mean().iloc[-1]
            ma10 = prices.rolling(window=10).mean().iloc[-1]
            ma20 = prices.rolling(window=20).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            score = 0.5
            
            # 價格相對移動平均線位置
            if current_price > ma5 > ma10 > ma20:
                score = 0.8  # 強勢上升趨勢
            elif current_price > ma5 > ma10:
                score = 0.65  # 上升趨勢
            elif current_price > ma10:
                score = 0.55  # 輕微上升
            elif current_price < ma5 < ma10 < ma20:
                score = 0.2  # 強勢下降趨勢
            elif current_price < ma5 < ma10:
                score = 0.35  # 下降趨勢
            elif current_price < ma10:
                score = 0.45  # 輕微下降
            
            return score
            
        except:
            return 0.5

# 測試功能
if __name__ == '__main__':
    tech = SimpleTechnicalIndicators()
    
    test_stocks = ['2330', '0050', '2454']
    
    print('🧪 測試簡化技術指標')
    print('=' * 40)
    
    for stock in test_stocks:
        score = tech.get_technical_score(stock)
        print(f'{stock}: 技術評分 {score:.3f}')
    
    print('✅ 測試完成')
