#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動態權重學習系統 - 台股AI交易
根據市場條件動態調整 AI、技術指標、市場情緒的權重
"""

import numpy as np
import pandas as pd
import json
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DynamicWeightLearner:
    def __init__(self, model_path="dynamic_weights_model.pkl"):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False
        
        # 載入預訓練模型（如果存在）
        if os.path.exists(model_path):
            self.load_model()
    
    def collect_market_features(self, days=30):
        """收集市場特徵用於權重學習"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # 台股加權指數
            taiex = yf.download('^TWII', start=start_date, end=end_date, progress=False)
            
            if len(taiex) == 0:
                return None
                
            # 計算市場特徵
            features = {
                # 波動率（過去5天）
                'volatility': float(taiex['Close'].pct_change().rolling(5).std().iloc[-1] * 100),
                
                # 成交量趨勢（5日均量 / 20日均量）
                'volume_trend': float(taiex['Volume'].rolling(5).mean().iloc[-1] / 
                                    taiex['Volume'].rolling(20).mean().iloc[-1]),
                
                # 價格動量（5日漲跌幅）
                'price_momentum': float((taiex['Close'].iloc[-1] - taiex['Close'].iloc[-6]) / 
                                      taiex['Close'].iloc[-6] * 100),
                
                # 交易時間特徵（9:00-13:30，14:00代表盤中）
                'time_factor': self.get_time_factor(),
                
                # RSI相對強弱指標
                'market_rsi': self.calculate_rsi(taiex['Close']),
                
                # 是否為趨勢市場
                'trend_strength': self.calculate_trend_strength(taiex['Close'])
            }
            
            return features
            
        except Exception as e:
            print(f"市場特徵收集錯誤: {e}")
            return None
    
    def get_time_factor(self):
        """根據當前時間計算時間因子"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # 盤前：0.2, 開盤初：0.8, 盤中：1.0, 收盤前：0.6
        if hour < 9:
            return 0.2
        elif hour == 9 and minute < 30:
            return 0.8
        elif hour >= 9 and hour < 13:
            return 1.0
        elif hour == 13 and minute <= 30:
            return 1.0
        else:
            return 0.6
    
    def calculate_rsi(self, prices, period=14):
        """計算RSI指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def calculate_trend_strength(self, prices, window=10):
        """計算趨勢強度"""
        ma_short = prices.rolling(5).mean()
        ma_long = prices.rolling(20).mean()
        
        # 短期均線與長期均線的偏離程度
        trend_strength = abs((ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1] * 100)
        return float(trend_strength) if not np.isnan(trend_strength) else 1.0
    
    def generate_training_data(self, num_samples=1000):
        """生成訓練數據（模擬不同市場情況）"""
        np.random.seed(42)
        
        # 模擬不同市場條件
        training_data = []
        
        for _ in range(num_samples):
            # 模擬市場特徵
            features = {
                'volatility': np.random.uniform(0.5, 3.0),
                'volume_trend': np.random.uniform(0.7, 1.5),
                'price_momentum': np.random.uniform(-5.0, 5.0),
                'time_factor': np.random.choice([0.2, 0.6, 0.8, 1.0]),
                'market_rsi': np.random.uniform(20, 80),
                'trend_strength': np.random.uniform(0.5, 5.0)
            }
            
            # 根據市場條件計算理想權重
            weights = self.calculate_ideal_weights(features)
            
            training_data.append({
                'features': features,
                'weights': weights
            })
        
        return training_data
    
    def calculate_ideal_weights(self, features):
        """根據市場特徵計算理想權重分配"""
        # 基礎權重
        ai_weight = 0.4
        technical_weight = 0.3
        sentiment_weight = 0.3
        
        # 根據波動率調整：高波動時增加技術指標權重
        if features['volatility'] > 2.0:
            technical_weight += 0.1
            ai_weight -= 0.05
            sentiment_weight -= 0.05
        
        # 根據趨勢強度調整：強趨勢時增加AI權重
        if features['trend_strength'] > 3.0:
            ai_weight += 0.1
            technical_weight -= 0.05
            sentiment_weight -= 0.05
        
        # 根據RSI調整：極端RSI時增加技術指標權重
        if features['market_rsi'] < 30 or features['market_rsi'] > 70:
            technical_weight += 0.1
            ai_weight -= 0.05
            sentiment_weight -= 0.05
        
        # 根據時間因子調整：盤中時增加情緒權重
        if features['time_factor'] == 1.0:
            sentiment_weight += 0.1
            ai_weight -= 0.05
            technical_weight -= 0.05
        
        # 確保權重總和為1
        total = ai_weight + technical_weight + sentiment_weight
        weights = [ai_weight/total, technical_weight/total, sentiment_weight/total]
        
        return weights
    
    def train_model(self, training_data=None):
        """訓練動態權重模型"""
        if training_data is None:
            training_data = self.generate_training_data()
        
        # 準備特徵和目標
        X = []
        y = []
        
        for data in training_data:
            features_vector = [
                data['features']['volatility'],
                data['features']['volume_trend'],
                data['features']['price_momentum'],
                data['features']['time_factor'],
                data['features']['market_rsi'],
                data['features']['trend_strength']
            ]
            X.append(features_vector)
            y.append(data['weights'])
        
        X = np.array(X)
        y = np.array(y)
        
        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 訓練模型（多輸出回歸）
        self.model.fit(X_train, y_train)
        
        # 評估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"模型訓練完成!")
        print(f"訓練集準確率: {train_score:.4f}")
        print(f"測試集準確率: {test_score:.4f}")
        
        self.is_trained = True
        self.save_model()
        
        return train_score, test_score
    
    def predict_weights(self, market_features=None):
        """預測當前市場條件下的最佳權重"""
        if not self.is_trained:
            print("模型尚未訓練，使用預設權重")
            return [0.4, 0.3, 0.3]  # AI, Technical, Sentiment
        
        if market_features is None:
            market_features = self.collect_market_features()
        
        if market_features is None:
            print("無法獲取市場特徵，使用預設權重")
            return [0.4, 0.3, 0.3]
        
        # 準備特徵向量
        features_vector = np.array([[
            market_features['volatility'],
            market_features['volume_trend'],
            market_features['price_momentum'],
            market_features['time_factor'],
            market_features['market_rsi'],
            market_features['trend_strength']
        ]])
        
        # 標準化
        features_scaled = self.scaler.transform(features_vector)
        
        # 預測權重
        predicted_weights = self.model.predict(features_scaled)[0]
        
        # 確保權重為正且總和為1
        predicted_weights = np.clip(predicted_weights, 0.1, 0.8)
        predicted_weights = predicted_weights / predicted_weights.sum()
        
        return predicted_weights.tolist()
    
    def save_model(self):
        """儲存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已儲存至: {self.model_path}")
    
    def load_model(self):
        """載入模型"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            print(f"模型已載入: {self.model_path}")
            
        except Exception as e:
            print(f"載入模型失敗: {e}")
            self.is_trained = False

def test_dynamic_weight_learner():
    """測試動態權重學習系統"""
    print("=== 動態權重學習系統測試 ===")
    
    # 初始化學習器
    learner = DynamicWeightLearner()
    
    # 訓練模型
    print("\n1. 訓練模型...")
    learner.train_model()
    
    # 獲取當前市場特徵
    print("\n2. 獲取當前市場特徵...")
    market_features = learner.collect_market_features()
    print("市場特徵:", market_features)
    
    # 預測權重
    print("\n3. 預測最佳權重...")
    weights = learner.predict_weights(market_features)
    print(f"AI權重: {weights[0]:.3f}")
    print(f"技術指標權重: {weights[1]:.3f}")
    print(f"市場情緒權重: {weights[2]:.3f}")
    print(f"權重總和: {sum(weights):.3f}")
    
    return learner, weights

if __name__ == "__main__":
    test_dynamic_weight_learner()