#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改進版動態權重學習系統 - 台股AI交易
修正pandas警告，增強市場適應性
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
import warnings

# 抑制pandas警告
warnings.filterwarnings('ignore', category=FutureWarning)

class ImprovedDynamicWeightLearner:
    def __init__(self, model_path="dynamic_weights_model_v2.pkl"):
        self.model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False
        
        # 市場狀態緩存
        self.market_cache = {}
        self.cache_time = None
        self.cache_duration = 300  # 5分鐘緩存
        
        # 載入預訓練模型（如果存在）
        if os.path.exists(model_path):
            self.load_model()
    
    def collect_market_features_safe(self, days=30):
        """安全的市場特徵收集（修正pandas警告）"""
        # 檢查緩存
        now = datetime.now()
        if (self.cache_time and 
            (now - self.cache_time).seconds < self.cache_duration and 
            self.market_cache):
            return self.market_cache
        
        end_date = now
        start_date = end_date - timedelta(days=days)
        
        try:
            # 台股加權指數
            taiex = yf.download('^TWII', start=start_date, end=end_date, progress=False)
            
            if len(taiex) < 10:  # 數據不足
                return self._get_fallback_features()
                
            # 安全的數值轉換
            def safe_float(series_value):
                try:
                    if pd.isna(series_value):
                        return 0.0
                    return float(series_value)
                except:
                    return 0.0
            
            # 計算市場特徵（修正pandas警告）
            volatility_series = taiex['Close'].pct_change().rolling(5).std() * 100
            volume_ma5 = taiex['Volume'].rolling(5).mean()
            volume_ma20 = taiex['Volume'].rolling(20).mean()
            
            features = {
                # 波動率（過去5天）
                'volatility': safe_float(volatility_series.iloc[-1]),
                
                # 成交量趨勢（5日均量 / 20日均量）
                'volume_trend': safe_float(volume_ma5.iloc[-1] / volume_ma20.iloc[-1]) if safe_float(volume_ma20.iloc[-1]) > 0 else 1.0,
                
                # 價格動量（5日漲跌幅）
                'price_momentum': safe_float((taiex['Close'].iloc[-1] - taiex['Close'].iloc[-6]) / taiex['Close'].iloc[-6] * 100),
                
                # 交易時間特徵
                'time_factor': self.get_time_factor(),
                
                # RSI相對強弱指標
                'market_rsi': self.calculate_rsi_safe(taiex['Close']),
                
                # 趨勢強度
                'trend_strength': self.calculate_trend_strength_safe(taiex['Close']),
                
                # 🆕 新增特徵：市場情緒指標
                'market_sentiment': self.calculate_market_sentiment(taiex),
                
                # 🆕 新增特徵：波動率變化
                'volatility_change': self.calculate_volatility_change(taiex['Close'])
            }
            
            # 更新緩存
            self.market_cache = features
            self.cache_time = now
            
            return features
            
        except Exception as e:
            print(f"市場特徵收集錯誤: {e}")
            return self._get_fallback_features()
    
    def _get_fallback_features(self):
        """回退特徵（當無法獲取實時數據時）"""
        return {
            'volatility': 1.0,
            'volume_trend': 1.0,
            'price_momentum': 0.0,
            'time_factor': self.get_time_factor(),
            'market_rsi': 50.0,
            'trend_strength': 1.0,
            'market_sentiment': 0.5,
            'volatility_change': 0.0
        }
    
    def get_time_factor(self):
        """根據當前時間計算時間因子（增強版）"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()
        
        # 週末因子
        if weekday >= 5:  # 週六日
            return 0.1
        
        # 盤前：0.2, 開盤初：0.9, 盤中前半：1.0, 盤中後半：0.8, 收盤前：0.6
        if hour < 9:
            return 0.2
        elif hour == 9 and minute < 30:
            return 0.9
        elif hour >= 9 and hour < 11:
            return 1.0
        elif hour >= 11 and hour < 13:
            return 0.8
        elif hour == 13 and minute <= 30:
            return 0.8
        else:
            return 0.6
    
    def calculate_rsi_safe(self, prices, period=14):
        """安全的RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            result = rsi.iloc[-1]
            return float(result) if not pd.isna(result) else 50.0
            
        except Exception:
            return 50.0
    
    def calculate_trend_strength_safe(self, prices, window=10):
        """安全的趨勢強度計算"""
        try:
            ma_short = prices.rolling(5).mean()
            ma_long = prices.rolling(20).mean()
            
            # 短期均線與長期均線的偏離程度
            if len(ma_short) > 0 and len(ma_long) > 0:
                strength = abs((ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1] * 100)
                return float(strength) if not pd.isna(strength) else 1.0
            return 1.0
            
        except Exception:
            return 1.0
    
    def calculate_market_sentiment(self, taiex_data):
        """計算市場整體情緒指標"""
        try:
            # 收盤價相對於最高價的位置
            recent_data = taiex_data.tail(5)
            high_low_pos = (recent_data['Close'] - recent_data['Low']) / (recent_data['High'] - recent_data['Low'])
            sentiment = float(high_low_pos.mean())
            return sentiment if not pd.isna(sentiment) else 0.5
            
        except Exception:
            return 0.5
    
    def calculate_volatility_change(self, prices):
        """計算波動率變化趨勢"""
        try:
            # 比較最近5天與前5天的波動率
            recent_vol = prices.tail(5).pct_change().std()
            previous_vol = prices.tail(10).head(5).pct_change().std()
            
            if previous_vol > 0:
                vol_change = (recent_vol - previous_vol) / previous_vol
                return float(vol_change) if not pd.isna(vol_change) else 0.0
            return 0.0
            
        except Exception:
            return 0.0
    
    def generate_enhanced_training_data(self, num_samples=1500):
        """生成增強版訓練數據"""
        np.random.seed(42)
        training_data = []
        
        # 定義不同市場情境
        market_scenarios = [
            # 牛市情境
            {'volatility_range': (0.5, 1.5), 'trend_range': (2.0, 5.0), 'rsi_range': (60, 80), 'weights': [0.5, 0.2, 0.3]},
            # 熊市情境  
            {'volatility_range': (2.0, 4.0), 'trend_range': (1.0, 3.0), 'rsi_range': (20, 40), 'weights': [0.3, 0.5, 0.2]},
            # 震盪市情境
            {'volatility_range': (1.5, 2.5), 'trend_range': (0.5, 2.0), 'rsi_range': (40, 60), 'weights': [0.3, 0.3, 0.4]},
            # 高波動情境
            {'volatility_range': (3.0, 5.0), 'trend_range': (1.0, 4.0), 'rsi_range': (30, 70), 'weights': [0.2, 0.6, 0.2]},
            # 低波動情境
            {'volatility_range': (0.3, 1.0), 'trend_range': (0.5, 2.0), 'rsi_range': (45, 55), 'weights': [0.6, 0.2, 0.2]}
        ]
        
        for scenario in market_scenarios:
            for _ in range(num_samples // len(market_scenarios)):
                # 生成符合情境的特徵
                features = {
                    'volatility': np.random.uniform(*scenario['volatility_range']),
                    'volume_trend': np.random.uniform(0.7, 1.5),
                    'price_momentum': np.random.uniform(-5.0, 5.0),
                    'time_factor': np.random.choice([0.2, 0.6, 0.8, 0.9, 1.0]),
                    'market_rsi': np.random.uniform(*scenario['rsi_range']),
                    'trend_strength': np.random.uniform(*scenario['trend_range']),
                    'market_sentiment': np.random.uniform(0.2, 0.8),
                    'volatility_change': np.random.uniform(-0.5, 0.5)
                }
                
                # 基於情境計算理想權重
                ideal_weights = self.calculate_scenario_weights(features, scenario['weights'])
                
                training_data.append({
                    'features': features,
                    'weights': ideal_weights
                })
        
        return training_data
    
    def calculate_scenario_weights(self, features, base_weights):
        """根據具體特徵微調權重"""
        ai_weight, technical_weight, sentiment_weight = base_weights
        
        # 根據波動率調整
        if features['volatility'] > 2.5:
            technical_weight += 0.1
            ai_weight -= 0.05
            sentiment_weight -= 0.05
        
        # 根據趨勢強度調整
        if features['trend_strength'] > 3.0:
            ai_weight += 0.1
            technical_weight -= 0.05
            sentiment_weight -= 0.05
        
        # 根據時間因子調整
        if features['time_factor'] >= 0.9:  # 開盤或盤中前半
            sentiment_weight += 0.05
            ai_weight -= 0.025
            technical_weight -= 0.025
        
        # 根據市場情緒調整
        if features['market_sentiment'] > 0.7 or features['market_sentiment'] < 0.3:
            sentiment_weight += 0.05
            ai_weight -= 0.025
            technical_weight -= 0.025
        
        # 確保權重總和為1
        total = ai_weight + technical_weight + sentiment_weight
        return [ai_weight/total, technical_weight/total, sentiment_weight/total]
    
    def train_enhanced_model(self):
        """訓練增強版模型"""
        print("🧠 訓練增強版動態權重模型...")
        
        # 生成增強版訓練數據
        training_data = self.generate_enhanced_training_data()
        
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
                data['features']['trend_strength'],
                data['features']['market_sentiment'],
                data['features']['volatility_change']
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
        
        # 訓練模型
        self.model.fit(X_train, y_train)
        
        # 評估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ 增強版模型訓練完成!")
        print(f"📊 訓練集準確率: {train_score:.4f}")
        print(f"📊 測試集準確率: {test_score:.4f}")
        
        # 特徵重要性分析
        feature_names = ['volatility', 'volume_trend', 'price_momentum', 'time_factor', 
                        'market_rsi', 'trend_strength', 'market_sentiment', 'volatility_change']
        importances = self.model.feature_importances_
        
        print("📈 特徵重要性排序:")
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {importance:.4f}")
        
        self.is_trained = True
        self.save_model()
        
        return train_score, test_score
    
    def predict_weights_enhanced(self, market_features=None):
        """增強版權重預測"""
        if not self.is_trained:
            print("⚠️ 模型尚未訓練，開始訓練...")
            self.train_enhanced_model()
        
        if market_features is None:
            market_features = self.collect_market_features_safe()
        
        if market_features is None:
            print("⚠️ 無法獲取市場特徵，使用預設權重")
            return [0.4, 0.3, 0.3]
        
        try:
            # 準備特徵向量
            features_vector = np.array([[
                market_features['volatility'],
                market_features['volume_trend'],
                market_features['price_momentum'],
                market_features['time_factor'],
                market_features['market_rsi'],
                market_features['trend_strength'],
                market_features['market_sentiment'],
                market_features['volatility_change']
            ]])
            
            # 標準化
            features_scaled = self.scaler.transform(features_vector)
            
            # 預測權重
            predicted_weights = self.model.predict(features_scaled)[0]
            
            # 確保權重為正且總和為1
            predicted_weights = np.clip(predicted_weights, 0.05, 0.85)
            predicted_weights = predicted_weights / predicted_weights.sum()
            
            return predicted_weights.tolist()
            
        except Exception as e:
            print(f"⚠️ 權重預測失敗: {e}")
            return [0.4, 0.3, 0.3]
    
    def save_model(self):
        """儲存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'version': '2.0'
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 增強版模型已儲存至: {self.model_path}")
    
    def load_model(self):
        """載入模型"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            version = model_data.get('version', '1.0')
            print(f"📥 增強版模型已載入: {self.model_path} (v{version})")
            
        except Exception as e:
            print(f"⚠️ 載入模型失敗: {e}")
            self.is_trained = False

def test_improved_system():
    """測試改進版系統"""
    print("🧪 測試改進版動態權重學習系統")
    
    # 初始化改進版學習器
    learner = ImprovedDynamicWeightLearner()
    
    # 訓練增強版模型
    learner.train_enhanced_model()
    
    # 測試不同市場情境
    test_scenarios = [
        {"name": "高波動熊市", "volatility": 3.5, "market_rsi": 25, "trend_strength": 1.5},
        {"name": "低波動牛市", "volatility": 0.8, "market_rsi": 75, "trend_strength": 4.0},
        {"name": "震盪整理", "volatility": 1.8, "market_rsi": 50, "trend_strength": 1.0}
    ]
    
    for scenario in test_scenarios:
        print(f"\n📊 測試情境: {scenario['name']}")
        
        # 模擬市場特徵
        mock_features = {
            'volatility': scenario['volatility'],
            'volume_trend': 1.1,
            'price_momentum': 0.5,
            'time_factor': 1.0,
            'market_rsi': scenario['market_rsi'],
            'trend_strength': scenario['trend_strength'],
            'market_sentiment': 0.6,
            'volatility_change': 0.1
        }
        
        weights = learner.predict_weights_enhanced(mock_features)
        
        print(f"  動態權重分配:")
        print(f"    AI權重: {weights[0]:.3f} ({weights[0]*100:.1f}%)")
        print(f"    技術權重: {weights[1]:.3f} ({weights[1]*100:.1f}%)")
        print(f"    情緒權重: {weights[2]:.3f} ({weights[2]*100:.1f}%)")
    
    return learner

if __name__ == "__main__":
    test_improved_system()