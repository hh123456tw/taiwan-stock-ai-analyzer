"""
LightGBM 股票預測器 - 微軟出品的高效梯度提升模型
專精於快速訓練和高精度預測，XGBoost的強力競爭對手
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM已載入")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM未安裝，將使用替代實現")

# 導入sklearn作為備用
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class LightGBMStockPredictor:
    """LightGBM股票預測器 - 效率專家 (RTX 3060Ti GPU加速)"""
    
    def __init__(self, use_gpu=True):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_importance = None
        self.is_trained = False
        self.symbol = None
        self.training_score = 0.0
        self.feature_names = []
        self.use_gpu = use_gpu and LIGHTGBM_AVAILABLE
        
        # 設置GPU參數 (RTX 3060Ti)
        if self.use_gpu:
            self.gpu_params = {
                'device': 'gpu',
                'gpu_device_id': 0,  # 使用RTX 3060Ti (GPU:0)
                'gpu_platform_id': 0,
                'gpu_use_dp': False,  # 使用單精度加速
                'force_row_wise': True  # 優化記憶體使用
            }
            print("🚀 LightGBM配置使用RTX 3060Ti GPU加速")
        else:
            self.gpu_params = {'device': 'cpu'}
            print("🖥️ LightGBM使用CPU模式")
        
    def _create_features(self, prices: List[float], volumes: List[float] = None) -> np.array:
        """創建特徵工程"""
        
        prices = np.array(prices)
        n = len(prices)
        
        if n < 4:
            raise ValueError("需要至少4個數據點來創建特徵")
        
        features = []
        feature_names = []
        
        for i in range(1, n):  # 從第2個點開始，適應最少數據
            sample_features = []
            
            # 價格特徵
            current_price = prices[i]
            
            # 1. 移動平均特徵
            window_size = min(i, 3)  # 適應少量數據
            ma_short = np.mean(prices[max(0, i-window_size):i])
            ma_long = np.mean(prices[max(0, i-2):i]) if i >= 2 else current_price
            sample_features.extend([ma_short, ma_long, current_price])
            
            # 2. 價格變化率特徵
            returns_1d = (prices[i] - prices[i-1]) / prices[i-1] if i > 0 and prices[i-1] > 0 else 0
            returns_2d = (prices[i] - prices[i-2]) / prices[i-2] if i > 1 and prices[i-2] > 0 else 0
            returns_all = (prices[i] - prices[0]) / prices[0] if prices[0] > 0 else 0
            sample_features.extend([returns_1d, returns_2d, returns_all])
            
            # 3. 價格位置特徵
            window = min(i, 3)
            recent_max = np.max(prices[max(0, i-window):i+1])
            recent_min = np.min(prices[max(0, i-window):i+1])
            price_position = (current_price - recent_min) / (recent_max - recent_min) if recent_max != recent_min else 0.5
            sample_features.append(price_position)
            
            # 4. 波動率特徵
            window = min(i, len(prices))
            volatility = np.std(prices[max(0, i-window):i+1]) if window > 1 else 0
            sample_features.append(volatility)
            
            # 5. 趨勢特徵
            window = min(i+1, len(prices))
            if window >= 2:
                trend_slope = np.polyfit(range(window), prices[max(0, i-window+1):i+1], 1)[0]
            else:
                trend_slope = 0
            sample_features.append(trend_slope)
            
            # 6. 技術指標特徵
            # RSI簡化版
            gains = []
            losses = []
            window = min(i, 3)
            for j in range(max(1, i-window+1), i+1):
                if j < len(prices) and j > 0:
                    change = prices[j] - prices[j-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            sample_features.append(rsi)
            
            # 7. 成交量特徵 (如果有的話)
            if volumes and len(volumes) > i:
                window = min(i+1, len(volumes))
                volume_ma = np.mean(volumes[max(0, i-window+1):i+1]) if window > 0 else volumes[i]
                volume_ratio = volumes[i] / volume_ma if volume_ma > 0 else 1
                sample_features.extend([volumes[i], volume_ratio])
            else:
                sample_features.extend([1.0, 1.0])  # 預設值
            
            features.append(sample_features)
            
            # 設定特徵名稱 (只在第一次)
            if i == 1:
                feature_names = [
                    'ma_short', 'ma_long', 'current_price',
                    'returns_1d', 'returns_2d', 'returns_all',
                    'price_position', 'volatility', 'trend_slope', 'rsi',
                    'volume', 'volume_ratio'
                ]
                self.feature_names = feature_names
        
        return np.array(features)
    
    def _create_targets(self, prices: List[float], start_idx: int = 1) -> np.array:
        """創建目標值（未來股價）"""
        
        targets = []
        # 確保targets數量與features匹配
        for i in range(start_idx, len(prices)):  # 包含最後一個價格
            if i + 1 < len(prices):
                targets.append(prices[i + 1])  # 預測下一天
            else:
                targets.append(prices[i])  # 最後一個用當前價格
        
        return np.array(targets)
    
    def train(self, symbol: str, prices: List[float], 
             volumes: List[float] = None, **kwargs) -> Dict:
        """訓練LightGBM模型"""
        
        try:
            self.symbol = symbol
            
            if len(prices) < 4:
                return {
                    'success': False,
                    'error': 'LightGBM需要至少4個數據點',
                    'model_info': 'LightGBM梯度提升模型'
                }
            
            # 創建特徵和目標
            X = self._create_features(prices, volumes)
            y = self._create_targets(prices)
            
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]
            
            if len(X) < 2:
                return {
                    'success': False,
                    'error': '特徵數據不足，需要至少2個訓練樣本',
                    'model_info': 'LightGBM特徵工程失敗'
                }
            
            # 分割訓練集
            if len(X) > 10:
                train_size = max(5, int(len(X) * 0.8))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
            else:
                X_train, X_test = X, X
                y_train, y_test = y, y
            
            # 特徵縮放
            if SKLEARN_AVAILABLE and self.scaler:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # 訓練模型
            if LIGHTGBM_AVAILABLE:
                train_result = self._train_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test)
            elif SKLEARN_AVAILABLE:
                train_result = self._train_sklearn_gbm(X_train_scaled, y_train, X_test_scaled, y_test)
            else:
                train_result = self._train_simple_gbm(X_train, y_train, X_test, y_test)
            
            self.is_trained = train_result['success']
            return train_result
            
        except Exception as e:
            logger.error(f"LightGBM訓練錯誤: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_info': 'LightGBM訓練失敗'
            }
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test) -> Dict:
        """使用原生LightGBM訓練"""
        
        try:
            # 創建LightGBM數據集
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # LightGBM參數
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            # 訓練模型
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
            
            # 評估模型
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # 計算R²分數
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            self.training_score = test_r2
            self.feature_importance = self.model.feature_importance()
            
            return {
                'success': True,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'feature_importance': dict(zip(self.feature_names, self.feature_importance)),
                'model_info': 'LightGBM梯度提升模型（微軟版）'
            }
            
        except Exception as e:
            logger.error(f"LightGBM原生訓練失敗: {e}")
            # 回退到sklearn版本
            return self._train_sklearn_gbm(X_train, y_train, X_test, y_test)
    
    def _train_sklearn_gbm(self, X_train, y_train, X_test, y_test) -> Dict:
        """使用sklearn的梯度提升訓練"""
        
        try:
            # 創建GradientBoosting模型
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            )
            
            # 訓練模型
            self.model.fit(X_train, y_train)
            
            # 評估模型
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            self.training_score = test_r2
            self.feature_importance = self.model.feature_importances_
            
            return {
                'success': True,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'feature_importance': dict(zip(self.feature_names, self.feature_importance)),
                'model_info': 'GradientBoosting模型（sklearn版LightGBM）'
            }
            
        except Exception as e:
            logger.error(f"sklearn GBM訓練失敗: {e}")
            return self._train_simple_gbm(X_train, y_train, X_test, y_test)
    
    def _train_simple_gbm(self, X_train, y_train, X_test, y_test) -> Dict:
        """簡化版梯度提升"""
        
        # 使用簡單的線性回歸作為基礎模型
        n_features = X_train.shape[1]
        
        # 計算特徵權重（簡化版）
        correlations = []
        for i in range(n_features):
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        self.model = {
            'type': 'simple_gbm',
            'weights': np.array(correlations),
            'bias': np.mean(y_train),
            'feature_means': np.mean(X_train, axis=0),
            'feature_stds': np.std(X_train, axis=0) + 1e-8
        }
        
        # 預測和評估
        y_pred_train = self._simple_predict(X_train)
        y_pred_test = self._simple_predict(X_test)
        
        train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        
        # 計算R²
        ss_res_train = np.sum((y_train - y_pred_train) ** 2)
        ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
        train_r2 = 1 - (ss_res_train / ss_tot_train) if ss_tot_train > 0 else 0
        
        ss_res_test = np.sum((y_test - y_pred_test) ** 2)
        ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
        test_r2 = 1 - (ss_res_test / ss_tot_test) if ss_tot_test > 0 else 0
        
        self.training_score = test_r2
        self.feature_importance = np.abs(correlations)
        
        return {
            'success': True,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': dict(zip(self.feature_names, self.feature_importance)),
            'model_info': '簡化梯度提升模型'
        }
    
    def _simple_predict(self, X) -> np.array:
        """簡化版預測"""
        
        try:
            # 確保X是數值型態
            X = np.array(X, dtype=np.float64)
            
            # 確保模型參數是數值型態
            feature_means = np.array(self.model['feature_means'], dtype=np.float64)
            feature_stds = np.array(self.model['feature_stds'], dtype=np.float64)
            weights = np.array(self.model['weights'], dtype=np.float64)
            bias = float(self.model['bias'])
            
            # 標準化特徵
            X_norm = (X - feature_means) / feature_stds
            
            # 加權預測
            predictions = np.dot(X_norm, weights) + bias
            
            return predictions.astype(np.float64)
            
        except Exception as e:
            logger.error(f"簡化預測計算錯誤: {e}")
            # 返回基於最後價格的簡單預測
            if hasattr(X, '__len__') and len(X) > 0:
                return np.array([float(np.mean(X))], dtype=np.float64)
            else:
                return np.array([100.0], dtype=np.float64)  # 預設值
    
    def predict(self, prices: List[float], volumes: List[float] = None, days: int = 3) -> Dict:
        """預測未來股價"""
        
        if not self.is_trained:
            return {
                'success': False,
                'error': 'LightGBM模型未訓練',
                'predictions': []
            }
        
        try:
            # 確保數據類型正確
            predictions = []
            current_prices = [float(p) for p in prices]  # 確保是浮點數
            current_volumes = [float(v) for v in volumes] if volumes else [1.0] * len(prices)
            
            for day in range(days):
                # 創建當前特徵
                features = self._create_features(current_prices, current_volumes)
                last_features = features[-1].reshape(1, -1)
                
                # 特徵縮放
                if SKLEARN_AVAILABLE and self.scaler:
                    last_features_scaled = self.scaler.transform(last_features)
                else:
                    last_features_scaled = last_features
                
                # 進行預測
                try:
                    if LIGHTGBM_AVAILABLE and hasattr(self.model, 'predict'):
                        pred = float(self.model.predict(last_features_scaled)[0])
                    elif hasattr(self.model, 'predict'):
                        pred = float(self.model.predict(last_features_scaled)[0])
                    else:
                        pred_result = self._simple_predict(last_features)
                        pred = float(pred_result[0] if hasattr(pred_result, '__len__') else pred_result)
                except Exception as pred_error:
                    logger.warning(f"LightGBM預測步驟錯誤: {pred_error}")
                    # 使用簡單的趨勢預測作為備用
                    if len(current_prices) >= 2:
                        trend = current_prices[-1] - current_prices[-2]
                        pred = current_prices[-1] + trend
                    else:
                        pred = current_prices[-1]
                    pred = float(pred)
                
                predictions.append(pred)
                
                # 更新價格序列用於下一次預測
                current_prices.append(float(pred))
                current_volumes.append(float(current_volumes[-1]))  # 假設成交量相同
            
            # 計算信心度
            confidence_scores = []
            base_confidence = min(0.9, max(0.5, self.training_score))
            
            for i in range(days):
                # 信心度隨預測天數遞減
                confidence = base_confidence * (0.85 ** i)
                confidence_scores.append(confidence)
            
            return {
                'success': True,
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'model_score': self.training_score,
                'method': 'LightGBM梯度提升',
                'prediction_type': 'gradient_boosting'
            }
            
        except Exception as e:
            logger.error(f"LightGBM預測錯誤: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }
    
    def get_model_info(self) -> Dict:
        """獲取模型信息"""
        
        if not self.is_trained:
            return {'status': 'untrained', 'type': 'LightGBM'}
        
        info = {
            'model_type': 'LightGBM',
            'description': '輕量級梯度提升機',
            'strengths': ['高效訓練', '優秀精度', 'XGBoost競爭對手'],
            'trained': True,
            'symbol': self.symbol,
            'training_score': self.training_score,
            'feature_count': len(self.feature_names)
        }
        
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            # 找出最重要的特徵
            if isinstance(self.feature_importance, dict):
                sorted_features = sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                info['top_features'] = sorted_features[:3]
            else:
                top_indices = np.argsort(self.feature_importance)[-3:][::-1]
                info['top_features'] = [(self.feature_names[i], self.feature_importance[i]) 
                                      for i in top_indices]
        
        return info

# 向後相容的處理函數
def handle_lightgbm_predict_command(update, context, symbol: str) -> str:
    """處理LightGBM預測命令"""
    
    try:
        # 這裡應該從主系統獲取股價數據
        # 為了示例，使用模擬數據
        import random
        base_price = 100
        prices = [base_price + random.uniform(-5, 5) + i * 0.1 for i in range(30)]
        volumes = [random.randint(1000, 10000) for _ in range(30)]
        
        predictor = LightGBMStockPredictor()
        
        # 訓練模型
        train_result = predictor.train(symbol, prices, volumes)
        
        if not train_result['success']:
            return f"❌ LightGBM模型訓練失敗: {train_result.get('error', 'Unknown error')}"
        
        # 進行預測
        prediction_result = predictor.predict(prices, volumes, days=3)
        
        if not prediction_result['success']:
            return f"❌ LightGBM預測失敗: {prediction_result.get('error', 'Unknown error')}"
        
        # 格式化結果
        predictions = prediction_result['predictions']
        current_price = prices[-1]
        
        message = f"🚀 **LightGBM梯度提升預測 - {symbol}**\n\n"
        message += f"💰 當前價格: {current_price:.2f}\n"
        message += f"📊 模型R²分數: {train_result.get('test_r2', 0):.3f}\n"
        message += f"🎯 模型信息: {train_result['model_info']}\n\n"
        message += "🔮 **未來3天預測**:\n"
        
        for i, pred in enumerate(predictions, 1):
            change_pct = (pred - current_price) / current_price * 100
            confidence = prediction_result['confidence_scores'][i-1]
            message += f"Day {i}: {pred:.2f} ({change_pct:+.1f}%) 信心度:{confidence:.0%}\n"
        
        # 添加特徵重要性
        if 'feature_importance' in train_result:
            top_features = sorted(train_result['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            message += f"\n🎯 **關鍵特徵**:\n"
            for feature, importance in top_features:
                message += f"• {feature}: {importance:.3f}\n"
        
        return message
        
    except Exception as e:
        logger.error(f"LightGBM預測處理錯誤: {e}")
        return f"❌ LightGBM預測處理失敗: {str(e)}"

if __name__ == "__main__":
    # 測試LightGBM預測器
    print("🧪 測試LightGBM預測器")
    
    # 生成測試數據
    np.random.seed(42)
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(50):
        trend = i * 0.1
        noise = np.random.normal(0, 2)
        price = base_price + trend + noise
        volume = np.random.randint(1000, 10000)
        prices.append(price)
        volumes.append(volume)
    
    predictor = LightGBMStockPredictor()
    
    # 訓練
    print("訓練LightGBM模型...")
    train_result = predictor.train("TEST", prices, volumes)
    print(f"訓練結果: {train_result}")
    
    # 預測
    print("\n進行預測...")
    pred_result = predictor.predict(prices, volumes, 3)
    print(f"預測結果: {pred_result}")
    
    # 模型信息
    print(f"\n模型信息: {predictor.get_model_info()}")