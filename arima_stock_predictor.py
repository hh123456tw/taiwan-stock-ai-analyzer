"""
ARIMA 股票預測器 - 時間序列分析經典模型
專精於捕捉股價的時間趨勢和季節性模式
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入必要套件
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels未安裝，ARIMA模型將使用簡化版本")

logger = logging.getLogger(__name__)

class ARIMAStockPredictor:
    """ARIMA股票預測器 - 時間序列分析專家"""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.is_trained = False
        self.symbol = None
        self.last_training_data = None
        
    def _check_stationarity(self, data: np.array) -> bool:
        """檢查時間序列平穩性"""
        if not STATSMODELS_AVAILABLE:
            # 簡化版平穩性檢查：看變化率
            returns = np.diff(data) / data[:-1]
            return np.std(returns) < 0.1  # 簡單標準
            
        try:
            result = adfuller(data)
            return result[1] < 0.05  # p-value < 0.05 表示平穩
        except:
            return False
    
    def _make_stationary(self, data: np.array) -> Tuple[np.array, int]:
        """使時間序列平穩化"""
        original_data = data.copy()
        diff_order = 0
        
        # 最多差分2次
        for i in range(3):
            if self._check_stationarity(data):
                return data, diff_order
            data = np.diff(data)
            diff_order += 1
            
        return data, diff_order
    
    def _find_best_params(self, data: np.array) -> Tuple[int, int, int]:
        """尋找最佳ARIMA參數 (p,d,q)"""
        if not STATSMODELS_AVAILABLE:
            # 簡化版：使用經驗值
            return (2, 1, 2)
        
        best_aic = float('inf')
        best_params = (1, 1, 1)
        
        # 嘗試不同參數組合
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                    except:
                        continue
                        
        return best_params
    
    def train(self, symbol: str, prices: List[float], 
             volumes: List[float] = None, **kwargs) -> Dict:
        """訓練ARIMA模型"""
        
        try:
            self.symbol = symbol
            prices_array = np.array(prices)
            
            if len(prices) < 15:
                return {
                    'success': False,
                    'error': f'ARIMA需要至少15個數據點才能有效分析時間序列，目前只有{len(prices)}點',
                    'model_info': 'ARIMA時間序列模型（需要足夠數據保證品質）'
                }
            
            self.last_training_data = prices_array.copy()
            
            if STATSMODELS_AVAILABLE:
                # 使用完整ARIMA
                best_params = self._find_best_params(prices_array)
                
                try:
                    self.model = ARIMA(prices_array, order=best_params)
                    self.fitted_model = self.model.fit()
                    self.is_trained = True
                    
                    # 計算訓練準確度
                    fitted_values = self.fitted_model.fittedvalues
                    # 確保兩個陣列長度一致
                    min_len = min(len(fitted_values), len(prices_array))
                    actual = prices_array[-min_len:]
                    fitted = fitted_values[-min_len:]
                    
                    mse = np.mean((actual - fitted) ** 2)
                    rmse = np.sqrt(mse)
                    accuracy = max(0, 1 - rmse / np.std(prices_array))
                    
                    return {
                        'success': True,
                        'model_params': best_params,
                        'training_accuracy': accuracy,
                        'rmse': rmse,
                        'aic': self.fitted_model.aic,
                        'model_info': f'ARIMA{best_params} 時間序列模型'
                    }
                    
                except Exception as e:
                    logger.warning(f"ARIMA擬合失敗，使用簡化模型: {e}")
                    return self._train_simple_arima(prices_array)
            else:
                return self._train_simple_arima(prices_array)
                
        except Exception as e:
            logger.error(f"ARIMA訓練錯誤: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_info': 'ARIMA訓練失敗'
            }
    
    def _train_simple_arima(self, prices: np.array) -> Dict:
        """簡化版ARIMA訓練（當statsmodels不可用時）"""
        
        # 確保prices是float類型的numpy數組
        prices = np.array(prices, dtype=np.float64)
        
        # 計算移動平均和趨勢
        window = min(5, len(prices) // 4)
        ma = pd.Series(prices).rolling(window=window).mean().dropna()
        
        # 計算一階差分
        diff1 = np.diff(prices)
        
        # 計算自相關係數（簡化版） - 增加錯誤處理
        try:
            autocorr = np.corrcoef(diff1[:-1], diff1[1:])[0, 1]
            autocorr_value = float(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            autocorr_value = 0.0
        
        # 存儲模型參數（確保數據類型正確）
        ma_values = ma.iloc[-window:].values if len(ma) >= window else ma.values
        trend_value = float(np.polyfit(range(len(prices)), prices, 1)[0])
        
        self.model = {
            'type': 'simple_arima',
            'ma': [float(x) for x in ma_values] if len(ma_values) > 0 else [float(prices[-1])],
            'trend': trend_value,
            'autocorr': autocorr_value,
            'recent_changes': [float(x) for x in diff1[-5:]] if len(diff1) >= 5 else [0.0],
            'volatility': float(np.std(diff1)) if len(diff1) > 0 else 0.0
        }
        
        self.is_trained = True
        
        # 計算簡化準確度
        predicted = self._simple_predict_in_sample(prices)
        mse = np.mean((prices[1:] - predicted) ** 2)
        accuracy = max(0, 1 - np.sqrt(mse) / np.std(prices))
        
        return {
            'success': True,
            'model_params': 'Simple ARIMA',
            'training_accuracy': accuracy,
            'autocorr': autocorr,
            'trend': self.model['trend'],
            'model_info': '簡化ARIMA模型（時間序列分析）'
        }
    
    def _simple_predict_in_sample(self, prices: np.array) -> np.array:
        """簡化版樣本內預測"""
        predictions = []
        
        # 確保prices是float類型的numpy數組
        prices = np.array(prices, dtype=np.float64)
        
        for i in range(1, len(prices)):
            # 趨勢分量 - 確保類型轉換
            trend_pred = float(prices[i-1]) + float(self.model['trend'])
            
            # 自相關分量
            if i > 1:
                autocorr_pred = float(self.model['autocorr']) * (float(prices[i-1]) - float(prices[i-2]))
            else:
                autocorr_pred = 0.0
            
            # 移動平均分量
            window = min(i, len(self.model['ma']))
            if window > 0:
                ma_pred = float(np.mean(prices[max(0, i-window):i]))
            else:
                ma_pred = float(prices[i-1])
            
            # 組合預測
            pred = 0.4 * float(trend_pred) + 0.3 * float(ma_pred) + 0.3 * float(autocorr_pred)
            predictions.append(float(pred))
            
        return np.array(predictions, dtype=np.float64)
    
    def predict(self, days: int = 3) -> Dict:
        """預測未來股價"""
        
        # 確保days是整數類型
        try:
            days = int(days) if days is not None else 3
        except (ValueError, TypeError):
            days = 3
        
        if not self.is_trained:
            return {
                'success': False,
                'error': 'ARIMA模型未訓練',
                'predictions': []
            }
        
        try:
            if STATSMODELS_AVAILABLE and hasattr(self.fitted_model, 'forecast'):
                try:
                    # 使用完整ARIMA預測
                    forecast = self.fitted_model.forecast(steps=days)
                    
                    # 確保forecast是數值型態
                    if hasattr(forecast, 'values'):
                        predictions = forecast.values.tolist()
                    elif hasattr(forecast, 'tolist'):
                        predictions = forecast.tolist()
                    else:
                        predictions = list(forecast)
                    
                    # 嘗試獲取信心區間
                    try:
                        forecast_obj = self.fitted_model.get_forecast(steps=days)
                        confidence_intervals = forecast_obj.conf_int()
                        if hasattr(confidence_intervals, 'iloc'):
                            lower_bounds = confidence_intervals.iloc[:, 0].tolist()
                            upper_bounds = confidence_intervals.iloc[:, 1].tolist()
                        else:
                            lower_bounds = [p * 0.95 for p in predictions]
                            upper_bounds = [p * 1.05 for p in predictions]
                    except:
                        lower_bounds = [p * 0.95 for p in predictions]
                        upper_bounds = [p * 1.05 for p in predictions]
                        
                except Exception as forecast_error:
                    logger.warning(f"ARIMA完整預測失敗，使用簡化方法: {forecast_error}")
                    # 回退到簡化預測
                    predictions, confidence_scores = self._simple_predict(days)
                    lower_bounds = [p * 0.95 for p in predictions]
                    upper_bounds = [p * 1.05 for p in predictions]
                    
                    return {
                        'success': True,
                        'predictions': predictions,
                        'confidence_scores': confidence_scores,
                        'lower_bounds': lower_bounds,
                        'upper_bounds': upper_bounds,
                        'method': 'ARIMA簡化預測',
                        'prediction_type': 'time_series'
                    }
                
                # 計算信心度
                confidence_scores = []
                for i in range(days):
                    interval_width = upper_bounds[i] - lower_bounds[i]
                    relative_width = interval_width / predictions[i]
                    confidence = max(0.1, 1 - relative_width)
                    confidence_scores.append(confidence)
                
            else:
                # 使用簡化版預測
                predictions, confidence_scores = self._simple_predict(days)
                lower_bounds = [p * 0.95 for p in predictions]
                upper_bounds = [p * 1.05 for p in predictions]
            
            return {
                'success': True,
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'method': 'ARIMA時間序列分析',
                'prediction_type': 'time_series'
            }
            
        except Exception as e:
            logger.error(f"ARIMA預測錯誤: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }
    
    def _simple_predict(self, days: int) -> Tuple[List[float], List[float]]:
        """簡化版預測"""
        
        try:
            # 確保days是整數類型
            days = int(days) if days is not None else 3
            
            # 確保數據類型正確
            current_price = float(self.last_training_data[-1])
            predictions = []
            confidence_scores = []
            
            # 確保模型參數是數值類型
            trend = float(self.model['trend']) if self.model['trend'] is not None else 0.0
            autocorr = float(self.model['autocorr']) if self.model['autocorr'] is not None else 0.0
            
            # 處理recent_changes
            if self.model['recent_changes'] is not None and len(self.model['recent_changes']) > 0:
                recent_changes = [float(x) for x in self.model['recent_changes']]
                recent_change = float(np.mean(recent_changes))
            else:
                recent_change = 0.0
                
            # 處理移動平均
            if self.model['ma'] is not None and len(self.model['ma']) > 0:
                ma_values = [float(x) for x in self.model['ma']]
                ma_pred = float(np.mean(ma_values))
            else:
                ma_pred = current_price
            
            for day in range(1, days + 1):
                # 趨勢預測
                trend_pred = current_price + trend * float(day)
                
                # 自相關預測
                autocorr_pred = current_price + recent_change * autocorr * float(day)
                
                # 組合預測（隨時間衰減自相關影響）
                decay_factor = 0.8 ** (day - 1)
                pred = (0.4 * float(trend_pred) + 
                       0.3 * float(ma_pred) + 
                       0.3 * float(autocorr_pred) * decay_factor)
                
                predictions.append(float(pred))
                
                # 信心度隨預測天數遞減
                base_confidence = 0.8
                confidence = base_confidence * (0.9 ** (day - 1))
                confidence_scores.append(float(confidence))
                
        except Exception as e:
            logger.error(f"簡化ARIMA預測錯誤: {e}")
            # 使用最簡單的線性預測作為備用
            # 確保days是整數類型
            days = int(days) if days is not None else 3
            
            current_price = float(self.last_training_data[-1])
            predictions = []
            confidence_scores = []
            
            # 簡單的線性趨勢預測
            if len(self.last_training_data) >= 2:
                trend = float(self.last_training_data[-1]) - float(self.last_training_data[-2])
            else:
                trend = 0.0
                
            for day in range(1, int(days) + 1):
                pred = current_price + trend * float(day)
                predictions.append(float(pred))
                confidence_scores.append(0.6)  # 基礎信心度
        
        return predictions, confidence_scores
    
    def get_model_info(self) -> Dict:
        """獲取模型信息"""
        
        if not self.is_trained:
            return {'status': 'untrained', 'type': 'ARIMA'}
        
        info = {
            'model_type': 'ARIMA',
            'description': '自回歸積分滑動平均模型',
            'strengths': ['時間序列專精', '趨勢捕捉', '經典金融模型'],
            'trained': True,
            'symbol': self.symbol
        }
        
        if STATSMODELS_AVAILABLE and hasattr(self.fitted_model, 'aic'):
            info.update({
                'aic': self.fitted_model.aic,
                'implementation': 'Full ARIMA'
            })
        else:
            info.update({
                'trend': self.model.get('trend', 0),
                'autocorr': self.model.get('autocorr', 0),
                'implementation': 'Simple ARIMA'
            })
        
        return info

# 向後相容的處理函數
def handle_arima_predict_command(update, context, symbol: str) -> str:
    """處理ARIMA預測命令"""
    
    try:
        # 這裡應該從主系統獲取股價數據
        # 為了示例，使用模擬數據
        import random
        base_price = 100
        prices = [base_price + random.uniform(-5, 5) + i * 0.1 for i in range(30)]
        
        predictor = ARIMAStockPredictor()
        
        # 訓練模型
        train_result = predictor.train(symbol, prices)
        
        if not train_result['success']:
            return f"❌ ARIMA模型訓練失敗: {train_result.get('error', 'Unknown error')}"
        
        # 進行預測
        prediction_result = predictor.predict(days=3)
        
        if not prediction_result['success']:
            return f"❌ ARIMA預測失敗: {prediction_result.get('error', 'Unknown error')}"
        
        # 格式化結果
        predictions = prediction_result['predictions']
        current_price = prices[-1]
        
        message = f"📈 **ARIMA時間序列預測 - {symbol}**\n\n"
        message += f"💰 當前價格: {current_price:.2f}\n"
        message += f"📊 模型信息: {train_result['model_info']}\n\n"
        message += "🔮 **未來3天預測**:\n"
        
        for i, pred in enumerate(predictions, 1):
            change_pct = (pred - current_price) / current_price * 100
            confidence = prediction_result['confidence_scores'][i-1]
            message += f"Day {i}: {pred:.2f} ({change_pct:+.1f}%) 信心度:{confidence:.0%}\n"
        
        return message
        
    except Exception as e:
        logger.error(f"ARIMA預測處理錯誤: {e}")
        return f"❌ ARIMA預測處理失敗: {str(e)}"

if __name__ == "__main__":
    import argparse
    import sys
    import os
    import pickle
    
    parser = argparse.ArgumentParser(description='ARIMA Stock Predictor')
    parser.add_argument('--symbol', type=str, help='Stock symbol')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--days', type=int, default=3, help='Days to predict')
    
    args = parser.parse_args()
    
    if args.symbol and args.train:
        # 訓練模式
        print(f"🧪 開始訓練 {args.symbol} ARIMA模型...")
        
        try:
            # 導入數據獲取功能
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            # from data_fetcher import DataFetcher
            
            # 使用統一數據管理器獲取資料
            from unified_data_manager import UnifiedDataManager
            data_manager = UnifiedDataManager()
            
            # 獲取訓練用歷史數據
            df = data_manager.get_training_data(args.symbol, days=365)
            
            if df.empty:
                print(f"❌ 無法獲取 {args.symbol} 的股價數據")
                sys.exit(1)
            
            prices = df['close'].tolist()
            
            if len(prices) < 15:
                print(f"❌ 數據不足: {len(prices)} < 15")
                sys.exit(1)
            
            # 創建並訓練預測器
            predictor = ARIMAStockPredictor()
            train_result = predictor.train(args.symbol, prices)
            
            if train_result['success']:
                # 保存模型
                models_dir = './models/arima'
                os.makedirs(models_dir, exist_ok=True)
                
                model_file = os.path.join(models_dir, f"{args.symbol}_arima_model.pkl")
                
                # 保存模型數據
                model_data = {
                    'symbol': predictor.symbol,
                    'model': predictor.model,
                    'fitted_model': predictor.fitted_model,
                    'is_trained': predictor.is_trained,
                    'last_training_data': predictor.last_training_data.tolist() if predictor.last_training_data is not None else None,
                    'training_result': train_result
                }
                
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"✅ {args.symbol} ARIMA模型訓練成功")
                print(f"📊 準確率: {train_result.get('training_accuracy', 0):.3f}")
                print(f"💾 模型已保存: {model_file}")
                
            else:
                print(f"❌ {args.symbol} ARIMA模型訓練失敗: {train_result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        except Exception as e:
            print(f"❌ 訓練過程發生錯誤: {e}")
            sys.exit(1)
    
    elif args.symbol and args.predict:
        # 預測模式
        print(f"🔮 使用 {args.symbol} ARIMA模型預測...")
        
        try:
            models_dir = './models/arima'
            model_file = os.path.join(models_dir, f"{args.symbol}_arima_model.pkl")
            
            if not os.path.exists(model_file):
                print(f"❌ 找不到 {args.symbol} 的ARIMA模型，請先訓練")
                sys.exit(1)
            
            # 載入模型
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            predictor = ARIMAStockPredictor()
            predictor.symbol = model_data['symbol']
            predictor.model = model_data['model']
            predictor.fitted_model = model_data['fitted_model']
            predictor.is_trained = model_data['is_trained']
            predictor.last_training_data = np.array(model_data['last_training_data']) if model_data['last_training_data'] else None
            
            # 進行預測
            pred_result = predictor.predict(args.days)
            
            if pred_result['success']:
                print(f"✅ {args.symbol} ARIMA預測完成")
                print(f"🔮 未來{args.days}天預測:")
                for i, pred in enumerate(pred_result['predictions'], 1):
                    confidence = pred_result['confidence_scores'][i-1]
                    print(f"  第{i}天: {pred:.2f} (信心度: {confidence:.1%})")
            else:
                print(f"❌ 預測失敗: {pred_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 預測過程發生錯誤: {e}")
            sys.exit(1)
    
    else:
        # 測試模式
        print("🧪 測試ARIMA預測器")
        
        # 生成測試數據
        import random
        np.random.seed(42)
        base_price = 100
        prices = []
        for i in range(50):
            trend = i * 0.1
            noise = random.uniform(-2, 2)
            price = base_price + trend + noise
            prices.append(price)
        
        predictor = ARIMAStockPredictor()
        
        # 訓練
        print("訓練ARIMA模型...")
        train_result = predictor.train("TEST", prices)
        print(f"訓練結果: {train_result}")
        
        # 預測
        print("\n進行預測...")
        pred_result = predictor.predict(3)
        print(f"預測結果: {pred_result}")
        
        # 模型信息
        print(f"\n模型信息: {predictor.get_model_info()}")