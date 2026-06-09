#!/usr/bin/env python3
"""
Random Forest 股價預測器 - 完整修復版
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestStockPredictor:
    """Random Forest 股價預測器 - AMD Ryzen 7 5700X多核心優化版"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # 🚀 AMD Ryzen 7 5700X 8核心優化配置
        import multiprocessing
        self.n_jobs = min(8, multiprocessing.cpu_count())  # 使用8核心或可用核心數
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = 'models/rf/'
        
        os.makedirs(self.model_path, exist_ok=True)
        logger.info(f"🌲 Random Forest 預測器已初始化")
        logger.info(f"   樹數量: {n_estimators}")
        logger.info(f"   🚀 使用AMD Ryzen 7 5700X: {self.n_jobs}核心並行加速")
    
    def prepare_features(self, stock_data):
        """準備特徵資料"""
        try:
            # 確保輸入是字典格式
            if isinstance(stock_data, dict):
                df = pd.DataFrame({
                    'close': stock_data['close'],
                    'high': stock_data['high'],
                    'low': stock_data['low'],
                    'volume': stock_data['volume']
                })
            else:
                df = pd.DataFrame(stock_data)
            
            # 確保所有列都是數值型且長度一致
            min_length = min(len(df[col]) for col in ['close', 'high', 'low', 'volume'])
            logger.info(f"📊 統一後長度: {min_length}")
            
            if min_length < 20:
                raise ValueError(f"資料量不足，需要至少 20 天的歷史資料，目前只有 {min_length} 天")
            
            # 截取至相同長度
            for col in ['close', 'high', 'low', 'volume']:
                df[col] = df[col][:min_length]
            
            # 重設索引
            df = df.reset_index(drop=True)
            
            # 基本價格特徵
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(2)
            df['price_change_5'] = df['close'].pct_change(5)
            
            # 移動平均線特徵
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
            
            # 移動平均線比率
            df['price_to_ma5'] = df['close'] / df['ma5']
            df['price_to_ma10'] = df['close'] / df['ma10']
            df['price_to_ma20'] = df['close'] / df['ma20']
            
            # 波動率特徵
            df['volatility_5'] = df['close'].rolling(window=5, min_periods=1).std()
            df['volatility_10'] = df['close'].rolling(window=10, min_periods=1).std()
            df['volatility_20'] = df['close'].rolling(window=20, min_periods=1).std()
            
            # 高低價特徵
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_to_high'] = df['close'] / df['high']
            df['close_to_low'] = df['close'] / df['low']
            
            # 成交量特徵
            df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
            df['volume_ma10'] = df['volume'].rolling(window=10, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
            
            # 技術指標
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 滯後特徵
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # 選擇特徵欄位
            feature_columns = [
                'price_change', 'price_change_2', 'price_change_5',
                'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
                'volatility_5', 'volatility_10', 'volatility_20',
                'high_low_ratio', 'close_to_high', 'close_to_low',
                'volume_ratio', 'rsi', 'macd', 'macd_signal', 'bb_position',
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5'
            ]
            
            # 過濾存在的欄位
            existing_columns = [col for col in feature_columns if col in df.columns]
            features_df = df[existing_columns].copy()
            
            # 使用新的 pandas 方法填充缺失值
            features_df = features_df.bfill().ffill()
            
            # 處理剩餘的 NaN 值
            features_df = features_df.fillna(0)
            
            # 處理無限值
            features_df = features_df.replace([np.inf, -np.inf], 0)
            
            # 儲存特徵名稱
            self.feature_names = features_df.columns.tolist()
            
            logger.info(f"✅ 特徵準備完成，共 {len(self.feature_names)} 個特徵，樣本數：{len(features_df)}")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ 特徵準備失敗: {e}")
            raise
    
    def _calculate_rsi(self, prices, period=14):
        """計算 RSI"""
        try:
            if len(prices) < period + 1:
                return pd.Series([50] * len(prices), index=prices.index)
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # 避免除零錯誤
            loss = loss.replace(0, 1e-8)
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 處理異常值
            rsi = rsi.fillna(50)
            rsi = rsi.clip(0, 100)
            
            return rsi
            
        except Exception as e:
            logger.warning(f"RSI 計算錯誤: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """計算 MACD"""
        ema_fast = prices.ewm(span=fast, min_periods=1).mean()
        ema_slow = prices.ewm(span=slow, min_periods=1).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, min_periods=1).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, period=20, std=2):
        """計算布林通道"""
        ma = prices.rolling(window=period, min_periods=1).mean()
        std_dev = prices.rolling(window=period, min_periods=1).std()
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        return upper, lower
    
    def train(self, stock_data, symbol, test_size=0.2):
        """訓練模型"""
        try:
            logger.info(f"🌲 開始訓練 {symbol} 的 Random Forest 模型")
            
            # 準備特徵
            features_df = self.prepare_features(stock_data)
            
            # 修復：確保目標變數與特徵矩陣長度一致
            close_prices = stock_data['close']
            
            # 統一長度
            min_length = min(len(features_df), len(close_prices))
            features_df = features_df.iloc[:min_length].copy()
            
            # 創建目標變數（下一天的收盤價）
            target = pd.Series(close_prices[:min_length]).shift(-1)
            target.iloc[-1] = close_prices[min_length-1]  # 最後一天用當天價格
            
            logger.info(f"📊 資料長度檢查: features={len(features_df)}, target={len(target)}")
            
            # 移除含有 NaN 的行
            valid_mask = ~(features_df.isnull().any(axis=1) | target.isnull())
            X = features_df[valid_mask].copy()
            y = target[valid_mask].copy()
            
            logger.info(f"📊 清理後資料: X={len(X)}, y={len(y)}")
            
            if len(X) < 10:
                raise ValueError(f"有效資料量不足: {len(X)} < 10")
            
            if len(X) != len(y):
                raise ValueError(f"特徵矩陣與目標變數長度不一致: X={len(X)}, y={len(y)}")
            
            # 分割訓練和測試集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, shuffle=False
            )
            
            # 特徵縮放
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)
            
            # 創建和訓練模型 - AMD Ryzen 7 5700X優化
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,  # 🚀 使用8核心並行
                max_depth=12,  # 增加深度利用CPU優勢
                min_samples_split=4,  # 優化分割策略
                min_samples_leaf=1,   # 允許更細緻分割
                max_features='sqrt',  # 優化特徵選擇
                bootstrap=True,       # 確保使用bootstrap
                oob_score=True,       # 啟用out-of-bag評分
                verbose=1            # 顯示訓練進度
            )
            
            logger.info(f"🚀 開始Random Forest訓練 (使用{self.n_jobs}核心)")
            
            self.model.fit(X_train_scaled, y_train)
            
            # 預測和評估
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            # 計算評估指標
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # 計算方向準確率
            train_direction_acc = self._calculate_direction_accuracy(y_train, y_pred_train)
            test_direction_acc = self._calculate_direction_accuracy(y_test, y_pred_test)
            
            # 特徵重要性
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # 儲存模型
            model_file = os.path.join(self.model_path, f'{symbol}_rf_model.pkl')
            scaler_file = os.path.join(self.model_path, f'{symbol}_rf_scaler.pkl')
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            results = {
                'symbol': symbol,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': {
                    'train_mse': float(train_mse),
                    'test_mse': float(test_mse),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_direction_accuracy': float(train_direction_acc),
                    'test_direction_accuracy': float(test_direction_acc)
                },
                'feature_importance': dict(top_features),
                'model_path': model_file
            }
            
            logger.info(f"✅ {symbol} Random Forest 模型訓練完成")
            logger.info(f"📊 測試集方向準確率: {test_direction_acc:.2%}")
            logger.info(f"📊 測試集 R² 分數: {test_r2:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 訓練失敗: {e}")
            raise
    
    def predict(self, stock_data, symbol, days=5):
        """預測未來股價 + 智能重訓練"""
        try:
            # 載入模型
            model_file = os.path.join(self.model_path, f'{symbol}_rf_model.pkl')
            scaler_file = os.path.join(self.model_path, f'{symbol}_rf_scaler.pkl')
            
            if not os.path.exists(model_file):
                raise ValueError(f"找不到 {symbol} 的 Random Forest 模型，請先訓練")
            
            # 🔧 新增：智能重訓練檢查 (模型超過7天自動重訓練)
            model_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_file))).days
            if model_age_days >= 7:
                logger.info(f"🔄 RF模型已 {model_age_days} 天未更新，觸發自動重訓練...")
                try:
                    # 重新訓練模型
                    self.train(stock_data, symbol, n_estimators=200)
                    logger.info(f"✅ {symbol} RF模型重訓練完成")
                except Exception as retrain_error:
                    logger.warning(f"⚠️ RF重訓練失敗，使用舊模型: {retrain_error}")
            else:
                logger.info(f"📊 使用 {model_age_days} 天前的RF模型 (7天內免重訓練)")
            
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            
            # 準備特徵
            features_df = self.prepare_features(stock_data)
            
            # 使用最新的資料進行預測
            latest_features = features_df.iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            predictions = []
            current_data = {
                'close': stock_data['close'][:],
                'high': stock_data['high'][:],
                'low': stock_data['low'][:],
                'volume': stock_data['volume'][:]
            }
            
            for day in range(days):
                # 預測下一天的價格
                pred_price = self.model.predict(latest_features_scaled)[0]
                predictions.append(float(pred_price))
                
                # 更新資料用於下一次預測
                try:
                    current_data['close'].append(pred_price)
                    current_data['high'].append(max(pred_price, current_data['high'][-1] * 1.02))
                    current_data['low'].append(min(pred_price, current_data['low'][-1] * 0.98))
                    current_data['volume'].append(current_data['volume'][-1])
                    
                    # 重新計算特徵
                    updated_features = self.prepare_features(current_data)
                    latest_features = updated_features.iloc[-1:].values
                    latest_features_scaled = self.scaler.transform(latest_features)
                    
                except Exception:
                    # 如果更新失敗，就重複使用同樣的特徵
                    pass
            
            logger.info(f"✅ {symbol} Random Forest 預測完成: {days} 天")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Random Forest 預測失敗: {e}")
            raise
    
    def _calculate_direction_accuracy(self, y_true, y_pred):
        """計算方向預測準確率"""
        if len(y_true) < 2:
            return 0.0
        
        # 計算實際和預測的變化方向
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # 計算準確率
        accuracy = np.mean(true_direction == pred_direction)
        return accuracy
    
    def evaluate_model(self, stock_data, symbol):
        """評估模型表現"""
        try:
            # 載入模型
            model_file = os.path.join(self.model_path, f'{symbol}_rf_model.pkl')
            scaler_file = os.path.join(self.model_path, f'{symbol}_rf_scaler.pkl')
            
            if not os.path.exists(model_file):
                return {
                    'mse': 0.0, 'mae': 0.0, 'r2': 0.0,
                    'mape': 0.0, 'direction_accuracy': 0.0,
                    'feature_importance': {}
                }
            
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            
            # 準備資料（使用與訓練相同的邏輯）
            features_df = self.prepare_features(stock_data)
            close_prices = stock_data['close']
            min_length = min(len(features_df), len(close_prices))
            features_df = features_df.iloc[:min_length].copy()
            target = pd.Series(close_prices[:min_length]).shift(-1)
            target.iloc[-1] = close_prices[min_length-1]
            
            # 移除缺失值
            valid_mask = ~(features_df.isnull().any(axis=1) | target.isnull())
            X = features_df[valid_mask].copy()
            y = target[valid_mask].copy()
            
            if len(X) < 5:
                return {
                    'mse': 0.0, 'mae': 0.0, 'r2': 0.0,
                    'mape': 0.0, 'direction_accuracy': 0.0,
                    'feature_importance': {}
                }
            
            # 預測和評估
            X_scaled = self.scaler.transform(X.values)
            y_pred = self.model.predict(X_scaled)
            
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            direction_accuracy = self._calculate_direction_accuracy(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            return {
                'mse': float(mse), 'mae': float(mae), 'r2': float(r2),
                'mape': float(mape), 'direction_accuracy': float(direction_accuracy),
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"❌ 評估失敗: {e}")
            return {
                'mse': 0.0, 'mae': 0.0, 'r2': 0.0,
                'mape': 0.0, 'direction_accuracy': 0.0,
                'feature_importance': {}
            }

# 整合到 Telegram Bot 的函數
async def handle_rf_predict_command(update, context, data_fetcher):
    """處理 Random Forest 預測指令"""
    if not context.args:
        if hasattr(update, 'message') and update.message:
            await update.message.reply_text("請提供股票代碼\n用法: /rf 2330")
        else:
            await update.callback_query.message.reply_text("請提供股票代碼")
        return

    symbol = context.args[0]
    
    # 判斷是從 message 還是 callback_query 來的
    if hasattr(update, 'message') and update.message:
        msg = await update.message.reply_text(f"🌲 正在使用 Random Forest 分析 {symbol}...")
    else:
        msg = await update.callback_query.message.reply_text(f"🌲 正在使用 Random Forest 分析 {symbol}...")

    try:
        # 使用統一數據管理器獲取資料
        from unified_data_manager import UnifiedDataManager
        data_manager = UnifiedDataManager()
        
        # 獲取訓練用歷史數據
        df = data_manager.get_training_data(symbol, days=365)
        
        if df.empty:
            await msg.edit_text(f"❌ 無法獲取 {symbol} 的資料")
            return

        # 準備資料格式
        df_data = {
            'close': df['close'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'volume': df['volume'].tolist()
        }

        # 檢查資料量是否足夠
        data_length = len(df_data['close'])
        if data_length < 30:
            await msg.edit_text(f"❌ {symbol} 的歷史資料不足（需要至少30天，目前只有{data_length}天）")
            return

        # 初始化預測器
        predictor = RandomForestStockPredictor(n_estimators=100)

        # 檢查是否需要訓練
        model_file = os.path.join(predictor.model_path, f'{symbol}_rf_model.pkl')

        if not os.path.exists(model_file):
            await msg.edit_text(f"🔧 首次使用 {symbol}，正在訓練 Random Forest 模型...\n⏳ 這可能需要 1-2 分鐘")

            # 訓練模型
            try:
                training_result = predictor.train(df_data, symbol)
                await msg.edit_text(f"✅ {symbol} Random Forest 模型訓練完成，正在進行預測...")
            except Exception as train_error:
                logger.error(f"訓練錯誤: {train_error}")
                await msg.edit_text(f"❌ 模型訓練失敗: {str(train_error)}")
                return

        # 預測未來 5 天
        try:
            predictions = predictor.predict(df_data, symbol, days=5)
        except Exception as pred_error:
            logger.error(f"預測錯誤: {pred_error}")
            await msg.edit_text(f"❌ 預測失敗: {str(pred_error)}")
            return

        # 評估模型
        try:
            metrics = predictor.evaluate_model(df_data, symbol)
        except Exception as eval_error:
            logger.warning(f"評估錯誤: {eval_error}")
            metrics = {
                'direction_accuracy': 0,
                'mape': 0,
                'r2': 0
            }

        # 準備回應訊息
        current_price = historical_data['close'][-1]

        # 計算預測趨勢
        trend_direction = "📈 看漲" if predictions[-1] > current_price else "📉 看跌"
        total_change = (predictions[-1] / current_price - 1) * 100

        # 獲取重要特徵
        top_features = list(metrics.get('feature_importance', {}).keys())[:5]

        message = f"""
🌲 **Random Forest 股價預測 - {symbol}**

📊 **目前價格:** {current_price:.2f}

📈 **未來 5 天預測:**
• 第1天: {predictions[0]:.2f} ({(predictions[0]/current_price-1)*100:+.2f}%)
• 第2天: {predictions[1]:.2f} ({(predictions[1]/current_price-1)*100:+.2f}%)
• 第3天: {predictions[2]:.2f} ({(predictions[2]/current_price-1)*100:+.2f}%)
• 第4天: {predictions[3]:.2f} ({(predictions[3]/current_price-1)*100:+.2f}%)
• 第5天: {predictions[4]:.2f} ({(predictions[4]/current_price-1)*100:+.2f}%)

🎯 **預測總結:**
• 5日趨勢: {trend_direction}
• 預期總變動: {total_change:+.2f}%

📊 **模型表現:**
• 方向準確率: {metrics['direction_accuracy']*100:.1f}%
• 平均誤差率: {metrics['mape']:.2f}%
• R² 決定係數: {metrics['r2']:.3f}

🔍 **重要特徵:**
• {', '.join(top_features) if top_features else '技術指標組合'}

⚠️ **模型特色:**
• Random Forest 使用多棵決策樹投票預測
• 對雜訊有較強的抗性
• 能捕捉特徵間的非線性關係
• 預測相對穩定保守

⚠️ **重要提醒:**
• AI 預測僅供參考，不構成投資建議
• 請結合基本面分析並謹慎投資

⏰ 預測時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        await msg.edit_text(message)

    except Exception as e:
        logger.error(f"Random Forest 預測過程發生錯誤: {str(e)}")
        await msg.edit_text(f"❌ 預測過程發生錯誤，請稍後再試\n錯誤詳情: {str(e)}")

# 使用範例
if __name__ == "__main__":
    # 測試用
    predictor = RandomForestStockPredictor()
    print("🌲 Random Forest 股價預測器已準備就緒")

