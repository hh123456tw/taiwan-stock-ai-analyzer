#!/usr/bin/env python3
"""
XGBoost 股價預測器 - 使用極端梯度提升演算法預測股價
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostStockPredictor:
    """XGBoost 股價預測器"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """初始化預測器
        
        Args:
            n_estimators: 樹的數量
            max_depth: 樹的最大深度
            learning_rate: 學習率
            random_state: 隨機種子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = 'models/xgb/'
        
        # 創建模型目錄
        os.makedirs(self.model_path, exist_ok=True)
        
        logger.info(f"⚡ XGBoost 預測器已初始化 (樹數量: {n_estimators}, 深度: {max_depth})")
    
    def prepare_features(self, stock_data):
        """準備特徵資料 - 與 Random Forest 相同但增加更多高階特徵
        
        Args:
            stock_data: 股票歷史資料
            
        Returns:
            pd.DataFrame: 特徵資料框
        """
        try:
            # 轉換為 DataFrame
            df = pd.DataFrame(stock_data)
            
            # 確保有足夠的資料
            if len(df) < 28:
                raise ValueError("資料量不足，需要至少 28 天的歷史資料")
            
            # 基本價格特徵
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(2)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            
            # 價格動量特徵
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # 移動平均線特徵
            for window in [5, 10, 20, 30]:
                df[f'ma{window}'] = df['close'].rolling(window=window).mean()
                df[f'price_to_ma{window}'] = df['close'] / df[f'ma{window}']
                df[f'ma{window}_slope'] = df[f'ma{window}'].diff(5)
            
            # 指數移動平均
            for span in [12, 26]:
                df[f'ema{span}'] = df['close'].ewm(span=span).mean()
                df[f'price_to_ema{span}'] = df['close'] / df[f'ema{span}']
            
            # 波動率特徵（多個時間窗口）
            for window in [5, 10, 20, 30]:
                df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
                df[f'volatility_{window}_norm'] = df[f'volatility_{window}'] / df['close']
            
            # 高低價特徵
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_to_high'] = df['close'] / df['high']
            df['close_to_low'] = df['close'] / df['low']
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            
            # 成交量特徵
            for window in [5, 10, 20]:
                df[f'volume_ma{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma{window}']
            
            df['volume_price_trend'] = df['volume'] * df['price_change']
            df['volume_change'] = df['volume'].pct_change()
            
            # 技術指標
            df['rsi'] = self._calculate_rsi(df['close'])
            df['rsi_ma'] = df['rsi'].rolling(window=5).mean()
            
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            df['bb_upper'], df['bb_lower'], df['bb_middle'] = self._calculate_bollinger_bands(df['close'])
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 威廉指標
            df['williams_r'] = self._calculate_williams_r(df['high'], df['low'], df['close'])
            
            # 隨機指標
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
            
            # 趨勢特徵
            df['trend_5'] = np.where(df['close'] > df['close'].shift(5), 1, 0)
            df['trend_10'] = np.where(df['close'] > df['close'].shift(10), 1, 0)
            df['trend_20'] = np.where(df['close'] > df['close'].shift(20), 1, 0)
            
            # 支撐阻力特徵
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['price_to_resistance'] = df['close'] / df['resistance']
            df['price_to_support'] = df['close'] / df['support']
            
            # 時間特徵
            df.index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
            
            # 週期性特徵
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
            
            # 滯後特徵
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'high_lag_{lag}'] = df['high'].shift(lag)
                df[f'low_lag_{lag}'] = df['low'].shift(lag)
            
            # 滾動統計特徵 - 修復：一次性構建所有特徵避免DataFrame碎片化
            rolling_features = pd.DataFrame(index=df.index)
            
            for window in [5, 10, 20]:
                rolling_data = {
                    f'close_max_{window}': df['close'].rolling(window).max(),
                    f'close_min_{window}': df['close'].rolling(window).min(),
                    f'close_std_{window}': df['close'].rolling(window).std(),
                    f'close_skew_{window}': df['close'].rolling(window).skew(),
                    f'close_kurt_{window}': df['close'].rolling(window).kurt()
                }
                
                # 一次性添加所有滾動特徵
                for col_name, col_data in rolling_data.items():
                    rolling_features[col_name] = col_data
            
            # 合併所有特徵
            df = pd.concat([df, rolling_features], axis=1)
            
            # 選擇所有數值特徵（排除原始價格資料）
            exclude_columns = ['close', 'high', 'low', 'volume'] + [col for col in df.columns if col.startswith('ma') and not any(x in col for x in ['_', 'ratio', 'slope'])]
            feature_columns = [col for col in df.columns if col not in exclude_columns and df[col].dtype in ['float64', 'int64']]
            
            # 過濾存在的欄位並移除無限值
            existing_columns = [col for col in feature_columns if col in df.columns]
            features_df = df[existing_columns].copy()
            
            # 處理無限值和缺失值 - 更徹底的清理
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            
            # 使用新的 pandas 方法
            features_df = features_df.bfill().ffill()
            
            # 強力清理缺失值
            for col in features_df.columns:
                if features_df[col].isnull().any():
                    median_val = features_df[col].median()
                    if pd.isna(median_val):
                        mean_val = features_df[col].mean()
                        if pd.isna(mean_val):
                            features_df[col] = features_df[col].fillna(0)
                        else:
                            features_df[col] = features_df[col].fillna(mean_val)
                    else:
                        features_df[col] = features_df[col].fillna(median_val)
            
            # 最終清理
            if features_df.isnull().any().any():
                features_df = features_df.fillna(0)
                
            if np.isinf(features_df.values).any():
                features_df = features_df.replace([np.inf, -np.inf], 0)
            
            # 數值轉換
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
            
            # 儲存特徵名稱
            self.feature_names = features_df.columns.tolist()
            
            logger.info(f"✅ XGBoost 特徵準備完成，共 {len(self.feature_names)} 個特徵")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ XGBoost 特徵準備失敗: {e}")
            raise
    
    def _calculate_rsi(self, prices, period=14):
        """計算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """計算 MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, period=20, std=2):
        """計算布林通道"""
        ma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        return upper, lower, ma
    
    def _calculate_williams_r(self, high, low, close, period=14):
        """計算威廉指標"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """計算隨機指標"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def train(self, stock_data, symbol, test_size=0.2, optimize_hyperparams=False):
        """訓練模型
        
        Args:
            stock_data: 股票資料
            symbol: 股票代碼
            test_size: 測試集比例
            optimize_hyperparams: 是否進行超參數優化
            
        Returns:
            dict: 訓練結果
        """
        try:
            logger.info(f"⚡ 開始訓練 {symbol} 的 XGBoost 模型")
            
            # 準備特徵
            features_df = self.prepare_features(stock_data)
            
            # 使用與 RF 相同的徹底修復方法
            close_prices = stock_data['close']
            logger.info(f"📊 XGBoost 原始數據: features={features_df.shape}, prices={len(close_prices)}")
            
            # 統一長度
            min_length = min(len(features_df), len(close_prices))
            if min_length < 28:
                raise ValueError(f"XGBoost 資料不足: {min_length} < 28")
            
            # 截取並轉換為 numpy
            features_matrix = features_df.iloc[:min_length].copy()
            price_array = close_prices[:min_length]
            # 修復: 正確建立目標陣列 - 預測下一日價格
            if min_length > 1:
                # 使用下一日價格作為目標，最後一天重複最後價格
                target_list = price_array[1:].tolist() + [price_array[-1]]
                target_array = target_list[:min_length]  # 確保長度一致
            else:
                target_array = price_array[:]
            target_array = target_array[:min_length]
            
            # 轉換為 numpy array
            X_raw = features_matrix.values
            y_raw = np.array(target_array)
            
            # 清理 NaN
            nan_mask_X = ~np.isnan(X_raw).any(axis=1)
            nan_mask_y = ~np.isnan(y_raw)
            valid_mask = nan_mask_X & nan_mask_y
            
            X = X_raw[valid_mask]
            y = y_raw[valid_mask]
            
            logger.info(f"📊 XGBoost 篩選後: X={X.shape}, y={y.shape}")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"XGBoost 樣本數不一致: X={X.shape[0]}, y={y.shape[0]}")
            
            if X.shape[0] < 28:
                raise ValueError(f"XGBoost 有效樣本不足: {X.shape[0]}")
            
            # 轉換回 DataFrame
            X = pd.DataFrame(X, columns=features_df.columns)
            y = pd.Series(y)
            
            logger.info(f"✅ XGBoost 資料準備完成: X={X.shape}, y={len(y)}")
            
            # 分割訓練和測試集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, shuffle=False
            )
            
            # 特徵縮放 - 避免特徵名稱警告
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)
            
            # 轉換為 DataFrame 保持一致性
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # 創建 XGBoost 模型
            if optimize_hyperparams and len(X_train) > 100:
                # 超參數優化
                logger.info("🔧 開始超參數優化...")
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
                
                xgb_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)
                grid_search = GridSearchCV(
                    xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
                logger.info(f"✅ 最佳參數: {grid_search.best_params_}")
            else:
                # 使用預設參數
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0
                )
                
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
            model_file = os.path.join(self.model_path, f'{symbol}_xgb_model.pkl')
            scaler_file = os.path.join(self.model_path, f'{symbol}_xgb_scaler.pkl')
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            results = {
                'symbol': symbol,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(self.feature_names),
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
            
            logger.info(f"✅ {symbol} XGBoost 模型訓練完成")
            logger.info(f"📊 測試集方向準確率: {test_direction_acc:.2%}")
            logger.info(f"📊 測試集 R² 分數: {test_r2:.4f}")
            logger.info(f"🔍 使用特徵數量: {len(self.feature_names)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ XGBoost 訓練失敗: {e}")
            raise
    
    def predict(self, stock_data, symbol, days=5):
        """預測未來股價
        
        Args:
            stock_data: 股票資料
            symbol: 股票代碼
            days: 預測天數
            
        Returns:
            list: 預測價格列表
        """
        try:
            # 載入模型
            model_file = os.path.join(self.model_path, f'{symbol}_xgb_model.pkl')
            scaler_file = os.path.join(self.model_path, f'{symbol}_xgb_scaler.pkl')
            
            if not os.path.exists(model_file):
                raise ValueError(f"找不到 {symbol} 的 XGBoost 模型，請先訓練")
            
            # 🔧 新增：智能重訓練檢查 (模型超過7天自動重訓練)
            model_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_file))).days
            if model_age_days >= 7:
                logger.info(f"🔄 XGBoost模型已 {model_age_days} 天未更新，觸發自動重訓練...")
                try:
                    # 重新訓練模型
                    self.train(stock_data, symbol, n_estimators=150)
                    logger.info(f"✅ {symbol} XGBoost模型重訓練完成")
                except Exception as retrain_error:
                    logger.warning(f"⚠️ XGBoost重訓練失敗，使用舊模型: {retrain_error}")
            else:
                logger.info(f"📊 使用 {model_age_days} 天前的XGBoost模型 (7天內免重訓練)")
            
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            
            # 準備特徵
            features_df = self.prepare_features(stock_data)
            
            # 使用最新的資料進行預測
            latest_features = features_df.iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            predictions = []
            current_data = stock_data.copy()
            
            for day in range(days):
                # 預測下一天的價格
                pred_price = self.model.predict(latest_features_scaled)[0]
                predictions.append(float(pred_price))
                
                # 更新資料用於下一次預測
                try:
                    # 將預測價格加入資料
                    current_data['close'].append(pred_price)
                    if len(current_data['high']) > 0:
                        current_data['high'].append(max(pred_price, current_data['high'][-1]))
                        current_data['low'].append(min(pred_price, current_data['low'][-1]))
                    else:
                        current_data['high'].append(pred_price)
                        current_data['low'].append(pred_price)
                    
                    if len(current_data['volume']) > 0:
                        current_data['volume'].append(current_data['volume'][-1])
                    else:
                        current_data['volume'].append(1000)
                    
                    # 重新計算特徵
                    updated_features = self.prepare_features(current_data)
                    latest_features = updated_features.iloc[-1:].values
                    latest_features_scaled = self.scaler.transform(latest_features)
                    
                except Exception:
                    # 如果更新失敗，就重複使用同樣的特徵
                    pass
            
            logger.info(f"✅ {symbol} XGBoost 預測完成: {days} 天")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ XGBoost 預測失敗: {e}")
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
        """評估模型表現
        
        Args:
            stock_data: 股票資料
            symbol: 股票代碼
            
        Returns:
            dict: 評估結果
        """
        try:
            # 載入模型
            model_file = os.path.join(self.model_path, f'{symbol}_xgb_model.pkl')
            scaler_file = os.path.join(self.model_path, f'{symbol}_xgb_scaler.pkl')
            
            if not os.path.exists(model_file):
                raise ValueError(f"找不到 {symbol} 的模型")
            
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            
            # 準備資料
            features_df = self.prepare_features(stock_data)
            target = pd.Series(stock_data['close']).shift(-1)
            
            # 移除缺失值
            valid_idx = ~(features_df.isnull().any(axis=1) | target.isnull())
            X = features_df[valid_idx]
            y = target[valid_idx]
            
            # 預測
            X_scaled = self.scaler.transform(X)
            y_pred = self.model.predict(X_scaled)
            
            # 計算指標
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            direction_accuracy = self._calculate_direction_accuracy(y, y_pred)
            
            # 計算 MAPE
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy),
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"❌ XGBoost 模型評估失敗: {e}")
            return {
                'mse': 0.0,
                'mae': 0.0,
                'r2': 0.0,
                'mape': 0.0,
                'direction_accuracy': 0.0
            }
    
    def get_feature_importance_plot_data(self, symbol, top_n=20):
        """獲取特徵重要性資料用於繪圖
        
        Args:
            symbol: 股票代碼
            top_n: 顯示前 N 個重要特徵
            
        Returns:
            dict: 特徵重要性資料
        """
        try:
            model_file = os.path.join(self.model_path, f'{symbol}_xgb_model.pkl')
            
            if not os.path.exists(model_file):
                return {}
            
            self.model = joblib.load(model_file)
            
            # 獲取特徵重要性
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            return {
                'features': [item[0] for item in top_features],
                'importance': [item[1] for item in top_features]
            }
            
        except Exception as e:
            logger.error(f"❌ 獲取特徵重要性失敗: {e}")
            return {}

# 整合到 Telegram Bot 的函數
async def handle_xgb_predict_command(update, context, data_fetcher):
    """處理 XGBoost 預測指令"""
    if not context.args:
        if hasattr(update, 'message') and update.message:
            await update.message.reply_text("請提供股票代碼\n用法: /xgb 2330")
        else:
            await update.callback_query.message.reply_text("請提供股票代碼")
        return

    symbol = context.args[0]
    
    # 判斷是從 message 還是 callback_query 來的
    if hasattr(update, 'message') and update.message:
        msg = await update.message.reply_text(f"⚡ 正在使用 XGBoost 分析 {symbol}...")
    else:
        msg = await update.callback_query.message.reply_text(f"⚡ 正在使用 XGBoost 分析 {symbol}...")

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
        if data_length < 28:
            await msg.edit_text(f"❌ {symbol} 的歷史資料不足（XGBoost 需要至少28天，目前只有{data_length}天）")
            return

        # 初始化預測器
        predictor = XGBoostStockPredictor(n_estimators=100, max_depth=6, learning_rate=0.1)

        # 檢查是否需要訓練
        model_file = os.path.join(predictor.model_path, f'{symbol}_xgb_model.pkl')

        if not os.path.exists(model_file):
            await msg.edit_text(f"🔧 首次使用 {symbol}，正在訓練 XGBoost 模型...\n⏳ 這可能需要 2-3 分鐘")

            # 訓練模型
            try:
                training_result = predictor.train(df_data, symbol, optimize_hyperparams=False)
                await msg.edit_text(f"✅ {symbol} XGBoost 模型訓練完成，正在進行預測...")
            except Exception as train_error:
                logger.error(f"XGBoost 訓練錯誤: {train_error}")
                await msg.edit_text(f"❌ 模型訓練失敗: {str(train_error)}")
                return

        # 預測未來 5 天
        try:
            predictions = predictor.predict(df_data, symbol, days=5)
        except Exception as pred_error:
            logger.error(f"XGBoost 預測錯誤: {pred_error}")
            await msg.edit_text(f"❌ 預測失敗: {str(pred_error)}")
            return

        # 評估模型
        try:
            metrics = predictor.evaluate_model(df_data, symbol)
        except Exception as eval_error:
            logger.warning(f"XGBoost 評估錯誤: {eval_error}")
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
⚡ **XGBoost 股價預測 - {symbol}**

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
• {', '.join(top_features) if top_features else '高階技術指標組合'}

⚠️ **模型特色:**
• XGBoost 使用梯度提升算法
• 擅長捕捉複雜的非線性關係
• 自動處理特徵交互作用
• 對異常值相對穩健

⚡ **演算法優勢:**
• 高精度預測能力
• 自動特徵選擇
• 防止過擬合機制
• 快速訓練和預測

⚠️ **重要提醒:**
• AI 預測僅供參考，不構成投資建議
• 股價受多種因素影響，實際結果可能差異很大
• 請結合基本面分析並謹慎投資

🔧 模型技術: XGBoost 極端梯度提升，使用 {len(predictor.feature_names)} 個特徵

⏰ 預測時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        await msg.edit_text(message)

    except Exception as e:
        logger.error(f"XGBoost 預測過程發生錯誤: {str(e)}")
        await msg.edit_text(f"❌ 預測過程發生錯誤，請稍後再試\n錯誤詳情: {str(e)}")

# 使用範例
if __name__ == "__main__":
    # 測試用
    predictor = XGBoostStockPredictor()
    print("⚡ XGBoost 股價預測器已準備就緒")
    
    # 測試基本功能
    test_data = {
        'close': [100 + i + np.random.normal(0, 2) for i in range(100)],
        'high': [102 + i + np.random.normal(0, 2) for i in range(100)],
        'low': [98 + i + np.random.normal(0, 2) for i in range(100)],
        'volume': [1000 + np.random.randint(-200, 200) for _ in range(100)]
    }
    
    try:
        features = predictor.prepare_features(test_data)
        print(f"✅ XGBoost 特徵準備功能正常，共 {len(features.columns)} 個特徵")
        print(f"特徵列表前10個: {list(features.columns[:10])}")
    except Exception as e:
        print(f"❌ XGBoost 特徵準備失敗: {e}")
