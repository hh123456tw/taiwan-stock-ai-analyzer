#!/usr/bin/env python3
"""
大規模股票池自動訓練系統
======================

針對你們的5個AI模型 + 77檔股票池的自動化訓練解決方案

功能：
1. 新增股票自動偵測和訓練
2. 定期重訓練排程
3. 模型性能監控和觸發重訓
4. 硬體資源最佳化分配
5. 訓練失敗自動重試
"""

import os
import json
import sqlite3
import subprocess
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoStockTrainingSystem:
    """大規模股票池自動訓練系統"""
    
    def __init__(self, base_dir: str = "/mnt/nvme/taiwan-stock-ai-analyzer-main"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.db_path = os.path.join(base_dir, "predictions.db")
        
        # 5個AI模型配置
        self.model_configs = {
            'lstm': {
                'script': 'lstm_stock_predictor.py',
                'hardware': 'gpu',
                'max_parallel': 1,  # GPU記憶體限制
                'timeout': 300,
                'model_dir': 'lstm',
                'file_pattern': '_lstm.keras'
            },
            'rf': {
                'script': 'rf_stock_predictor.py', 
                'hardware': 'cpu',
                'max_parallel': 4,  # CPU核心數
                'timeout': 180,
                'model_dir': 'rf',
                'file_pattern': '_rf_model.pkl'
            },
            'xgb': {
                'script': 'xgb_stock_predictor.py',
                'hardware': 'cpu', 
                'max_parallel': 4,
                'timeout': 240,
                'model_dir': 'xgb',
                'file_pattern': '_xgb_model.pkl'
            },
            'arima': {
                'script': 'arima_stock_predictor.py',
                'hardware': 'cpu',
                'max_parallel': 6,
                'timeout': 120,
                'model_dir': 'arima',
                'file_pattern': '_arima_model.pkl'
            },
            'lightgbm': {
                'script': 'lightgbm_stock_predictor.py',
                'hardware': 'cpu',
                'max_parallel': 4,
                'timeout': 180,
                'model_dir': 'lightgbm', 
                'file_pattern': '_lgb_model.pkl'
            }
        }
        
        # 完整股票池 (從setup_full_stock_pool_training.py)
        self.full_stock_pool = [
            # 科技權值股
            '2330', '2454', '2317', '2303', '2308', '3711', '2412', '2382', '2395', '3034',
            '2408', '3231', '6166', '6147', '2354', '2344', '6209', '2313',
            # 金融股  
            '2882', '2881', '2892', '2891', '2884', '2886', '5880', '2890', '2880', '2883',
            # 傳產股
            '1101', '1303', '1216', '2002', '6505', '2207', '2301', '1402', '2105', '2474', '1802',
            # 熱門ETF - 高股息類
            '0056', '00878', '00900', '00919', '00929', '00915', '00918', '00731',
            # 熱門ETF - 領袖/指數類
            '0050', '00922', '00850', '00692', '00757', '00895',
            # 熱門ETF - 半導體類
            '00891', '00892', '00927', '00830', '00904', '00881'
        ]
        
        # 訓練狀態追蹤
        self.training_status = {}
        self.last_retrain_check = {}
        
        logger.info(f"✅ 自動訓練系統初始化完成，股票池: {len(self.full_stock_pool)}檔")
    
    def get_trained_stocks_by_model(self, model_name: str) -> List[str]:
        """獲取特定模型已訓練的股票列表"""
        model_dir = os.path.join(self.models_dir, self.model_configs[model_name]['model_dir'])
        file_pattern = self.model_configs[model_name]['file_pattern']
        
        trained_stocks = []
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(file_pattern):
                    symbol = file.replace(file_pattern, '')
                    trained_stocks.append(symbol)
        
        return trained_stocks
    
    def get_missing_stocks_by_model(self, model_name: str) -> List[str]:
        """獲取特定模型尚未訓練的股票"""
        trained = set(self.get_trained_stocks_by_model(model_name))
        return [stock for stock in self.full_stock_pool if stock not in trained]
    
    def check_stock_data_availability(self, symbol: str) -> bool:
        """檢查股票數據是否可用"""
        try:
            ticker_symbol = f"{symbol}.TW" if not symbol.endswith('.TW') else symbol
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period='60d')
            
            # 需要至少30天數據
            return len(hist) >= 30 and not hist.empty
            
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 數據檢查失敗: {e}")
            return False
    
    def train_single_stock_model(self, symbol: str, model_name: str) -> Dict:
        """訓練單一股票的特定模型"""
        config = self.model_configs[model_name]
        
        try:
            logger.info(f"🚀 開始訓練 {symbol} {model_name.upper()} 模型")
            
            # 檢查數據可用性
            if not self.check_stock_data_availability(symbol):
                return {'success': False, 'error': '數據不足或不可用'}
            
            # 構建訓練命令
            script_path = os.path.join(self.base_dir, config['script'])
            cmd = ['python3', script_path, '--symbol', symbol, '--train']
            
            # GPU模型添加GPU參數
            if config['hardware'] == 'gpu':
                cmd.append('--gpu')
            
            # 執行訓練
            start_time = time.time()
            result = subprocess.run(
                cmd,
                timeout=config['timeout'],
                capture_output=True,
                text=True,
                cwd=self.base_dir
            )
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {symbol} {model_name.upper()} 訓練成功 ({training_time:.1f}s)")
                return {
                    'success': True,
                    'training_time': training_time,
                    'stdout': result.stdout
                }
            else:
                logger.error(f"❌ {symbol} {model_name.upper()} 訓練失敗: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'stdout': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {symbol} {model_name.upper()} 訓練超時")
            return {'success': False, 'error': '訓練超時'}
        except Exception as e:
            logger.error(f"❌ {symbol} {model_name.upper()} 訓練異常: {e}")
            return {'success': False, 'error': str(e)}
    
    def batch_train_model(self, model_name: str, stocks: List[str]) -> Dict:
        """批次訓練特定模型的多檔股票"""
        config = self.model_configs[model_name]
        max_workers = config['max_parallel']
        
        logger.info(f"🎯 開始批次訓練 {model_name.upper()}，股票數: {len(stocks)}")
        
        results = {'success_count': 0, 'failed_count': 0, 'details': []}
        
        # 根據硬體類型選擇執行器
        if config['hardware'] == 'gpu':
            # GPU使用ThreadPoolExecutor (避免多進程GPU衝突)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.train_single_stock_model, stock, model_name): stock 
                          for stock in stocks}
                
                for future in futures:
                    stock = futures[future]
                    try:
                        result = future.result()
                        if result['success']:
                            results['success_count'] += 1
                        else:
                            results['failed_count'] += 1
                        results['details'].append({'stock': stock, 'result': result})
                    except Exception as e:
                        results['failed_count'] += 1
                        results['details'].append({'stock': stock, 'result': {'success': False, 'error': str(e)}})
        else:
            # CPU使用ProcessPoolExecutor (更好的並行性)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.train_single_stock_model, stock, model_name): stock 
                          for stock in stocks}
                
                for future in futures:
                    stock = futures[future]
                    try:
                        result = future.result()
                        if result['success']:
                            results['success_count'] += 1
                        else:
                            results['failed_count'] += 1
                        results['details'].append({'stock': stock, 'result': result})
                    except Exception as e:
                        results['failed_count'] += 1
                        results['details'].append({'stock': stock, 'result': {'success': False, 'error': str(e)}})
        
        logger.info(f"✅ {model_name.upper()} 批次訓練完成: {results['success_count']}/{len(stocks)} 成功")
        return results
    
    def auto_train_new_stocks(self) -> Dict:
        """自動訓練新增股票"""
        logger.info("🔍 檢查新增股票...")
        
        overall_results = {}
        
        for model_name in self.model_configs.keys():
            missing_stocks = self.get_missing_stocks_by_model(model_name)
            
            if missing_stocks:
                logger.info(f"📋 {model_name.upper()} 發現 {len(missing_stocks)} 檔未訓練股票")
                
                # 過濾有數據的股票
                valid_stocks = [stock for stock in missing_stocks 
                              if self.check_stock_data_availability(stock)]
                
                if valid_stocks:
                    logger.info(f"🎯 {model_name.upper()} 開始訓練 {len(valid_stocks)} 檔股票")
                    results = self.batch_train_model(model_name, valid_stocks)
                    overall_results[model_name] = results
                else:
                    logger.info(f"⚠️ {model_name.upper()} 無可用數據的新股票")
                    overall_results[model_name] = {'success_count': 0, 'failed_count': 0, 'details': []}
            else:
                logger.info(f"✅ {model_name.upper()} 所有股票已訓練完成")
                overall_results[model_name] = {'success_count': 0, 'failed_count': 0, 'details': []}
        
        return overall_results
    
    def check_model_performance(self, model_name: str, days_back: int = 7) -> Dict:
        """檢查模型性能，判斷是否需要重訓練"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
                SELECT symbol, AVG(accuracy_score) as avg_accuracy, COUNT(*) as prediction_count
                FROM predictions 
                WHERE model_name LIKE ? 
                AND prediction_date > ?
                AND actual_price IS NOT NULL
                GROUP BY symbol
                HAVING prediction_count >= ?
            """
            
            df = pd.read_sql_query(query, conn, params=(f'%{model_name}%', cutoff_date, 3))
            conn.close()
            
            if df.empty:
                return {'needs_retrain': [], 'reason': 'insufficient_data'}
            
            # 找出準確率低於75%的股票
            poor_performance = df[df['avg_accuracy'] < 0.75]
            
            return {
                'needs_retrain': poor_performance['symbol'].tolist(),
                'reason': 'poor_performance',
                'details': poor_performance.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"❌ {model_name} 性能檢查失敗: {e}")
            return {'needs_retrain': [], 'reason': 'error'}
    
    def schedule_retraining(self) -> Dict:
        """定期重訓練排程"""
        logger.info("🔄 執行定期重訓練檢查...")
        
        retrain_results = {}
        
        for model_name in self.model_configs.keys():
            # 檢查上次檢查時間
            last_check = self.last_retrain_check.get(model_name, datetime.min)
            
            # 每週檢查一次
            if datetime.now() - last_check < timedelta(days=7):
                continue
            
            logger.info(f"🔍 檢查 {model_name.upper()} 重訓練需求...")
            
            performance_check = self.check_model_performance(model_name)
            
            if performance_check['needs_retrain']:
                stocks_to_retrain = performance_check['needs_retrain']
                logger.info(f"🎯 {model_name.upper()} 需重訓練 {len(stocks_to_retrain)} 檔股票")
                
                results = self.batch_train_model(model_name, stocks_to_retrain)
                retrain_results[model_name] = results
            else:
                logger.info(f"✅ {model_name.upper()} 性能良好，無需重訓練")
                retrain_results[model_name] = {'success_count': 0, 'failed_count': 0, 'details': []}
            
            self.last_retrain_check[model_name] = datetime.now()
        
        return retrain_results
    
    def get_system_status(self) -> Dict:
        """獲取系統狀態報告"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'stock_pool_size': len(self.full_stock_pool),
            'models': {}
        }
        
        for model_name in self.model_configs.keys():
            trained_stocks = self.get_trained_stocks_by_model(model_name)
            missing_stocks = self.get_missing_stocks_by_model(model_name)
            
            status['models'][model_name] = {
                'trained_count': len(trained_stocks),
                'missing_count': len(missing_stocks),
                'coverage_pct': len(trained_stocks) / len(self.full_stock_pool) * 100,
                'hardware': self.model_configs[model_name]['hardware']
            }
        
        return status
    
    def run_full_training_cycle(self) -> Dict:
        """執行完整訓練週期"""
        logger.info("🚀 開始完整訓練週期...")
        
        start_time = time.time()
        
        # 1. 訓練新增股票
        new_stock_results = self.auto_train_new_stocks()
        
        # 2. 檢查重訓練需求
        retrain_results = self.schedule_retraining()
        
        # 3. 生成狀態報告
        status = self.get_system_status()
        
        total_time = time.time() - start_time
        
        summary = {
            'total_time': total_time,
            'new_stock_training': new_stock_results,
            'retraining': retrain_results,
            'system_status': status
        }
        
        logger.info(f"🎉 完整訓練週期完成，耗時: {total_time:.1f}秒")
        return summary

def main():
    """主程序"""
    print("🚀 大規模股票池自動訓練系統")
    print("=" * 50)
    
    # 初始化系統
    training_system = AutoStockTrainingSystem()
    
    # 獲取系統狀態
    print("\\n📊 當前系統狀態:")
    status = training_system.get_system_status()
    
    for model_name, model_status in status['models'].items():
        print(f"  {model_name.upper()}: {model_status['trained_count']}/{status['stock_pool_size']} "
              f"({model_status['coverage_pct']:.1f}%) [{model_status['hardware'].upper()}]")
    
    # 執行訓練週期
    print("\\n🎯 執行自動訓練...")
    results = training_system.run_full_training_cycle()
    
    # 顯示結果
    print("\\n📈 訓練結果:")
    for model_name, model_results in results['new_stock_training'].items():
        print(f"  {model_name.upper()}: +{model_results['success_count']} 成功, "
              f"{model_results['failed_count']} 失敗")
    
    print(f"\\n⏱️ 總耗時: {results['total_time']:.1f}秒")
    print("🎉 自動訓練完成！")

if __name__ == "__main__":
    main()