#!/usr/bin/env python3
"""
定期重訓練排程器
===============

智能排程系統：
1. 基於性能監控觸發重訓練
2. 定期檢查模型老化
3. 市場變化適應性重訓練
4. 硬體資源最佳化排程
5. 訓練失敗自動重試
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import subprocess
import time
import schedule
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrainingScheduler:
    """定期重訓練排程器"""
    
    def __init__(self, base_dir: str = "/mnt/nvme/taiwan-stock-ai-analyzer-main"):
        self.base_dir = base_dir
        self.db_path = os.path.join(base_dir, "predictions.db")
        self.models_dir = os.path.join(base_dir, "models")
        
        # 重訓練觸發條件
        self.retrain_triggers = {
            'accuracy_threshold': 0.75,     # 準確率低於75%
            'prediction_days': 14,          # 14天內預測數據
            'min_predictions': 5,           # 最少5筆預測
            'age_threshold_days': 30,       # 模型超過30天未更新
            'market_volatility_threshold': 0.05  # 市場波動超過5%
        }
        
        # 模型優先級 (影響重訓練順序)
        self.model_priority = {
            'arima': 1,      # 最快，優先重訓
            'lightgbm': 2,   # 效率高
            'rf': 3,         # 穩定性好
            'xgb': 4,        # 精度高
            'lstm': 5        # 最慢，最後重訓
        }
        
        # 排程配置
        self.schedule_config = {
            'daily_check': "02:00",         # 每日凌晨2點檢查
            'weekly_retrain': "SUN 03:00",  # 每週日凌晨3點重訓練
            'monthly_full': "01 04:00",     # 每月1日凌晨4點全面重訓練
            'emergency_check_hours': 6      # 緊急檢查間隔(小時)
        }
        
        # 訓練狀態追蹤
        self.training_status = {}
        self.last_retrain_times = {}
        
        logger.info("✅ 定期重訓練排程器初始化完成")
    
    def check_model_performance(self, symbol: str, model_name: str) -> Dict:
        """檢查特定股票模型的性能"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 查詢最近預測記錄
            cutoff_date = (datetime.now() - timedelta(days=self.retrain_triggers['prediction_days'])).strftime('%Y-%m-%d')
            
            query = """
                SELECT predicted_price, actual_price, accuracy_score, price_error_pct, prediction_date
                FROM predictions 
                WHERE symbol = ? 
                AND model_name LIKE ?
                AND prediction_date >= ?
                AND actual_price IS NOT NULL
                ORDER BY prediction_date DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, f'%{model_name}%', cutoff_date))
            conn.close()
            
            if len(df) < self.retrain_triggers['min_predictions']:
                return {
                    'needs_retrain': False,
                    'reason': 'insufficient_data',
                    'prediction_count': len(df)
                }
            
            # 計算性能指標
            avg_accuracy = df['accuracy_score'].mean()
            avg_error = df['price_error_pct'].mean()
            recent_accuracy = df.head(5)['accuracy_score'].mean()  # 最近5筆
            
            # 判斷是否需要重訓練
            needs_retrain = False
            reasons = []
            
            if avg_accuracy < self.retrain_triggers['accuracy_threshold']:
                needs_retrain = True
                reasons.append(f'平均準確率過低: {avg_accuracy:.3f} < {self.retrain_triggers["accuracy_threshold"]}')
            
            if recent_accuracy < avg_accuracy * 0.9:  # 最近表現比平均差10%以上
                needs_retrain = True
                reasons.append(f'近期表現下降: {recent_accuracy:.3f} vs {avg_accuracy:.3f}')
            
            if avg_error > 10:  # 平均誤差超過10%
                needs_retrain = True
                reasons.append(f'平均誤差過高: {avg_error:.1f}%')
            
            return {
                'needs_retrain': needs_retrain,
                'reasons': reasons,
                'avg_accuracy': avg_accuracy,
                'avg_error': avg_error,
                'recent_accuracy': recent_accuracy,
                'prediction_count': len(df),
                'latest_prediction': df.iloc[0]['prediction_date'] if len(df) > 0 else None
            }
        
        except Exception as e:
            logger.error(f"❌ {symbol} {model_name} 性能檢查失敗: {e}")
            return {
                'needs_retrain': False,
                'reason': 'error',
                'error': str(e)
            }
    
    def check_model_age(self, symbol: str, model_name: str) -> Dict:
        """檢查模型文件年齡"""
        try:
            model_dir = os.path.join(self.models_dir, model_name)
            
            # 根據模型類型查找主要模型文件
            model_files = {
                'lstm': f'{symbol}_lstm.keras',
                'rf': f'{symbol}_rf_model.pkl',
                'xgb': f'{symbol}_xgb_model.pkl',
                'arima': f'{symbol}_arima_model.pkl',
                'lightgbm': f'{symbol}_lgb_model.pkl'
            }
            
            model_file = os.path.join(model_dir, model_files.get(model_name, ''))
            
            if not os.path.exists(model_file):
                return {
                    'needs_retrain': True,
                    'reason': 'model_not_exists',
                    'age_days': float('inf')
                }
            
            # 計算檔案年齡
            file_time = datetime.fromtimestamp(os.path.getmtime(model_file))
            age_days = (datetime.now() - file_time).days
            
            needs_retrain = age_days > self.retrain_triggers['age_threshold_days']
            
            return {
                'needs_retrain': needs_retrain,
                'reason': f'model_age_{age_days}_days' if needs_retrain else 'model_fresh',
                'age_days': age_days,
                'last_modified': file_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"❌ {symbol} {model_name} 年齡檢查失敗: {e}")
            return {
                'needs_retrain': True,
                'reason': 'age_check_error',
                'error': str(e)
            }
    
    def check_market_volatility(self, symbol: str) -> Dict:
        """檢查市場波動性變化"""
        try:
            import yfinance as yf
            
            ticker_symbol = f"{symbol}.TW" if not symbol.endswith('.TW') else symbol
            ticker = yf.Ticker(ticker_symbol)
            
            # 獲取最近30天數據
            hist = ticker.history(period='30d')
            
            if hist.empty or len(hist) < 10:
                return {
                    'high_volatility': False,
                    'reason': 'insufficient_data'
                }
            
            # 計算波動率
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # 計算最近7天 vs 過去23天的波動率變化
            recent_volatility = returns.tail(7).std()
            historical_volatility = returns.head(23).std()
            
            volatility_change = abs(recent_volatility - historical_volatility) / historical_volatility
            
            high_volatility = (
                volatility > self.retrain_triggers['market_volatility_threshold'] or
                volatility_change > 0.5  # 波動率變化超過50%
            )
            
            return {
                'high_volatility': high_volatility,
                'current_volatility': volatility,
                'recent_volatility': recent_volatility,
                'historical_volatility': historical_volatility,
                'volatility_change': volatility_change,
                'reason': 'high_volatility' if high_volatility else 'normal_volatility'
            }
            
        except Exception as e:
            logger.error(f"❌ {symbol} 波動率檢查失敗: {e}")
            return {
                'high_volatility': False,
                'reason': 'volatility_check_error',
                'error': str(e)
            }
    
    def evaluate_retrain_priority(self, symbol: str, model_name: str) -> Dict:
        """評估重訓練優先級"""
        logger.info(f"🔍 評估 {symbol} {model_name.upper()} 重訓練需求...")
        
        # 檢查各項指標
        performance = self.check_model_performance(symbol, model_name)
        age_check = self.check_model_age(symbol, model_name)
        volatility = self.check_market_volatility(symbol)
        
        # 計算優先級分數 (0-100)
        priority_score = 0
        retrain_reasons = []
        
        # 性能因素 (40分)
        if performance['needs_retrain']:
            if 'insufficient_data' not in performance.get('reason', ''):
                priority_score += 40
                retrain_reasons.extend(performance.get('reasons', []))
        
        # 年齡因素 (30分)
        if age_check['needs_retrain']:
            age_score = min(30, age_check['age_days'] - self.retrain_triggers['age_threshold_days'])
            priority_score += age_score
            retrain_reasons.append(f"模型過舊: {age_check['age_days']}天")
        
        # 波動性因素 (20分)
        if volatility['high_volatility']:
            priority_score += 20
            retrain_reasons.append(f"市場波動過高: {volatility['current_volatility']:.3f}")
        
        # 模型類型調整 (10分)
        model_priority_bonus = (6 - self.model_priority.get(model_name, 3)) * 2
        priority_score += model_priority_bonus
        
        # 綜合評估
        needs_retrain = priority_score >= 30  # 30分以上才重訓練
        
        return {
            'symbol': symbol,
            'model_name': model_name,
            'needs_retrain': needs_retrain,
            'priority_score': priority_score,
            'retrain_reasons': retrain_reasons,
            'performance_check': performance,
            'age_check': age_check,
            'volatility_check': volatility,
            'evaluation_time': datetime.now().isoformat()
        }
    
    def get_retrain_candidates(self) -> List[Dict]:
        """獲取所有需要重訓練的候選"""
        logger.info("🔍 掃描重訓練候選...")
        
        candidates = []
        
        # 獲取所有已訓練的股票
        all_stocks = set()
        for model_name in self.model_priority.keys():
            model_dir = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if '_' in file and file.endswith(('.pkl', '.keras')):
                        symbol = file.split('_')[0]
                        all_stocks.add(symbol)
        
        # 評估每個股票的每個模型
        for symbol in sorted(all_stocks):
            for model_name in self.model_priority.keys():
                evaluation = self.evaluate_retrain_priority(symbol, model_name)
                
                if evaluation['needs_retrain']:
                    candidates.append(evaluation)
        
        # 按優先級排序
        candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"📋 發現 {len(candidates)} 個重訓練候選")
        return candidates
    
    def execute_retrain_batch(self, candidates: List[Dict], max_concurrent: int = 3) -> Dict:
        """執行批次重訓練"""
        logger.info(f"🚀 開始批次重訓練，候選數: {len(candidates)}")
        
        results = {'success': 0, 'failed': 0, 'details': []}
        
        # 按模型類型分組以優化資源使用
        gpu_tasks = [c for c in candidates if c['model_name'] == 'lstm']
        cpu_tasks = [c for c in candidates if c['model_name'] != 'lstm']
        
        # 先執行CPU任務 (可並行)
        if cpu_tasks:
            logger.info(f"💻 執行 {len(cpu_tasks)} 個CPU重訓練任務...")
            cpu_results = self._execute_cpu_retrains(cpu_tasks, max_concurrent)
            results['success'] += cpu_results['success']
            results['failed'] += cpu_results['failed']
            results['details'].extend(cpu_results['details'])
        
        # 再執行GPU任務 (序列)
        if gpu_tasks:
            logger.info(f"🖥️ 執行 {len(gpu_tasks)} 個GPU重訓練任務...")
            gpu_results = self._execute_gpu_retrains(gpu_tasks)
            results['success'] += gpu_results['success']
            results['failed'] += gpu_results['failed']
            results['details'].extend(gpu_results['details'])
        
        logger.info(f"✅ 批次重訓練完成: {results['success']}/{len(candidates)} 成功")
        return results
    
    def _execute_cpu_retrains(self, tasks: List[Dict], max_workers: int) -> Dict:
        """執行CPU重訓練任務"""
        results = {'success': 0, 'failed': 0, 'details': []}
        
        def retrain_single(task):
            symbol = task['symbol']
            model_name = task['model_name']
            
            try:
                # 構建重訓練命令
                script_map = {
                    'rf': 'rf_stock_predictor.py',
                    'xgb': 'xgb_stock_predictor.py', 
                    'arima': 'arima_stock_predictor.py',
                    'lightgbm': 'lightgbm_stock_predictor.py'
                }
                
                script = script_map.get(model_name)
                if not script:
                    return {'success': False, 'error': f'未知模型: {model_name}'}
                
                cmd = ['python3', script, '--symbol', symbol, '--train', '--force']
                
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    timeout=300,
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )
                
                training_time = time.time() - start_time
                
                if result.returncode == 0:
                    return {
                        'success': True,
                        'training_time': training_time,
                        'stdout': result.stdout[-200:] if result.stdout else ''
                    }
                else:
                    return {
                        'success': False,
                        'error': result.stderr[-200:] if result.stderr else '重訓練失敗'
                    }
                    
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # 並行執行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(retrain_single, task): task for task in tasks}
            
            for future in futures:
                task = futures[future]
                try:
                    result = future.result()
                    if result['success']:
                        results['success'] += 1
                        logger.info(f"✅ {task['symbol']} {task['model_name'].upper()} 重訓練成功")
                    else:
                        results['failed'] += 1
                        logger.error(f"❌ {task['symbol']} {task['model_name'].upper()} 重訓練失敗: {result['error']}")
                    
                    results['details'].append({
                        'symbol': task['symbol'],
                        'model': task['model_name'],
                        'result': result
                    })
                    
                except Exception as e:
                    results['failed'] += 1
                    logger.error(f"❌ {task['symbol']} {task['model_name'].upper()} 重訓練異常: {e}")
                    results['details'].append({
                        'symbol': task['symbol'],
                        'model': task['model_name'],
                        'result': {'success': False, 'error': str(e)}
                    })
        
        return results
    
    def _execute_gpu_retrains(self, tasks: List[Dict]) -> Dict:
        """執行GPU重訓練任務 (序列執行避免記憶體衝突)"""
        results = {'success': 0, 'failed': 0, 'details': []}
        
        for task in tasks:
            symbol = task['symbol']
            model_name = task['model_name']
            
            logger.info(f"🖥️ GPU重訓練 {symbol} {model_name.upper()}...")
            
            try:
                cmd = ['python3', 'lstm_stock_predictor.py', '--symbol', symbol, '--train', '--gpu', '--force']
                
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    timeout=600,  # LSTM訓練時間較長
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )
                
                training_time = time.time() - start_time
                
                if result.returncode == 0:
                    results['success'] += 1
                    logger.info(f"✅ {symbol} LSTM 重訓練成功 ({training_time:.1f}s)")
                    task_result = {'success': True, 'training_time': training_time}
                else:
                    results['failed'] += 1
                    logger.error(f"❌ {symbol} LSTM 重訓練失敗")
                    task_result = {'success': False, 'error': result.stderr[-200:] if result.stderr else '重訓練失敗'}
                
            except subprocess.TimeoutExpired:
                results['failed'] += 1
                logger.error(f"⏰ {symbol} LSTM 重訓練超時")
                task_result = {'success': False, 'error': '重訓練超時'}
            except Exception as e:
                results['failed'] += 1
                logger.error(f"❌ {symbol} LSTM 重訓練異常: {e}")
                task_result = {'success': False, 'error': str(e)}
            
            results['details'].append({
                'symbol': symbol,
                'model': model_name,
                'result': task_result
            })
            
            # GPU任務間休息
            time.sleep(5)
        
        return results
    
    def daily_retrain_check(self):
        """每日重訓練檢查"""
        logger.info("🌅 執行每日重訓練檢查...")
        
        candidates = self.get_retrain_candidates()
        
        if not candidates:
            logger.info("✅ 所有模型性能良好，無需重訓練")
            return
        
        # 只重訓練高優先級的 (分數 >= 60)
        high_priority = [c for c in candidates if c['priority_score'] >= 60]
        
        if high_priority:
            logger.info(f"🎯 執行高優先級重訓練: {len(high_priority)} 個任務")
            self.execute_retrain_batch(high_priority, max_concurrent=2)
        else:
            logger.info("📋 發現重訓練候選，但優先級不高，等待週末執行")
    
    def weekly_retrain_batch(self):
        """週末批次重訓練"""
        logger.info("📅 執行週末批次重訓練...")
        
        candidates = self.get_retrain_candidates()
        
        if candidates:
            # 週末可以執行較多任務
            self.execute_retrain_batch(candidates, max_concurrent=4)
        else:
            logger.info("✅ 所有模型狀態良好，無需重訓練")
    
    def setup_schedule(self):
        """設置定期排程"""
        logger.info("⏰ 設置重訓練排程...")
        
        # 每日檢查
        schedule.every().day.at(self.schedule_config['daily_check']).do(self.daily_retrain_check)
        
        # 週末批次重訓練
        schedule.every().sunday.at("03:00").do(self.weekly_retrain_batch)
        
        logger.info("✅ 排程設置完成")
        logger.info(f"📋 每日檢查: {self.schedule_config['daily_check']}")
        logger.info(f"📋 週末重訓練: 週日 03:00")
    
    def run_scheduler(self):
        """執行排程器"""
        logger.info("🚀 啟動重訓練排程器...")
        
        self.setup_schedule()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分鐘檢查一次

def main():
    """測試排程器"""
    print("🧪 測試定期重訓練排程器")
    print("=" * 50)
    
    scheduler = RetrainingScheduler()
    
    # 立即執行一次檢查
    print("\\n🔍 執行重訓練候選掃描...")
    candidates = scheduler.get_retrain_candidates()
    
    if candidates:
        print(f"\\n📋 發現 {len(candidates)} 個重訓練候選:")
        for i, candidate in enumerate(candidates[:5], 1):  # 只顯示前5個
            print(f"  {i}. {candidate['symbol']} {candidate['model_name'].upper()} "
                  f"(優先級: {candidate['priority_score']})")
            print(f"     原因: {', '.join(candidate['retrain_reasons'])}")
        
        if len(candidates) > 5:
            print(f"  ... 還有 {len(candidates) - 5} 個候選")
        
        # 詢問是否執行測試重訓練
        choice = input("\\n是否執行測試重訓練? (y/N): ").strip().lower()
        if choice == 'y':
            # 只執行前3個候選作為測試
            test_candidates = candidates[:3]
            print(f"\\n🚀 測試重訓練 {len(test_candidates)} 個候選...")
            results = scheduler.execute_retrain_batch(test_candidates, max_concurrent=2)
            print(f"\\n📊 測試結果: {results['success']}/{len(test_candidates)} 成功")
    else:
        print("\\n✅ 所有模型狀態良好，無需重訓練")

if __name__ == "__main__":
    main()