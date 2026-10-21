#!/usr/bin/env python3
"""
批次模型訓練器
=============

專門補齊缺失的模型訓練：
1. ARIMA模型: 0/59 (完全缺失)
2. LightGBM模型: 0/59 (完全缺失)  
3. LSTM模型: 21/59 (需補38檔)
4. XGB模型: 55/59 (需補4檔)
"""

import os
import json
import subprocess
import time
import logging
from datetime import datetime
from typing import Dict, List, Set
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchModelTrainer:
    """批次模型訓練器"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.config_file = os.path.join(base_dir, "stock_pool_config.json")
        
        # 載入股票池配置
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.active_stocks = config.get('active_stocks', [])
        
        # 模型訓練配置
        self.model_configs = {
            'arima': {
                'script': 'arima_stock_predictor.py',
                'file_pattern': '{}_arima_model.pkl',
                'max_workers': 6,  # 最輕量，可高併發
                'timeout': 120
            },
            'lightgbm': {
                'script': 'lightgbm_stock_predictor.py', 
                'file_pattern': '{}_lgb_model.pkl',
                'max_workers': 4,
                'timeout': 180
            },
            'xgb': {
                'script': 'xgb_stock_predictor.py',
                'file_pattern': '{}_xgb_model.pkl', 
                'max_workers': 3,
                'timeout': 240
            },
            'lstm': {
                'script': 'lstm_stock_predictor.py',
                'file_pattern': '{}_lstm.keras',
                'max_workers': 1,  # GPU序列
                'timeout': 300
            },
            'rf': {
                'script': 'rf_stock_predictor.py',
                'file_pattern': '{}_rf_model.pkl',
                'max_workers': 4,
                'timeout': 180
            }
        }
        
        logger.info(f"✅ 批次訓練器初始化完成，股票池: {len(self.active_stocks)}檔")
    
    def get_trained_stocks(self, model_name: str) -> Set[str]:
        """獲取已訓練的股票"""
        trained_stocks = set()
        model_dir = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_dir):
            logger.warning(f"⚠️ {model_name} 目錄不存在: {model_dir}")
            return trained_stocks
        
        file_pattern = self.model_configs[model_name]['file_pattern']
        
        for file in os.listdir(model_dir):
            if file.endswith(('.pkl', '.keras')) and '_' in file:
                # 從檔案名提取股票代碼
                symbol = file.split('_')[0]
                
                # 檢查是否符合預期的檔案格式
                expected_file = file_pattern.format(symbol)
                if file == expected_file or file == expected_file.replace('.pkl', '_model.pkl'):
                    trained_stocks.add(symbol)
        
        return trained_stocks
    
    def get_missing_stocks(self, model_name: str) -> List[str]:
        """獲取需要訓練的股票"""
        trained_stocks = self.get_trained_stocks(model_name)
        missing_stocks = [stock for stock in self.active_stocks if stock not in trained_stocks]
        return missing_stocks
    
    def train_single_stock(self, model_name: str, symbol: str) -> Dict:
        """訓練單一股票的特定模型"""
        config = self.model_configs[model_name]
        script = config['script']
        timeout = config['timeout']
        
        try:
            cmd = ['python3', script, '--symbol', symbol, '--train']
            
            # LSTM需要GPU參數
            if model_name == 'lstm':
                cmd.append('--gpu')
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=self.base_dir
            )
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'symbol': symbol,
                    'model': model_name,
                    'training_time': training_time,
                    'stdout': result.stdout[-200:] if result.stdout else ''
                }
            else:
                return {
                    'success': False,
                    'symbol': symbol,
                    'model': model_name,
                    'error': result.stderr[-200:] if result.stderr else '未知錯誤',
                    'stdout': result.stdout[-200:] if result.stdout else ''
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'symbol': symbol,
                'model': model_name, 
                'error': f'訓練超時 (>{timeout}s)'
            }
        except Exception as e:
            return {
                'success': False,
                'symbol': symbol,
                'model': model_name,
                'error': str(e)
            }
    
    def batch_train_model(self, model_name: str, stocks: List[str]) -> Dict:
        """批次訓練特定模型"""
        if not stocks:
            return {'success_count': 0, 'failed_count': 0, 'details': []}
        
        config = self.model_configs[model_name]
        max_workers = config['max_workers']
        
        logger.info(f"🚀 開始批次訓練 {model_name.upper()}: {len(stocks)} 檔股票")
        
        results = {'success_count': 0, 'failed_count': 0, 'details': []}
        
        if model_name == 'lstm':
            # GPU模型序列執行
            for symbol in stocks:
                result = self.train_single_stock(model_name, symbol)
                
                if result['success']:
                    results['success_count'] += 1
                    logger.info(f"✅ {symbol} LSTM ({result['training_time']:.1f}s)")
                else:
                    results['failed_count'] += 1
                    logger.error(f"❌ {symbol} LSTM: {result['error']}")
                
                results['details'].append(result)
                time.sleep(2)  # GPU間隔
        else:
            # CPU模型並行執行
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.train_single_stock, model_name, symbol): symbol 
                    for symbol in stocks
                }
                
                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        
                        if result['success']:
                            results['success_count'] += 1
                            logger.info(f"✅ {symbol} {model_name.upper()} ({result.get('training_time', 0):.1f}s)")
                        else:
                            results['failed_count'] += 1
                            logger.error(f"❌ {symbol} {model_name.upper()}: {result['error']}")
                        
                        results['details'].append(result)
                        
                    except Exception as e:
                        results['failed_count'] += 1
                        error_result = {
                            'success': False,
                            'symbol': symbol,
                            'model': model_name,
                            'error': str(e)
                        }
                        results['details'].append(error_result)
                        logger.error(f"❌ {symbol} {model_name.upper()}: {e}")
        
        success_rate = results['success_count'] / len(stocks) * 100 if stocks else 0
        logger.info(f"🎉 {model_name.upper()} 批次訓練完成: {results['success_count']}/{len(stocks)} ({success_rate:.1f}%)")
        
        return results
    
    def analyze_training_gaps(self) -> Dict:
        """分析訓練缺口"""
        analysis = {}
        total_stocks = len(self.active_stocks)
        
        print("📊 模型訓練缺口分析")
        print("=" * 50)
        print(f"股票池大小: {total_stocks} 檔")
        print()
        
        for model_name in self.model_configs.keys():
            trained_stocks = self.get_trained_stocks(model_name)
            missing_stocks = self.get_missing_stocks(model_name)
            
            coverage = len(trained_stocks) / total_stocks * 100
            
            status = "✅" if len(missing_stocks) == 0 else "❌" if len(missing_stocks) > 10 else "⚠️"
            
            print(f"{status} {model_name.upper():<8}: {len(trained_stocks):>2}/{total_stocks} ({coverage:>5.1f}%) - 缺少{len(missing_stocks)}檔")
            
            analysis[model_name] = {
                'trained_count': len(trained_stocks),
                'missing_count': len(missing_stocks),
                'coverage_pct': coverage,
                'missing_stocks': missing_stocks[:10]  # 只記錄前10個
            }
        
        return analysis
    
    def run_comprehensive_training(self) -> Dict:
        """執行全面補齊訓練"""
        logger.info("🚀 開始全面模型補齊訓練...")
        
        start_time = time.time()
        overall_results = {}
        
        # 按優先級順序訓練 (快的先做)
        training_order = ['arima', 'lightgbm', 'xgb', 'lstm']
        
        for model_name in training_order:
            missing_stocks = self.get_missing_stocks(model_name)
            
            if missing_stocks:
                logger.info(f"\\n📋 {model_name.upper()} 需要訓練 {len(missing_stocks)} 檔股票")
                
                # 分批訓練避免過載
                batch_size = 20 if model_name != 'lstm' else 15
                batches = [missing_stocks[i:i+batch_size] for i in range(0, len(missing_stocks), batch_size)]
                
                model_results = {'success_count': 0, 'failed_count': 0, 'details': []}
                
                for i, batch in enumerate(batches, 1):
                    logger.info(f"🎯 {model_name.upper()} 批次 {i}/{len(batches)}: {batch}")
                    batch_result = self.batch_train_model(model_name, batch)
                    
                    model_results['success_count'] += batch_result['success_count']
                    model_results['failed_count'] += batch_result['failed_count']
                    model_results['details'].extend(batch_result['details'])
                    
                    # 批次間休息
                    if i < len(batches):
                        time.sleep(5)
                
                overall_results[model_name] = model_results
            else:
                logger.info(f"✅ {model_name.upper()} 所有股票已訓練完成")
                overall_results[model_name] = {'success_count': 0, 'failed_count': 0, 'details': []}
        
        total_time = time.time() - start_time
        
        # 生成總結報告
        total_success = sum(r['success_count'] for r in overall_results.values())
        total_failed = sum(r['failed_count'] for r in overall_results.values())
        total_trained = total_success + total_failed
        
        summary = {
            'total_time': total_time,
            'total_trained': total_trained,
            'total_success': total_success,
            'total_failed': total_failed,
            'success_rate': total_success / total_trained * 100 if total_trained > 0 else 0,
            'model_results': overall_results
        }
        
        logger.info(f"\\n🎉 全面補齊訓練完成!")
        logger.info(f"📊 總結: {total_success}/{total_trained} 成功 ({summary['success_rate']:.1f}%)")
        logger.info(f"⏱️ 總耗時: {total_time:.1f} 秒")
        
        return summary

def main():
    """主程序"""
    print("🛠️ 批次模型訓練器")
    print("=" * 50)
    
    trainer = BatchModelTrainer()
    
    # 分析訓練缺口
    analysis = trainer.analyze_training_gaps()
    
    # 統計需要訓練的總數
    total_missing = sum(info['missing_count'] for info in analysis.values())
    
    if total_missing == 0:
        print("\\n✅ 所有模型都已訓練完成，無需補齊")
        return
    
    print(f"\\n📋 發現 {total_missing} 個缺失模型需要訓練")
    
    # 詢問是否執行訓練
    choice = input("\\n是否開始批次補齊訓練? (y/N): ").strip().lower()
    
    if choice == 'y':
        print("\\n🚀 開始批次補齊訓練...")
        results = trainer.run_comprehensive_training()
        
        print(f"\\n📈 最終結果:")
        for model, result in results['model_results'].items():
            if result['success_count'] > 0 or result['failed_count'] > 0:
                print(f"  {model.upper()}: +{result['success_count']} 成功, {result['failed_count']} 失敗")
        
        print(f"\\n⏱️ 總耗時: {results['total_time']:.1f} 秒")
        print("🎉 批次補齊完成!")
    else:
        print("👋 取消訓練，再見!")

if __name__ == "__main__":
    main()