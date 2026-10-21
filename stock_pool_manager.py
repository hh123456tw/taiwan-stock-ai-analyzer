#!/usr/bin/env python3
"""
股票池管理工具
=============

功能：
1. 查看當前股票池狀態
2. 新增股票 (自動驗證 + 5模型訓練)
3. 移除問題股票
4. 股票池配置文件管理
5. 訓練狀態監控
"""

import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPoolManager:
    """股票池管理器"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.config_file = os.path.join(base_dir, "stock_pool_config.json")
        
        # 從新增股票訓練器導入
        import sys
        sys.path.append(base_dir)
        from new_stock_auto_trainer import NewStockAutoTrainer
        self.trainer = NewStockAutoTrainer(base_dir)
        
        # 載入或創建股票池配置
        self.stock_pool_config = self._load_or_create_config()
        
        logger.info("✅ 股票池管理器初始化完成")
    
    def _load_or_create_config(self) -> Dict:
        """載入或創建股票池配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"📋 載入現有配置: {len(config.get('active_stocks', []))}檔股票")
                return config
            except Exception as e:
                logger.warning(f"⚠️ 配置文件載入失敗: {e}，創建新配置")
        
        # 創建預設配置
        default_config = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "active_stocks": [
                # 科技權值股
                "2330", "2454", "2317", "2303", "2308", "3711", "2412", "2382", "2395", "3034",
                "2408", "3231", "6166", "2354", "2344", "6209", "2313",
                # 金融股  
                "2882", "2881", "2892", "2891", "2884", "2886", "5880", "2890", "2880", "2883",
                # 傳產股
                "1101", "1303", "1216", "2002", "6505", "2207", "2301", "1402", "2105", "2474", "1802",
                # 熱門ETF
                "0056", "00878", "00900", "00919", "00929", "00915", "00918", "00731",
                "0050", "00922", "00850", "00692", "00757", "00895",
                "00891", "00892", "00927", "00830", "00904", "00881"
            ],
            "removed_stocks": [
                # 記錄已移除的股票和原因
            ],
            "model_types": ["lstm", "rf", "xgb", "arima", "lightgbm"],
            "auto_training": True
        }
        
        self._save_config(default_config)
        logger.info(f"📋 創建新配置: {len(default_config['active_stocks'])}檔股票")
        return default_config
    
    def _save_config(self, config: Dict):
        """儲存配置文件"""
        try:
            config["last_updated"] = datetime.now().isoformat()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info("💾 配置文件已儲存")
        except Exception as e:
            logger.error(f"❌ 配置文件儲存失敗: {e}")
    
    def get_stock_training_status(self, symbol: str) -> Dict:
        """獲取股票訓練狀態"""
        status = {"symbol": symbol, "models": {}}
        
        model_files = {
            "lstm": f"{symbol}_lstm.keras",
            "rf": f"{symbol}_rf_model.pkl", 
            "xgb": f"{symbol}_xgb_model.pkl",
            "arima": f"{symbol}_arima_model.pkl",
            "lightgbm": f"{symbol}_lgb_model.pkl"
        }
        
        for model, filename in model_files.items():
            model_path = os.path.join(self.models_dir, model, filename)
            if os.path.exists(model_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                status["models"][model] = {
                    "trained": True,
                    "last_updated": file_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "file_size": os.path.getsize(model_path)
                }
            else:
                status["models"][model] = {"trained": False}
        
        # 計算完成度
        trained_count = sum(1 for m in status["models"].values() if m["trained"])
        status["completion_rate"] = trained_count / len(model_files) * 100
        
        return status
    
    def show_stock_pool_status(self):
        """顯示股票池整體狀態"""
        print("📊 股票池狀態總覽")
        print("=" * 60)
        
        active_stocks = self.stock_pool_config.get("active_stocks", [])
        removed_stocks = self.stock_pool_config.get("removed_stocks", [])
        
        print(f"活躍股票數: {len(active_stocks)}檔")
        print(f"已移除股票: {len(removed_stocks)}檔")
        print(f"上次更新: {self.stock_pool_config.get('last_updated', 'N/A')}")
        
        # 統計各模型訓練狀況
        model_stats = {}
        for model in ["lstm", "rf", "xgb", "arima", "lightgbm"]:
            model_dir = os.path.join(self.models_dir, model)
            if os.path.exists(model_dir):
                count = len([f for f in os.listdir(model_dir) 
                           if f.endswith(('.pkl', '.keras')) and '_' in f])
                model_stats[model] = count
            else:
                model_stats[model] = 0
        
        print(f"\\n🤖 模型訓練統計:")
        for model, count in model_stats.items():
            coverage = count / len(active_stocks) * 100 if active_stocks else 0
            print(f"  {model.upper():<8}: {count:>3}檔 ({coverage:>5.1f}%)")
        
        # 顯示問題股票
        print(f"\\n⚠️ 需要檢查的股票:")
        problem_count = 0
        for symbol in active_stocks[:10]:  # 只檢查前10檔避免太慢
            status = self.get_stock_training_status(symbol)
            if status["completion_rate"] < 80:  # 少於80%完成度
                print(f"  {symbol}: {status['completion_rate']:.1f}% 完成")
                problem_count += 1
        
        if problem_count == 0:
            print("  無問題股票 ✅")
    
    def add_stock(self, symbol: str, auto_train: bool = True) -> Dict:
        """新增股票到股票池"""
        print(f"\\n🎯 新增股票: {symbol}")
        print("-" * 40)
        
        # 檢查是否已存在
        active_stocks = self.stock_pool_config.get("active_stocks", [])
        if symbol in active_stocks:
            return {"success": False, "error": f"{symbol} 已在股票池中"}
        
        # 使用訓練器進行完整驗證和訓練
        if auto_train:
            result = self.trainer.add_new_stock_complete(symbol)
            
            if result["success"]:
                # 加入配置文件
                active_stocks.append(symbol)
                self.stock_pool_config["active_stocks"] = sorted(active_stocks)
                self._save_config(self.stock_pool_config)
                
                print(f"✅ {symbol} 成功加入股票池")
                print(f"   訓練結果: {result['training_results']['success_count']}/5 模型成功")
                print(f"   總耗時: {result['total_time']:.1f}秒")
                
                return {"success": True, "training_result": result}
            else:
                print(f"❌ {symbol} 新增失敗: {result.get('error', '未知錯誤')}")
                return {"success": False, "error": result.get('error')}
        else:
            # 只驗證不訓練
            validation = self.trainer.validate_stock_symbol(symbol)
            if validation["valid"]:
                active_stocks.append(symbol)
                self.stock_pool_config["active_stocks"] = sorted(active_stocks)
                self._save_config(self.stock_pool_config)
                
                print(f"✅ {symbol} 已加入股票池 (未訓練)")
                return {"success": True, "training_result": None}
            else:
                print(f"❌ {symbol} 驗證失敗: {validation['error']}")
                return {"success": False, "error": validation["error"]}
    
    def remove_stock(self, symbol: str, reason: str = "用戶移除") -> Dict:
        """從股票池移除股票"""
        print(f"\\n🗑️ 移除股票: {symbol}")
        print("-" * 40)
        
        active_stocks = self.stock_pool_config.get("active_stocks", [])
        removed_stocks = self.stock_pool_config.get("removed_stocks", [])
        
        if symbol not in active_stocks:
            return {"success": False, "error": f"{symbol} 不在股票池中"}
        
        # 移除股票
        active_stocks.remove(symbol)
        removed_stocks.append({
            "symbol": symbol,
            "removed_date": datetime.now().isoformat(),
            "reason": reason
        })
        
        self.stock_pool_config["active_stocks"] = active_stocks
        self.stock_pool_config["removed_stocks"] = removed_stocks
        self._save_config(self.stock_pool_config)
        
        print(f"✅ {symbol} 已從股票池移除")
        print(f"   移除原因: {reason}")
        
        # 詢問是否刪除模型文件
        delete_models = input(f"是否刪除 {symbol} 的模型文件? (y/N): ").strip().lower()
        if delete_models == 'y':
            self._delete_stock_models(symbol)
        
        return {"success": True}
    
    def _delete_stock_models(self, symbol: str):
        """刪除股票的所有模型文件"""
        deleted_count = 0
        
        model_patterns = {
            "lstm": f"{symbol}_lstm.keras",
            "rf": f"{symbol}_rf_model.pkl",
            "xgb": f"{symbol}_xgb_model.pkl", 
            "arima": f"{symbol}_arima_model.pkl",
            "lightgbm": f"{symbol}_lgb_model.pkl"
        }
        
        for model, filename in model_patterns.items():
            model_path = os.path.join(self.models_dir, model, filename)
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    deleted_count += 1
                    # 同時刪除相關文件
                    related_files = [
                        f"{symbol}_scaler.pkl",
                        f"{symbol}_training_history.png"
                    ]
                    for related in related_files:
                        related_path = os.path.join(self.models_dir, model, related)
                        if os.path.exists(related_path):
                            os.remove(related_path)
                except Exception as e:
                    logger.warning(f"⚠️ 刪除 {model_path} 失敗: {e}")
        
        print(f"🗑️ 已刪除 {deleted_count} 個模型文件")
    
    def batch_train_missing_models(self):
        """批次訓練缺失的模型"""
        print("\\n🚀 批次訓練缺失模型")
        print("-" * 40)
        
        active_stocks = self.stock_pool_config.get("active_stocks", [])
        
        # 統計缺失情況
        missing_stats = {"lstm": [], "rf": [], "xgb": [], "arima": [], "lightgbm": []}
        
        for symbol in active_stocks:
            status = self.get_stock_training_status(symbol)
            for model, info in status["models"].items():
                if not info["trained"]:
                    missing_stats[model].append(symbol)
        
        # 顯示統計
        total_missing = sum(len(stocks) for stocks in missing_stats.values())
        print(f"發現 {total_missing} 個缺失模型:")
        
        for model, stocks in missing_stats.items():
            if stocks:
                print(f"  {model.upper()}: {len(stocks)}檔 - {stocks[:5]}{'...' if len(stocks) > 5 else ''}")
        
        if total_missing == 0:
            print("✅ 所有模型都已訓練完成")
            return
        
        # 詢問是否執行訓練
        choice = input(f"\\n是否開始批次訓練? (y/N): ").strip().lower()
        if choice != 'y':
            return
        
        # 執行訓練 (優先ARIMA和LightGBM)
        for model in ["arima", "lightgbm", "rf", "xgb", "lstm"]:
            stocks = missing_stats[model]
            if stocks:
                print(f"\\n🎯 開始訓練 {model.upper()} 模型...")
                self._train_model_batch(model, stocks[:10])  # 限制一次最多10檔
    
    def _train_model_batch(self, model: str, stocks: List[str]):
        """執行特定模型的批次訓練"""
        script_map = {
            "lstm": "lstm_stock_predictor.py",
            "rf": "rf_stock_predictor.py",
            "xgb": "xgb_stock_predictor.py",
            "arima": "arima_stock_predictor.py",
            "lightgbm": "lightgbm_stock_predictor.py"
        }
        
        script = script_map.get(model)
        if not script:
            print(f"❌ {model} 訓練腳本不存在")
            return
        
        success_count = 0
        for i, symbol in enumerate(stocks, 1):
            print(f"  [{i}/{len(stocks)}] 訓練 {symbol}...", end=" ")
            
            try:
                cmd = ["python3", script, "--symbol", symbol, "--train"]
                if model == "lstm":
                    cmd.append("--gpu")
                
                result = subprocess.run(
                    cmd,
                    timeout=300,
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )
                
                if result.returncode == 0:
                    print("✅")
                    success_count += 1
                else:
                    print("❌")
                    
            except Exception as e:
                print(f"❌ ({e})")
        
        print(f"🎉 {model.upper()} 批次訓練完成: {success_count}/{len(stocks)} 成功")

def main():
    """主程序界面"""
    manager = StockPoolManager()
    
    while True:
        print("\\n" + "="*60)
        print("🏢 股票池管理系統")
        print("="*60)
        print("1. 查看股票池狀態")
        print("2. 新增股票")
        print("3. 移除股票")
        print("4. 批次訓練缺失模型")
        print("5. 退出")
        
        choice = input("\\n請選擇操作 (1-5): ").strip()
        
        if choice == "1":
            manager.show_stock_pool_status()
            
        elif choice == "2":
            symbol = input("請輸入股票代碼: ").strip().upper()
            if symbol:
                auto_train = input("是否自動訓練5個AI模型? (Y/n): ").strip().lower()
                manager.add_stock(symbol, auto_train != 'n')
            
        elif choice == "3":
            symbol = input("請輸入要移除的股票代碼: ").strip().upper()
            reason = input("移除原因 (可選): ").strip() or "用戶移除"
            if symbol:
                manager.remove_stock(symbol, reason)
                
        elif choice == "4":
            manager.batch_train_missing_models()
            
        elif choice == "5":
            print("👋 再見！")
            break
            
        else:
            print("❌ 無效選擇，請重新輸入")

if __name__ == "__main__":
    main()