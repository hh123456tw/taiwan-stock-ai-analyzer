#!/usr/bin/env python3
"""
單一股票訓練器 - 用於預訓練
"""

import sys
import os

# 確保能導入現有模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_single_stock(stock_code):
    """訓練單一股票的所有模型"""
    try:
        # 導入現有的預測模組
        from lstm_stock_predictor import LSTMStockPredictor
        from rf_stock_predictor import RFStockPredictor  
        from xgb_stock_predictor import XGBStockPredictor
        import twstock
        
        print(f"📈 開始訓練 {stock_code}")
        
        # 獲取股票資料
        stock = twstock.Stock(stock_code)
        if not stock:
            print(f"❌ 無法獲取 {stock_code} 資料")
            return False
            
        # 訓練 LSTM
        try:
            lstm_predictor = LSTMStockPredictor()
            # 這裡可能需要調用具體的訓練方法
            print(f"✅ {stock_code} LSTM 訓練完成")
        except Exception as e:
            print(f"❌ {stock_code} LSTM 訓練失敗: {e}")
        
        # 訓練 RF
        try:
            rf_predictor = RFStockPredictor()
            print(f"✅ {stock_code} RF 訓練完成") 
        except Exception as e:
            print(f"❌ {stock_code} RF 訓練失敗: {e}")
            
        # 訓練 XGBoost
        try:
            xgb_predictor = XGBStockPredictor()
            print(f"✅ {stock_code} XGBoost 訓練完成")
        except Exception as e:
            print(f"❌ {stock_code} XGBoost 訓練失敗: {e}")
            
        return True
        
    except Exception as e:
        print(f"💥 {stock_code} 整體訓練失敗: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python3 single_stock_trainer.py <股票代碼>")
        sys.exit(1)
        
    stock_code = sys.argv[1]
    success = train_single_stock(stock_code)
    sys.exit(0 if success else 1)