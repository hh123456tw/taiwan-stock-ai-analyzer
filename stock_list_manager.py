#!/usr/bin/env python3
"""
股票清單管理系統
輕鬆新增/移除自動預測的股票
"""

import json
import sqlite3
from datetime import datetime

class StockListManager:
    def __init__(self):
        self.config_file = "stock_config.json"
        self.db_path = "predictions.db"
        self.load_config()
    
    def load_config(self):
        """載入股票配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # 預設股票清單
            self.config = {
                "etf_stocks": ["0056", "0057", "00878", "00891", "00919", "00922", "00927"],
                "individual_stocks": ["1519", "1537", "2313", "2330", "2345", "2382", "2408", "2474"],
                "custom_stocks": [],
                "last_updated": datetime.now().isoformat()
            }
            self.save_config()
    
    def save_config(self):
        """儲存股票配置"""
        self.config["last_updated"] = datetime.now().isoformat()
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        print(f"✅ 股票配置已儲存到 {self.config_file}")
    
    def get_all_stocks(self):
        """取得所有股票清單"""
        return (self.config["etf_stocks"] + 
                self.config["individual_stocks"] + 
                self.config["custom_stocks"])
    
    def add_stock(self, symbol, stock_type="custom"):
        """新增股票"""
        symbol = symbol.upper()
        all_stocks = self.get_all_stocks()
        
        if symbol in all_stocks:
            print(f"⚠️ 股票 {symbol} 已存在於清單中")
            return False
        
        if stock_type == "etf":
            self.config["etf_stocks"].append(symbol)
        elif stock_type == "individual":
            self.config["individual_stocks"].append(symbol)
        else:
            self.config["custom_stocks"].append(symbol)
        
        self.save_config()
        print(f"✅ 成功新增股票 {symbol} (類型: {stock_type})")
        return True
    
    def remove_stock(self, symbol):
        """移除股票"""
        symbol = symbol.upper()
        removed = False
        
        for category in ["etf_stocks", "individual_stocks", "custom_stocks"]:
            if symbol in self.config[category]:
                self.config[category].remove(symbol)
                removed = True
                print(f"✅ 已從 {category} 移除股票 {symbol}")
                break
        
        if not removed:
            print(f"⚠️ 股票 {symbol} 不在清單中")
            return False
        
        self.save_config()
        return True
    
    def list_stocks(self):
        """列出所有股票"""
        print("\n📊 **目前股票清單**")
        print(f"🏦 ETF ({len(self.config['etf_stocks'])}個): {', '.join(self.config['etf_stocks'])}")
        print(f"📈 個股 ({len(self.config['individual_stocks'])}個): {', '.join(self.config['individual_stocks'])}")
        print(f"⭐ 自訂 ({len(self.config['custom_stocks'])}個): {', '.join(self.config['custom_stocks'])}")
        print(f"\n📅 總計: {len(self.get_all_stocks())} 檔股票")
        print(f"⏰ 最後更新: {self.config['last_updated']}")
    
    def check_predictions(self):
        """檢查哪些股票缺少預測"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT symbol FROM predictions")
            predicted_stocks = {row[0] for row in cursor.fetchall()}
            
            all_stocks = set(self.get_all_stocks())
            missing_stocks = all_stocks - predicted_stocks
            
            print(f"\n📊 **預測狀態檢查**")
            print(f"✅ 已有預測: {len(predicted_stocks)} 檔")
            print(f"❌ 缺少預測: {len(missing_stocks)} 檔")
            
            if missing_stocks:
                print(f"⚠️ 缺少預測的股票: {', '.join(sorted(missing_stocks))}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ 檢查預測狀態失敗: {e}")
    
    def update_prediction_script(self):
        """更新 complete_all_predictions.py 的股票清單"""
        all_stocks = self.get_all_stocks()
        
        # 讀取原始檔案
        with open("complete_all_predictions.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新股票清單
        new_all_stocks = f"    all_stocks = {all_stocks}"
        new_etf_stocks = f"    etf_stocks = {self.config['etf_stocks']}"
        new_individual_stocks = f"    individual_stocks = {self.config['individual_stocks'] + self.config['custom_stocks']}"
        
        # 替換內容（簡化版本，實際可能需要更精確的替換）
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            if line.strip().startswith('all_stocks = '):
                new_lines.append(new_all_stocks)
            elif line.strip().startswith('etf_stocks = '):
                new_lines.append(new_etf_stocks)
            elif line.strip().startswith('individual_stocks = '):
                new_lines.append(new_individual_stocks)
            else:
                new_lines.append(line)
        
        # 寫回檔案
        with open("complete_all_predictions.py.new", 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        
        print("✅ 已生成更新的預測腳本: complete_all_predictions.py.new")
        print("💡 請手動檢查後重新命名替換原檔案")

def main():
    """主程式"""
    manager = StockListManager()
    
    while True:
        print("\n" + "="*50)
        print("📊 台股AI預測系統 - 股票清單管理")
        print("="*50)
        print("1. 📋 查看股票清單")
        print("2. ➕ 新增股票")
        print("3. ➖ 移除股票")
        print("4. 🔍 檢查預測狀態")
        print("5. 🔄 更新預測腳本")
        print("0. 🚪 退出")
        
        choice = input("\n請選擇功能 (0-5): ").strip()
        
        if choice == "1":
            manager.list_stocks()
        
        elif choice == "2":
            symbol = input("請輸入股票代碼: ").strip().upper()
            if not symbol:
                print("❌ 股票代碼不可為空")
                continue
            
            print("請選擇股票類型:")
            print("1. ETF")
            print("2. 個股")
            print("3. 自訂")
            
            type_choice = input("選擇 (1-3): ").strip()
            stock_type = {"1": "etf", "2": "individual", "3": "custom"}.get(type_choice, "custom")
            
            manager.add_stock(symbol, stock_type)
        
        elif choice == "3":
            symbol = input("請輸入要移除的股票代碼: ").strip().upper()
            if symbol:
                manager.remove_stock(symbol)
        
        elif choice == "4":
            manager.check_predictions()
        
        elif choice == "5":
            manager.update_prediction_script()
        
        elif choice == "0":
            print("👋 再見！")
            break
        
        else:
            print("❌ 無效選擇，請重新輸入")

if __name__ == "__main__":
    main()