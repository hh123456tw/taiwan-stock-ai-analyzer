#!/usr/bin/env python3
"""
快速檢查實時交易系統狀態
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
import subprocess

def check_service_status():
    """檢查服務狀態"""
    try:
        result = subprocess.run(['systemctl', 'is-active', 'taiwan-realtime-trading'], 
                              capture_output=True, text=True)
        status = result.stdout.strip()
        print(f"🔧 服務狀態: {status}")
        
        if status == 'active':
            print("✅ 實時交易系統正在運行")
        else:
            print("❌ 實時交易系統未運行")
            print("   啟動指令: sudo systemctl start taiwan-realtime-trading")
            
    except Exception as e:
        print(f"❌ 檢查服務狀態失敗: {e}")

def check_trading_activity():
    """檢查交易活動"""
    today = datetime.now().strftime('%Y%m%d')
    
    # 檢查交易日誌
    log_file = f"/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system/logs/trading_system_{today}.log"
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"📊 今日日誌記錄: {len(lines)} 行")
        if lines:
            print(f"🕐 最後記錄時間: {lines[-1][:19] if lines else '無'}")
    else:
        print("⚠️ 今日交易日誌尚未生成")
    
    # 檢查交易會話
    session_file = f"/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system/logs/trading_sessions_{today}.json"
    if os.path.exists(session_file):
        with open(session_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        print(f"🔍 今日掃描次數: {len(sessions)}")
        
        if sessions:
            last_session = sessions[-1]
            print(f"📈 發現機會: {last_session.get('candidates_found', 0)} 個")
            print(f"🤖 AI決策: {last_session.get('decisions_made', 0)} 次")
            
            daily_stats = last_session.get('daily_stats', {})
            trades = daily_stats.get('trades', 0)
            pnl = daily_stats.get('pnl', 0)
            print(f"💼 執行交易: {trades} 筆")
            print(f"💰 今日損益: ${pnl:+.0f}")
    else:
        print("⚠️ 今日交易會話記錄尚未生成")

def check_portfolio():
    """檢查投資組合"""
    portfolio_file = "/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system/data/portfolio.json"
    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r', encoding='utf-8') as f:
            portfolio = json.load(f)
        
        print(f"\n💰 投資組合狀態:")
        print(f"💵 現金餘額: ${portfolio.get('cash', 0):,.0f}")
        print(f"📊 持股檔數: {len(portfolio.get('positions', {}))}")
        print(f"📈 今日損益: ${portfolio.get('daily_pnl', 0):+.0f}")
        print(f"📊 總損益: ${portfolio.get('total_pnl', 0):+.0f}")
        
        positions = portfolio.get('positions', {})
        if positions:
            print("\n📋 持股明細:")
            for stock, pos in positions.items():
                print(f"  {stock}: {pos['quantity']}股 @${pos['avg_price']:.2f}")
    else:
        print("⚠️ 投資組合文件尚未生成")

def check_predictions_db():
    """檢查AI預測數據"""
    db_path = "/mnt/nvme/taiwan-stock-ai-analyzer-main/predictions.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 檢查今日預測
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE DATE(created_at) = ?
            """, (today,))
            today_predictions = cursor.fetchone()[0]
            
            # 檢查最新預測
            cursor.execute("""
                SELECT stock_code, prediction_price, confidence, created_at
                FROM predictions 
                ORDER BY created_at DESC LIMIT 5
            """)
            latest = cursor.fetchall()
            
            print(f"\n🧠 AI預測數據:")
            print(f"📊 今日預測數量: {today_predictions}")
            
            if latest:
                print("📈 最新預測 (前5筆):")
                for stock, price, conf, time in latest:
                    print(f"  {stock}: ${price:.2f} (信心度:{conf:.3f}) - {time[:16]}")
            
            conn.close()
        except Exception as e:
            print(f"❌ 讀取預測數據失敗: {e}")
    else:
        print("❌ 預測數據庫不存在")

def is_trading_time():
    """檢查是否在交易時間"""
    now = datetime.now()
    weekday = now.weekday()
    current_time = now.time()
    
    # 週一到週五
    if weekday < 5:
        # 9:30 到 13:00
        if current_time >= datetime.strptime('09:30', '%H:%M').time() and \
           current_time <= datetime.strptime('13:00', '%H:%M').time():
            return True, "交易時間內"
        elif current_time < datetime.strptime('09:30', '%H:%M').time():
            return False, "尚未到交易時間"
        else:
            return False, "已過交易時間"
    else:
        return False, "週末非交易日"

def main():
    print("🚀 台股AI實時交易系統 - 快速狀態檢查")
    print("=" * 50)
    
    # 檢查交易時間
    is_trading, time_status = is_trading_time()
    print(f"⏰ 當前時間狀態: {time_status}")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if is_trading:
        print("🟢 系統應該正在執行實時交易")
    else:
        print("🟡 系統處於待機狀態")
    
    print("\n" + "=" * 50)
    
    # 檢查各項狀態
    check_service_status()
    print("\n" + "-" * 30)
    check_trading_activity()
    print("\n" + "-" * 30)
    check_portfolio()
    print("\n" + "-" * 30)
    check_predictions_db()
    
    print("\n" + "=" * 50)
    print("📱 常用監控指令:")
    print("  完整監控: bash monitor_trading.sh")
    print("  實時日誌: tail -f logs/trading_system_$(date +%Y%m%d).log")
    print("  服務狀態: sudo systemctl status taiwan-realtime-trading")

if __name__ == "__main__":
    main()