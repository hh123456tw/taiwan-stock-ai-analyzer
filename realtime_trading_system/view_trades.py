#!/usr/bin/env python3
"""
查看實時交易詳細記錄
"""

import os
import json
from datetime import datetime, timedelta
import argparse

def view_daily_sessions(date_str=None):
    """查看指定日期的交易會話"""
    if not date_str:
        date_str = datetime.now().strftime('%Y%m%d')
    
    session_file = f"/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system/logs/trading_sessions_{date_str}.json"
    
    if not os.path.exists(session_file):
        print(f"❌ {date_str} 的交易會話記錄不存在")
        return
    
    with open(session_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
    
    print(f"📊 {date_str} 交易會話詳細記錄")
    print("=" * 60)
    
    total_candidates = 0
    total_decisions = 0
    total_trades = 0
    
    for i, session in enumerate(sessions, 1):
        timestamp = session.get('timestamp', '')
        candidates = session.get('candidates_found', 0)
        decisions = session.get('decisions_made', 0)
        daily_stats = session.get('daily_stats', {})
        trades = daily_stats.get('trades', 0)
        pnl = daily_stats.get('pnl', 0)
        
        print(f"🔍 第{i}次掃描 - {timestamp[:16]}")
        print(f"   候選股票: {candidates} 個")
        print(f"   AI決策: {decisions} 次")
        print(f"   累計交易: {trades} 筆")
        print(f"   累計損益: ${pnl:+.0f}")
        
        # 顯示投資組合狀態
        portfolio = session.get('portfolio_state', {})
        if portfolio:
            cash = portfolio.get('cash', 0)
            positions = len(portfolio.get('positions', {}))
            print(f"   現金餘額: ${cash:,.0f}")
            print(f"   持股檔數: {positions}")
        
        print("-" * 40)
        
        total_candidates += candidates
        total_decisions += decisions
        total_trades = trades  # 累計值
    
    print(f"📈 當日統計:")
    print(f"   總掃描次數: {len(sessions)}")
    print(f"   發現機會總數: {total_candidates}")
    print(f"   AI決策總數: {total_decisions}")
    print(f"   執行交易總數: {total_trades}")

def view_daily_report(date_str=None):
    """查看每日交易報告"""
    if not date_str:
        date_str = datetime.now().strftime('%Y%m%d')
    
    report_file = f"/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system/logs/daily_report_{date_str}.json"
    
    if not os.path.exists(report_file):
        print(f"❌ {date_str} 的每日報告不存在 (收盤後才會生成)")
        return
    
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    print(f"📋 {date_str} 每日交易報告")
    print("=" * 50)
    
    # 交易統計
    stats = report.get('trading_stats', {})
    print(f"💼 交易統計:")
    print(f"   交易筆數: {stats.get('trades', 0)}")
    print(f"   決策次數: {stats.get('decisions', 0)}")
    print(f"   當日損益: ${stats.get('pnl', 0):+.0f}")
    
    # 投資組合
    portfolio = report.get('portfolio', {})
    print(f"\n💰 投資組合:")
    print(f"   現金餘額: ${portfolio.get('cash', 0):,.0f}")
    print(f"   持股檔數: {len(portfolio.get('positions', {}))}")
    print(f"   總損益: ${portfolio.get('total_pnl', 0):+.0f}")
    
    # 持股明細
    positions = portfolio.get('positions', {})
    if positions:
        print(f"\n📊 持股明細:")
        for stock, pos in positions.items():
            quantity = pos.get('quantity', 0)
            avg_price = pos.get('avg_price', 0)
            current_value = quantity * avg_price
            print(f"   {stock}: {quantity}股 @${avg_price:.2f} (價值${current_value:.0f})")

def view_trading_log(date_str=None, lines=20):
    """查看交易系統日誌"""
    if not date_str:
        date_str = datetime.now().strftime('%Y%m%d')
    
    log_file = f"/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system/logs/trading_system_{date_str}.log"
    
    if not os.path.exists(log_file):
        print(f"❌ {date_str} 的交易日誌不存在")
        return
    
    print(f"📝 {date_str} 交易系統日誌 (最新{lines}行)")
    print("=" * 80)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # 顯示最新的lines行
    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
    
    for line in recent_lines:
        print(line.rstrip())

def list_available_dates():
    """列出可用的交易記錄日期"""
    log_dir = "/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system/logs"
    
    if not os.path.exists(log_dir):
        print("❌ 日誌目錄不存在")
        return
    
    dates = set()
    
    # 掃描交易會話文件
    for filename in os.listdir(log_dir):
        if filename.startswith('trading_sessions_') and filename.endswith('.json'):
            date_str = filename.replace('trading_sessions_', '').replace('.json', '')
            dates.add(date_str)
        elif filename.startswith('daily_report_') and filename.endswith('.json'):
            date_str = filename.replace('daily_report_', '').replace('.json', '')
            dates.add(date_str)
        elif filename.startswith('trading_system_') and filename.endswith('.log'):
            date_str = filename.replace('trading_system_', '').replace('.log', '')
            dates.add(date_str)
    
    if dates:
        print("📅 可用的交易記錄日期:")
        for date in sorted(dates):
            # 轉換格式顯示
            try:
                formatted_date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
                print(f"   {date} ({formatted_date})")
            except:
                print(f"   {date}")
    else:
        print("❌ 沒有找到交易記錄")

def main():
    parser = argparse.ArgumentParser(description='查看實時交易記錄')
    parser.add_argument('--date', '-d', help='指定查看日期 (格式: YYYYMMDD)')
    parser.add_argument('--sessions', '-s', action='store_true', help='查看交易會話')
    parser.add_argument('--report', '-r', action='store_true', help='查看每日報告')
    parser.add_argument('--log', '-l', action='store_true', help='查看系統日誌')
    parser.add_argument('--lines', type=int, default=20, help='日誌顯示行數 (默認20)')
    parser.add_argument('--list', action='store_true', help='列出可用日期')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_dates()
        return
    
    date_str = args.date
    
    if args.sessions:
        view_daily_sessions(date_str)
    elif args.report:
        view_daily_report(date_str)
    elif args.log:
        view_trading_log(date_str, args.lines)
    else:
        # 默認顯示所有信息
        print("🚀 實時交易記錄查看器")
        print("=" * 50)
        
        current_date = date_str or datetime.now().strftime('%Y%m%d')
        print(f"📅 查看日期: {current_date}")
        
        print("\n" + "=" * 50)
        view_daily_sessions(date_str)
        
        print("\n" + "=" * 50)
        view_daily_report(date_str)
        
        print("\n" + "=" * 50)
        view_trading_log(date_str, 10)

if __name__ == "__main__":
    main()