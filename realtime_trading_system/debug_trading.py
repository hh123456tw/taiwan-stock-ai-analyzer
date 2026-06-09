#!/usr/bin/env python3
"""
調試實時交易系統
"""
import sys
import os
import asyncio
import logging
from datetime import datetime

sys.path.append('/mnt/nvme/taiwan-stock-ai-analyzer-main')
sys.path.append('/mnt/nvme/taiwan-stock-ai-analyzer-main/realtime_trading_system')

from core.market_scanner_fixed import RealTimeMarketScanner
from core.ai_decision_engine import AIDecisionEngine

async def debug_trading_system():
    """調試交易系統"""
    print("🔍 開始調試實時交易系統...")
    
    try:
        # 1. 初始化組件
        print("📊 初始化市場掃描器...")
        scanner = RealTimeMarketScanner()
        
        print("🧠 初始化AI決策引擎...")
        decision_engine = AIDecisionEngine()
        
        # 2. 測試市場掃描
        print("🔍 測試市場掃描...")
        candidates, market_overview = await scanner.scan_market()
        
        print(f"✅ 掃描結果: {len(candidates)} 個候選, 市場概況: {market_overview}")
        
        if candidates:
            print("📋 候選股票:")
            for i, candidate in enumerate(candidates[:3], 1):
                print(f"  {i}. {candidate}")
        
        # 3. 測試AI決策
        if candidates:
            print("🧠 測試AI決策...")
            test_portfolio = {
                'cash': 400000,
                'positions': {},
                'available_capital': 400000
            }
            
            decisions = decision_engine.make_trading_decisions(candidates, test_portfolio)
            print(f"✅ 決策結果: {len(decisions)} 個決策")
            
            if decisions:
                print("📋 交易決策:")
                for i, decision in enumerate(decisions[:3], 1):
                    print(f"  {i}. {decision}")
        
        print("✅ 調試完成")
        
    except Exception as e:
        print(f"❌ 調試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_trading_system())