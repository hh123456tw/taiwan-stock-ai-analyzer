#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
  📈 Taiwan Stock AI Analyzer (台股 AI 分析與智能量化系統)
  全功能整合命令列入口 (Unified CLI Interface)
==============================================================================
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

# 強制 Windows 終端輸出 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 確保專案根目錄納入 Python 模組搜索路徑
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REALTIME_DIR = PROJECT_ROOT / "realtime_trading_system"
if str(REALTIME_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_DIR))

BANNER = r"""
==============================================================================
       📈  Taiwan Stock AI Analyzer (台股 AI 分析與智能量化系統)  🚀
      Multi-Model ML + Dynamic Adaptive Weights + Realtime Engine
==============================================================================
"""

def cmd_pre_market(args):
    """執行盤前跳空與國際盤勢智能分析"""
    print("🌅 啟動【台股盤前智能分析】流水線...")
    try:
        from pre_market_analysis import main as run_pre
        asyncio.run(run_pre())
    except Exception as e:
        print(f"❌ 盤前分析執行失敗: {e}")
        import traceback
        traceback.print_exc()

def cmd_post_market(args):
    """執行盤後籌碼、三大法人與深度體檢"""
    print("📊 啟動【台股盤後深度體檢】流水線...")
    try:
        from enhanced_real_post_market_analysis import main as run_post
        asyncio.run(run_post())
    except Exception as e:
        print(f"❌ 盤後分析執行失敗: {e}")
        import traceback
        traceback.print_exc()

def cmd_realtime(args):
    """啟動盤中實時模擬交易系統"""
    print("⚡ 啟動【盤中實時模擬交易引擎】(09:00 - 13:30)...")
    try:
        from realtime_trading_system.realtime_trading_main import main as run_trading
        asyncio.run(run_trading())
    except Exception as e:
        print(f"❌ 實時交易系統執行失敗: {e}")
        import traceback
        traceback.print_exc()

def cmd_screener(args):
    """執行多維度智能選股器"""
    print(f"🔍 啟動【智能股票篩選器】 (策略: {args.strategy or '預設每日推薦'})...")
    try:
        from smart_stock_screener import SmartStockScreener
        screener = SmartStockScreener()
        if args.budget:
            recs = screener.get_budget_recommendation(float(args.budget))
        elif args.strategy:
            recs = screener.screen_by_strategy(args.strategy)
        else:
            recs = screener.daily_recommendation()
        
        print("\n================== 🎯 智能選股推薦結果 ==================")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. 【{rec['symbol']}】 {rec.get('name', '')}")
            print(f"   策略類別: {rec.get('strategy', '綜合')} | 評分: {rec.get('score', 0)} | 即時價格: {rec.get('price', 0):.2f}")
            print(f"   推薦亮點: {', '.join(rec.get('reasons', []))}")
            print("-" * 55)
    except Exception as e:
        print(f"❌ 智能選股執行失敗: {e}")
        import traceback
        traceback.print_exc()

def cmd_train(args):
    """執行機器學習模型訓練"""
    if args.symbol:
        print(f"🧪 啟動單檔股票模型訓練: 【{args.symbol}】...")
        try:
            from single_stock_trainer import SingleStockTrainer
            trainer = SingleStockTrainer(args.symbol)
            trainer.train_all_models()
        except Exception as e:
            print(f"❌ 單股訓練失敗: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🚀 啟動【全股票池批次模型訓練】(LSTM / XGBoost / LightGBM / RF / ARIMA)...")
        try:
            from batch_model_trainer import BatchModelTrainer
            trainer = BatchModelTrainer()
            trainer.run_batch_training()
        except Exception as e:
            print(f"❌ 批次訓練失敗: {e}")
            import traceback
            traceback.print_exc()

def cmd_schedule(args):
    """啟動定時自動化巡檢與排程"""
    print("⏰ 啟動【盤前/盤後定時自動化調度器】...")
    try:
        from auto_scheduler import main as run_sched
        run_sched()
    except Exception as e:
        print(f"❌ 排程器啟動失敗: {e}")
        import traceback
        traceback.print_exc()

def cmd_test(args):
    """執行全系統環境、依賴套件與健康度檢查"""
    print("🩺 執行【全系統環境與健康度檢查】...\n")
    checks = []

    # 1. Python 核心庫
    libs = ["numpy", "pandas", "yfinance", "sklearn", "xgboost", "lightgbm", "requests", "yaml"]
    for lib in libs:
        try:
            mod = __import__(lib)
            v = getattr(mod, "__version__", "OK")
            checks.append((f"Python 套件: {lib}", True, f"v{v}"))
        except ImportError:
            checks.append((f"Python 套件: {lib}", False, "未安裝 (請 pip install)"))

    # 2. 本地 Ollama 服務探測
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            checks.append(("本地 Ollama AI 大腦", True, f"在線 (模型: {', '.join(models[:3])})"))
        else:
            checks.append(("本地 Ollama AI 大腦", False, f"狀態碼: {resp.status_code}"))
    except Exception:
        checks.append(("本地 Ollama AI 大腦", False, "離線 (若無安裝 Ollama 仍可跑純數理 ML 模式)"))

    # 3. 專案目錄結構檢查
    for d in ["data", "models", "logs", "reports", "realtime_trading_system"]:
        p = PROJECT_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        checks.append((f"系統目錄: {d}/", True, "正常存在"))

    print(f"{'項目名稱':<28} | {'狀態':<6} | {'備註'}")
    print("-" * 65)
    for name, ok, note in checks:
        st = "✅ 通過" if ok else "⚠️ 警告"
        print(f"{name:<28} | {st:<6} | {note}")
    print("-" * 65)
    print("🎉 系統健康度檢查完畢！")

def build_parser():
    parser = argparse.ArgumentParser(
        description="📈 Taiwan Stock AI Analyzer (台股 AI 分析與智能量化系統)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用指令示範:
  python main.py pre-market       # 盤前 08:30 自動分析
  python main.py post-market      # 盤後 15:00 深度體檢與 Telegram 回報
  python main.py realtime         # 盤中 09:00-13:30 實時模擬交易
  python main.py screener         # 每日智能選股
  python main.py train --all      # 批次訓練所有股票之多模型矩陣
  python main.py test             # 快速檢查本地環境依賴與狀態
"""
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="子指令功能")

    # 1. 盤前分析
    sp_pre = subparsers.add_parser("pre-market", aliases=["pre"], help="盤前跳空與國際盤勢智能分析")
    sp_pre.set_defaults(func=cmd_pre_market)

    # 2. 盤後分析
    sp_post = subparsers.add_parser("post-market", aliases=["post"], help="盤後籌碼、三大法人與深度體檢")
    sp_post.set_defaults(func=cmd_post_market)

    # 3. 實時模擬交易
    sp_real = subparsers.add_parser("realtime", help="盤中實時模擬交易與技術指標掃描")
    sp_real.set_defaults(func=cmd_realtime)

    # 4. 智能選股
    sp_screen = subparsers.add_parser("screener", aliases=["scan"], help="多維度智能選股器")
    sp_screen.add_argument("--strategy", type=str, help="指定篩選策略 (high_price, mid_price, low_price, etf, affordable 等)")
    sp_screen.add_argument("--budget", type=float, help="根據新台幣預算智能配置標的")
    sp_screen.set_defaults(func=cmd_screener)

    # 5. 模型訓練
    sp_train = subparsers.add_parser("train", help="批次或單股 AI 預測模型訓練")
    sp_train.add_argument("--symbol", type=str, help="指定訓練單一股票 (如 2330.TW)")
    sp_train.add_argument("--all", action="store_true", help="批次訓練股票池中所有標的")
    sp_train.set_defaults(func=cmd_train)

    # 6. 排程器
    sp_sched = subparsers.add_parser("schedule", help="啟動定時自動化巡檢與排程")
    sp_sched.set_defaults(func=cmd_schedule)

    # 7. 環境自檢
    sp_test = subparsers.add_parser("test", help="環境、依賴與本地 Ollama 連線自檢")
    sp_test.set_defaults(func=cmd_test)

    return parser

def main():
    print(BANNER)
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
