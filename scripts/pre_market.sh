#!/bin/bash
# 盤前分析 - 每週一到週五 08:00
cd /mnt/nvme/taiwan-stock-ai-analyzer-main
echo "$(date): 開始盤前分析" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/pre_market.log

# 🆕 先更新最新股價數據
python3 -c "from unified_data_manager import UnifiedDataManager; dm = UnifiedDataManager(); dm.update_all_stocks()" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/pre_market.log 2>&1

# 🚀 執行116檔股票完整盤前分析
python3 comprehensive_market_analyzer.py >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/pre_market.log 2>&1

# TG推送盤前分析結果
python3 ai_trading_notification_bot.py --message "📊 盤前分析完成 - 準備開盤 $(date +%H:%M)"

echo "$(date): 盤前分析完成" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/pre_market.log
