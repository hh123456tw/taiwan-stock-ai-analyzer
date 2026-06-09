#!/bin/bash
# 盤後報告 - 每週一到週五 14:00
cd /mnt/nvme/taiwan-stock-ai-analyzer-main
echo "$(date): 開始盤後分析" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/post_market.log

# 停止盤中監控
pkill -f realtime_trading_monitor.py

# 執行盤後分析
python3 comprehensive_market_analyzer.py >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/post_market.log 2>&1

# TG推送盤後報告
python3 ai_trading_notification_bot.py --message "📈 盤後分析完成 - 交易日結束 $(date +%H:%M)"

echo "$(date): 盤後分析完成" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/post_market.log
