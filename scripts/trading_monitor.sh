#!/bin/bash
# 盤中監控 - 每週一到週五 09:00-13:30
cd /mnt/nvme/taiwan-stock-ai-analyzer-main
echo "$(date): 開始盤中監控" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/trading_monitor.log

# 啟動實時監控 (背景執行)
nohup python3 realtime_trading_monitor.py >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/trading_monitor.log 2>&1 &
MONITOR_PID=$!

# TG通知開始監控
python3 ai_trading_notification_bot.py --message "🚀 盤中監控啟動 PID:$MONITOR_PID $(date +%H:%M)"

echo "$(date): 盤中監控啟動 PID:$MONITOR_PID" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/trading_monitor.log
