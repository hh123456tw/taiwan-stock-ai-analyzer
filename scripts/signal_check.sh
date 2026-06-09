#!/bin/bash
# 交易訊號檢查 - 交易時間內每10分鐘
cd /mnt/nvme/taiwan-stock-ai-analyzer-main

# 檢查監控程序是否運行
if ! pgrep -f "realtime_trading_monitor.py" > /dev/null; then
    echo "$(date): 監控程序未運行，重新啟動" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/signal_check.log
    nohup python3 realtime_trading_monitor.py >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/trading_monitor.log 2>&1 &
    python3 ai_trading_notification_bot.py --message "⚠️ 監控程序重新啟動 $(date +%H:%M)"
fi

# 🆕 檢查交易訊號 (使用統一數據管理)
python3 check_trading_signals.py >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/signal_check.log 2>&1
