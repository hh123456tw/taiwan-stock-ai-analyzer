#!/bin/bash
# 週末深度訓練 - 每週六 22:00
cd /mnt/nvme/taiwan-stock-ai-analyzer-main
echo "$(date): 開始週末深度訓練" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/weekend_training.log

# 確保GPU資源完全可用
sudo systemctl stop ollama
pkill -f ollama
sleep 10

# 🆕 先使用統一數據管理器更新所有數據  
python3 -c "from unified_data_manager import UnifiedDataManager; dm = UnifiedDataManager(); dm.update_all_stocks()" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/weekend_training.log 2>&1

# 🆕 執行完整智能訓練工作流程 - 所有優先級
python3 -c "from intelligent_training_scheduler import IntelligentTrainingScheduler; s = IntelligentTrainingScheduler(); s.manual_training('high_priority'); s.manual_training('medium_priority')" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/weekend_training.log 2>&1

# 完整訓練週期 (保留原有的)
python3 complete_training_cycle.py >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/weekend_training.log 2>&1

# TG通知完成
python3 ai_trading_notification_bot.py --message "🔧 週末深度訓練完成 $(date +%Y-%m-%d-%H:%M)"

echo "$(date): 週末深度訓練完成" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/weekend_training.log
