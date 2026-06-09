#!/bin/bash
# 夜間AI模型訓練 - 每週一到週五 23:00
cd /mnt/nvme/taiwan-stock-ai-analyzer-main
echo "$(date): 開始夜間AI訓練" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/night_training.log

# 確保GPU資源可用
sudo systemctl stop ollama
pkill -f ollama

# 🆕 先使用統一數據管理器更新數據
python3 -c "from unified_data_manager import UnifiedDataManager; dm = UnifiedDataManager(); dm.update_all_stocks()" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/night_training.log 2>&1

# 🆕 執行智能訓練工作流程 - 高優先級
python3 -c "from intelligent_training_scheduler import IntelligentTrainingScheduler; IntelligentTrainingScheduler().manual_training('high_priority')" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/night_training.log 2>&1

# 執行LSTM訓練 (RTX 3060 Ti)
export CUDA_VISIBLE_DEVICES=0
./rtx3060ti_isolated_lstm_training.sh >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/night_training.log 2>&1

# TG通知完成
python3 ai_trading_notification_bot.py --message "🌙 夜間訓練完成 $(date +%Y-%m-%d-%H:%M)"

echo "$(date): 夜間訓練完成" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/night_training.log
