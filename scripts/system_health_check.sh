#!/bin/bash
# 系統健康檢查
cd /mnt/nvme/taiwan-stock-ai-analyzer-main

# 檢查GPU狀態
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/gpu_status.log

# 檢查磁碟空間
df -h /mnt/nvme/taiwan-stock-ai-analyzer-main >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/disk_usage.log

# 檢查重要進程
ps aux | grep -E "(python3|training|monitor)" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/process_status.log

# 記錄時間戳
echo "$(date): 系統檢查完成" >> /mnt/nvme/taiwan-stock-ai-analyzer-main/logs/health_check.log
