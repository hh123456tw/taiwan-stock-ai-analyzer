# 修復台股交易時間配置
import re

with open('config/trading_config.py', 'r') as f:
    content = f.read()

# 台股是連續交易9:00-13:30，沒有午休
new_schedule = '''    # 交易時間配置
    SCHEDULE_CONFIG = {
        'market_open': time(9, 0),         # 市場開盤
        'trading_start': time(9, 0),       # 開始交易 (台股9:00開始)
        'lunch_break_start': time(14, 0),  # 無午休，設為收盤後
        'lunch_break_end': time(14, 0),    # 無午休，設為收盤後  
        'trading_end': time(13, 30),       # 結束交易 (台股13:30收盤)
        'market_close': time(13, 30),      # 市場收盤
        'scan_interval': 300,              # 掃描間隔 (5分鐘)
        'trading_days': [0, 1, 2, 3, 4],   # 週一至週五 (0=週一)
    }'''

# 替換 SCHEDULE_CONFIG 區塊
pattern = r'    # 交易時間配置\s*\n    SCHEDULE_CONFIG = \{[^}]+\}'
content = re.sub(pattern, new_schedule, content, flags=re.MULTILINE | re.DOTALL)

with open('config/trading_config.py', 'w') as f:
    f.write(content)

print(✅ 台股交易時間配置已修復)
