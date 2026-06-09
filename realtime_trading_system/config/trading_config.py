#\!/usr/bin/env python3
"""
實時AI自動選股交易系統 - 配置管理
"""

import os
from datetime import datetime, time
from typing import Dict, List, Any

class TradingConfig:
    """交易系統配置管理"""
    
    # 資金管理配置
    CAPITAL_CONFIG = {
        'total_capital': 2000000,          # 總模擬資金 (200萬)
        'max_single_investment': 200000,   # 單筆投資上限 (20萬)
        'min_single_investment': 20000,    # 單筆投資下限 (2萬)
        'max_daily_loss': 100000,         # 日損失上限 (10萬)
        'max_positions': 15,              # 最大持股檔數
        'max_position_ratio': 0.15,       # 單檔最大比例 (15%)
        'cash_reserve_ratio': 0.2,        # 現金保留比例 (20%)
    }
    
    # AI決策配置
    AI_CONFIG = {
        'min_confidence': 0.6,            # 最低信心度閾值
        'price_change_threshold': 0.02,   # 價格變動閾值 (2%)
        'max_prediction_age': 3600,       # 預測最大有效時間 (1小時)
        'confidence_weight': 0.25,        # 信心度權重係數
        'base_investment_ratio': 0.05,    # 基礎投資比例 (5%)
    }
    
    # 交易時間配置
    SCHEDULE_CONFIG = {
        'market_open': time(9, 0),         # 市場開盤
        'trading_start': time(9, 0),      # 開始交易
        'lunch_break_start': time(14, 0),  # 午休開始
        'lunch_break_end': time(13, 30),    # 午休結束
        'trading_end': time(13, 30),        # 結束交易
        'market_close': time(13, 30),      # 市場收盤
        'scan_interval': 300,              # 掃描間隔 (5分鐘)
        'trading_days': [0, 1, 2, 3, 4],   # 週一至週五 (0=週一)
    }
    
    # 風險控制配置
    RISK_CONFIG = {
        'stop_loss_ratio': 0.05,          # 停損比例 (5%)
        'max_consecutive_losses': 3,       # 最大連續虧損次數
        'emergency_stop_threshold': 0.1,   # 緊急停止閾值 (10%虧損)
        'max_daily_trades': 20,           # 日最大交易次數
        'cooling_period': 3600,           # 冷卻期 (1小時)
    }
    
    # 系統配置
    SYSTEM_CONFIG = {
        'log_level': 'INFO',
        'log_rotation': 'daily',
        'max_log_files': 30,
        'database_backup_interval': 24,    # 小時
        'health_check_interval': 60,       # 秒
        'performance_report_interval': 24, # 小時
    }
    
    # 數據源配置
    DATA_CONFIG = {
        'predictions_db': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'predictions.db'),
        'trading_db': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'trading.db'),
        'log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs'),
        'report_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'reports'),
        'backup_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'backups'),
    }
    
    @classmethod
    def get_investment_amount(cls, confidence: float, available_capital: float) -> float:
        """根據信心度計算投資金額"""
        if confidence < cls.AI_CONFIG['min_confidence']:
            return 0
            
        # 基礎投資比例 + 信心度加成
        base_ratio = cls.AI_CONFIG['base_investment_ratio']
        confidence_bonus = (confidence - cls.AI_CONFIG['min_confidence']) * cls.AI_CONFIG['confidence_weight']
        investment_ratio = min(base_ratio + confidence_bonus, 0.15)  # 最大15%
        
        investment_amount = available_capital * investment_ratio
        
        # 確保在單筆投資限制範圍內
        investment_amount = max(investment_amount, cls.CAPITAL_CONFIG['min_single_investment'])
        investment_amount = min(investment_amount, cls.CAPITAL_CONFIG['max_single_investment'])
        
        return investment_amount
    
    @classmethod
    def is_trading_time(cls, check_time: datetime = None) -> bool:
        """檢查是否為交易時間"""
        if check_time is None:
            check_time = datetime.now()
            
        # 檢查是否為交易日
        if check_time.weekday() not in cls.SCHEDULE_CONFIG['trading_days']:
            return False
            
        current_time = check_time.time()
        
        # 檢查是否在交易時間內
        if (current_time >= cls.SCHEDULE_CONFIG['trading_start'] and 
            current_time < cls.SCHEDULE_CONFIG['lunch_break_start']):
            return True
            
        if (current_time >= cls.SCHEDULE_CONFIG['lunch_break_end'] and 
            current_time <= cls.SCHEDULE_CONFIG['trading_end']):
            return True
            
        return False
    
    @classmethod
    def validate_config(cls) -> bool:
        """驗證配置的合理性"""
        try:
            # 檢查資金配置
            assert cls.CAPITAL_CONFIG['max_single_investment'] <= cls.CAPITAL_CONFIG['total_capital']
            assert cls.CAPITAL_CONFIG['min_single_investment'] <= cls.CAPITAL_CONFIG['max_single_investment']
            assert cls.CAPITAL_CONFIG['max_daily_loss'] <= cls.CAPITAL_CONFIG['total_capital']
            
            # 檢查AI配置
            assert 0 < cls.AI_CONFIG['min_confidence'] <= 1
            assert cls.AI_CONFIG['price_change_threshold'] > 0
            
            # 檢查風險配置
            assert 0 < cls.RISK_CONFIG['stop_loss_ratio'] < 1
            assert cls.RISK_CONFIG['max_consecutive_losses'] > 0
            
            return True
        except AssertionError:
            return False

# 測試配置
if __name__ == '__main__':
    print('🔧 交易系統配置測試')
    
    # 驗證配置
    if TradingConfig.validate_config():
        print('✅ 配置驗證通過')
    else:
        print('❌ 配置驗證失敗')
        
    # 測試投資金額計算
    test_cases = [
        (0.6, 100000),  # 最低信心度
        (0.8, 100000),  # 高信心度
        (1.0, 100000),  # 最高信心度
        (0.5, 100000),  # 低於閾值
    ]
    
    print('\n💰 投資金額計算測試:')
    for confidence, capital in test_cases:
        amount = TradingConfig.get_investment_amount(confidence, capital)
        print(f'信心度 {confidence:.1f}, 可用資金 {capital:,} → 投資金額 {amount:,.0f}')
        
    # 測試交易時間
    print('\n⏰ 交易時間測試:')
    test_times = [
        datetime(2025, 7, 30, 9, 0),   # 開盤前
        datetime(2025, 7, 30, 10, 0),  # 交易時間
        datetime(2025, 7, 30, 12, 30), # 午休時間
        datetime(2025, 7, 30, 14, 0),  # 收盤後
        datetime(2025, 8, 2, 10, 0),   # 週六
    ]
    
    for test_time in test_times:
        is_trading = TradingConfig.is_trading_time(test_time)
        print(f'{test_time.strftime("%Y-%m-%d %H:%M")} ({["週一","週二","週三","週四","週五","週六","週日"][test_time.weekday()]}) → {"交易時間" if is_trading else "非交易時間"}')
