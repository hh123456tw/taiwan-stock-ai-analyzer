#!/usr/bin/env python3
"""
台股交易日判斷工具
"""

import datetime
from typing import Dict, List

class TradingDayChecker:
    """台股交易日檢查器"""
    
    # 台股國定假日 (需要定期更新)
    TAIWAN_HOLIDAYS_2025 = [
        # 新年假期
        '2025-01-01',  # 元旦
        
        # 農曆春節
        '2025-01-25', '2025-01-26', '2025-01-27', '2025-01-28', '2025-01-29',
        '2025-01-30', '2025-01-31',  # 春節假期
        
        # 其他假日
        '2025-02-28',  # 和平紀念日
        '2025-04-04',  # 兒童節
        '2025-04-05',  # 清明節
        '2025-05-01',  # 勞動節
        '2025-06-23',  # 端午節
        '2025-10-10',  # 國慶日
        '2025-12-25',  # 聖誕節
    ]
    
    @classmethod
    def is_trading_day(cls, check_date: datetime.date = None) -> bool:
        """
        檢查指定日期是否為交易日
        
        Args:
            check_date: 要檢查的日期，預設為今天
            
        Returns:
            bool: True為交易日，False為非交易日
        """
        if check_date is None:
            check_date = datetime.date.today()
        
        # 檢查是否為週末 (週六=5, 週日=6)
        if check_date.weekday() >= 5:
            return False
        
        # 檢查是否為國定假日
        date_str = check_date.strftime('%Y-%m-%d')
        if date_str in cls.TAIWAN_HOLIDAYS_2025:
            return False
        
        return True
    
    @classmethod
    def is_trading_time(cls, check_time: datetime.datetime = None) -> bool:
        """
        檢查指定時間是否在交易時間內
        台股交易時間: 9:00-13:30 (週一到週五)
        
        Args:
            check_time: 要檢查的時間，預設為現在
            
        Returns:
            bool: True為交易時間，False為非交易時間
        """
        if check_time is None:
            check_time = datetime.datetime.now()
        
        # 首先檢查是否為交易日
        if not cls.is_trading_day(check_time.date()):
            return False
        
        # 檢查是否在交易時間內 (9:00-13:30)
        trading_start = datetime.time(9, 0)
        trading_end = datetime.time(13, 30)
        current_time = check_time.time()
        
        return trading_start <= current_time <= trading_end
    
    @classmethod
    def get_next_trading_day(cls, from_date: datetime.date = None) -> datetime.date:
        """
        取得下一個交易日
        
        Args:
            from_date: 起始日期，預設為今天
            
        Returns:
            date: 下一個交易日
        """
        if from_date is None:
            from_date = datetime.date.today()
        
        check_date = from_date + datetime.timedelta(days=1)
        
        while not cls.is_trading_day(check_date):
            check_date += datetime.timedelta(days=1)
        
        return check_date
    
    @classmethod
    def get_previous_trading_day(cls, from_date: datetime.date = None) -> datetime.date:
        """
        取得前一個交易日
        
        Args:
            from_date: 起始日期，預設為今天
            
        Returns:
            date: 前一個交易日
        """
        if from_date is None:
            from_date = datetime.date.today()
        
        check_date = from_date - datetime.timedelta(days=1)
        
        while not cls.is_trading_day(check_date):
            check_date -= datetime.timedelta(days=1)
        
        return check_date
    
    @classmethod
    def should_run_trading_task(cls) -> Dict[str, bool]:
        """
        判斷各種交易任務是否應該執行
        
        Returns:
            dict: 各種任務的執行建議
        """
        now = datetime.datetime.now()
        current_date = now.date()
        current_time = now.time()
        
        is_trading_day = cls.is_trading_day(current_date)
        is_trading_time = cls.is_trading_time(now)
        
        return {
            'pre_market_analysis': (
                is_trading_day and 
                datetime.time(7, 0) <= current_time <= datetime.time(8, 59)
            ),
            'intraday_analysis': is_trading_time,
            'post_market_analysis': (
                is_trading_day and 
                current_time >= datetime.time(14, 0)
            ),
            'prediction_generation': is_trading_day,
            'prediction_validation': is_trading_day,
            'model_training': True,  # 訓練可以任何時間進行
            'system_maintenance': True,  # 系統維護可以任何時間進行
        }

def main():
    """測試用主程式"""
    checker = TradingDayChecker()
    now = datetime.datetime.now()
    
    print("=== 台股交易日檢查工具 ===")
    print(f"當前時間: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"星期: {['週一', '週二', '週三', '週四', '週五', '週六', '週日'][now.weekday()]}")
    print()
    
    print("交易狀態檢查:")
    print(f"是否為交易日: {'是' if checker.is_trading_day() else '否'}")
    print(f"是否在交易時間: {'是' if checker.is_trading_time() else '否'}")
    print()
    
    print("日期資訊:")
    print(f"下一個交易日: {checker.get_next_trading_day()}")
    print(f"前一個交易日: {checker.get_previous_trading_day()}")
    print()
    
    print("任務執行建議:")
    recommendations = checker.should_run_trading_task()
    for task, should_run in recommendations.items():
        status = "✅ 建議執行" if should_run else "❌ 不建議執行"
        print(f"  {task}: {status}")

if __name__ == "__main__":
    main()