"""
簡化的錯誤處理 - 只保留必要功能
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """輸入驗證錯誤"""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class DataFetchError(Exception):
    """資料獲取錯誤"""
    def __init__(self, symbol: str, message: str = None):
        self.symbol = symbol
        self.message = message or f"無法獲取 {symbol} 的股票資料"
        super().__init__(self.message)

def validate_stock_symbol(symbol: str) -> str:
    """驗證股票代碼"""
    if not symbol:
        raise ValidationError("股票代碼不能為空")
    
    symbol = str(symbol).strip()
    
    if not symbol.isdigit() or len(symbol) < 4 or len(symbol) > 6:
        raise ValidationError(f"股票代碼格式錯誤: {symbol}，請輸入4-6位數字")
    
    return symbol

def handle_errors(func):
    """簡化的錯誤處理裝飾器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"執行 {func.__name__} 發生錯誤: {e}")
            return None
    return wrapper
