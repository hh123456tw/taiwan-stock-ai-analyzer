"""
Jiansoft 股票數據 API 整合
提供台股財務資料、營收數據、技術指標等
"""
import requests
import json
import time
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class JiansoftAPI:
    """Jiansoft API 客戶端"""
    
    def __init__(self):
        self.base_url = "https://jiansoft.freeddns.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_stock_revenues(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """
        獲取股票營收資料
        
        Args:
            symbol: 股票代號 (可選)
            limit: 返回筆數限制
            
        Returns:
            List[Dict]: 股票營收資料列表
        """
        try:
            url = f"{self.base_url}/stock/revenues"
            params = {
                'limit': limit
            }
            if symbol:
                params['symbol'] = symbol
                
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✅ 成功獲取 {len(data)} 筆營收資料")
            return data
            
        except Exception as e:
            logger.error(f"❌ 獲取營收資料失敗: {e}")
            return []
    
    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """
        獲取股票基本面資料
        
        Args:
            symbol: 股票代號
            
        Returns:
            Dict: 基本面資料 (EPS, 本益比, 殖利率等)
        """
        try:
            url = f"{self.base_url}/stock/fundamentals/{symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ 成功獲取 {symbol} 基本面資料")
                return data
            else:
                logger.warning(f"⚠️ {symbol} 基本面資料不可用")
                return {}
                
        except Exception as e:
            logger.error(f"❌ 獲取 {symbol} 基本面資料失敗: {e}")
            return {}
    
    def get_market_overview(self) -> Dict:
        """
        獲取市場總覽資料
        
        Returns:
            Dict: 市場總覽數據
        """
        try:
            url = f"{self.base_url}/market/overview"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ 成功獲取市場總覽")
                return data
            else:
                return self._get_fallback_market_data()
                
        except Exception as e:
            logger.error(f"❌ 獲取市場總覽失敗: {e}")
            return self._get_fallback_market_data()
    
    def get_industry_analysis(self, industry: str = None) -> List[Dict]:
        """
        獲取產業分析資料
        
        Args:
            industry: 產業類別 (可選)
            
        Returns:
            List[Dict]: 產業分析資料
        """
        try:
            url = f"{self.base_url}/industry/analysis"
            params = {}
            if industry:
                params['industry'] = industry
                
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ 成功獲取產業分析資料")
                return data
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ 獲取產業分析失敗: {e}")
            return []
    
    def search_stocks(self, keyword: str, limit: int = 20) -> List[Dict]:
        """
        搜尋股票
        
        Args:
            keyword: 搜尋關鍵字
            limit: 返回筆數限制
            
        Returns:
            List[Dict]: 搜尋結果
        """
        try:
            url = f"{self.base_url}/stock/search"
            params = {
                'q': keyword,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ 搜尋到 {len(data)} 筆股票資料")
                return data
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ 搜尋股票失敗: {e}")
            return []
    
    def _get_fallback_market_data(self) -> Dict:
        """備用市場資料"""
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'fallback',
            'indices': {
                'taiex': {'value': 0, 'change': 0, 'change_percent': 0}
            },
            'market_sentiment': 'neutral',
            'trading_volume': 0
        }
    
    def test_connection(self) -> bool:
        """測試 API 連線"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Jiansoft API 連線正常")
                return True
            else:
                logger.warning("⚠️ Jiansoft API 連線異常")
                return False
                
        except Exception as e:
            logger.error(f"❌ Jiansoft API 連線失敗: {e}")
            return False

# 使用範例
if __name__ == "__main__":
    # 基本測試
    api = JiansoftAPI()
    
    # 測試連線
    if api.test_connection():
        # 獲取台積電資料
        revenues = api.get_stock_revenues("2330", limit=10)
        print(f"台積電營收資料: {len(revenues)} 筆")
        
        # 獲取市場總覽
        market = api.get_market_overview()
        print(f"市場狀態: {market.get('market_sentiment', 'unknown')}")