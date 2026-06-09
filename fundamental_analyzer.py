"""
基本面分析器 - 台股AI系統基本面分析核心模組
整合Jiansoft API，提供股票基本面評分和分析

作者: AI助手 & 用戶協作開發
日期: 2025-07-24
版本: 1.0
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from data_sources.jiansoft_api import JiansoftAPI

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """
    基本面分析器
    
    整合Jiansoft API，提供完整的股票基本面分析功能：
    - 財務指標評分 (本益比、ROE、殖利率等)
    - 基本面綜合評分
    - 投資價值評估
    - 風險等級分析
    """
    
    def __init__(self):
        """初始化基本面分析器"""
        self.jiansoft_api = JiansoftAPI()
        self.logger = logger
        self.cache = {}  # 簡單快取機制
        self.cache_timeout = 3600  # 1小時快取
        
        # 測試API連接
        if self.jiansoft_api.test_connection():
            self.logger.info("✅ Jiansoft API 連接成功")
        else:
            self.logger.warning("⚠️ Jiansoft API 連接失敗，將使用備用評分機制")
    
    def get_fundamental_score(self, symbol: str) -> Dict:
        """
        計算股票基本面綜合評分
        
        Args:
            symbol: 股票代號 (如 "2330")
            
        Returns:
            Dict: 包含各項評分和原始數據的字典
        """
        try:
            # 檢查快取
            cache_key = f"fundamental_{symbol}"
            if self._is_cache_valid(cache_key):
                self.logger.info(f"📊 使用快取數據: {symbol}")
                return self.cache[cache_key]['data']
            
            # 獲取基本面數據
            self.logger.info(f"📊 開始分析 {symbol} 基本面...")
            fundamentals = self.jiansoft_api.get_stock_fundamentals(symbol)
            
            if not fundamentals:
                self.logger.warning(f"⚠️ {symbol} 基本面數據不可用，使用估計值")
                return self._get_fallback_score(symbol)
            
            # 計算各項指標評分
            pe_score = self.evaluate_pe_ratio(fundamentals.get('pe_ratio'))
            roe_score = self.evaluate_roe(fundamentals.get('roe'))
            dividend_score = self.evaluate_dividend_yield(fundamentals.get('dividend_yield'))
            revenue_growth_score = self.evaluate_revenue_growth(fundamentals.get('revenue_growth'))
            debt_ratio_score = self.evaluate_debt_ratio(fundamentals.get('debt_ratio'))
            
            # 計算綜合評分 (加權平均)
            weights = {
                'pe': 0.25,      # 本益比 25%
                'roe': 0.25,     # ROE 25%
                'dividend': 0.20, # 殖利率 20%
                'revenue': 0.20,  # 營收成長 20%
                'debt': 0.10      # 負債比 10%
            }
            
            total_score = (
                pe_score * weights['pe'] +
                roe_score * weights['roe'] +
                dividend_score * weights['dividend'] +
                revenue_growth_score * weights['revenue'] +
                debt_ratio_score * weights['debt']
            )
            
            # 獲取投資等級
            investment_grade = self._get_investment_grade(total_score)
            risk_level = self._get_risk_level(fundamentals)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'total_score': round(total_score, 2),
                'investment_grade': investment_grade,
                'risk_level': risk_level,
                'scores': {
                    'pe_score': round(pe_score, 2),
                    'roe_score': round(roe_score, 2),
                    'dividend_score': round(dividend_score, 2),
                    'revenue_growth_score': round(revenue_growth_score, 2),
                    'debt_ratio_score': round(debt_ratio_score, 2)
                },
                'raw_data': fundamentals,
                'analysis_summary': self._generate_summary(total_score, fundamentals)
            }
            
            # 存入快取
            self._cache_result(cache_key, result)
            
            self.logger.info(f"✅ {symbol} 基本面分析完成，總分: {total_score:.1f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 基本面分析失敗: {e}")
            return self._get_fallback_score(symbol)
    
    def evaluate_pe_ratio(self, pe: Optional[float]) -> float:
        """
        評估本益比 (PE Ratio)
        越低越好，但需考慮合理性
        
        Args:
            pe: 本益比數值
            
        Returns:
            float: 評分 (0-100)
        """
        if pe is None or pe <= 0:
            return 20  # 虧損股或數據不可用
        
        if pe <= 8:
            return 95   # 極度低估
        elif pe <= 12:
            return 85   # 低估
        elif pe <= 15:
            return 75   # 合理偏低
        elif pe <= 20:
            return 60   # 合理
        elif pe <= 25:
            return 45   # 略高
        elif pe <= 35:
            return 30   # 偏高
        else:
            return 15   # 高估
    
    def evaluate_roe(self, roe: Optional[float]) -> float:
        """
        評估股東權益報酬率 (ROE)
        越高越好，顯示公司獲利能力
        
        Args:
            roe: ROE百分比
            
        Returns:
            float: 評分 (0-100)
        """
        if roe is None:
            return 50  # 數據不可用，給予中性評分
        
        if roe >= 25:
            return 100  # 優秀
        elif roe >= 20:
            return 90   # 很好
        elif roe >= 15:
            return 80   # 良好
        elif roe >= 12:
            return 70   # 尚可
        elif roe >= 8:
            return 55   # 普通
        elif roe >= 5:
            return 40   # 偏低
        elif roe > 0:
            return 25   # 很低
        else:
            return 10   # 虧損
    
    def evaluate_dividend_yield(self, dividend: Optional[float]) -> float:
        """
        評估殖利率 (Dividend Yield)
        適中最好，太高可能有問題，太低缺乏吸引力
        
        Args:
            dividend: 殖利率百分比
            
        Returns:
            float: 評分 (0-100)
        """
        if dividend is None:
            return 50  # 數據不可用
        
        if 3 <= dividend <= 6:
            return 100   # 理想區間
        elif 2 <= dividend < 3:
            return 80    # 略低但可接受
        elif 6 < dividend <= 8:
            return 85    # 略高但還好
        elif 8 < dividend <= 10:
            return 65    # 偏高，需注意
        elif 1 <= dividend < 2:
            return 60    # 較低
        elif dividend > 10:
            return 30    # 異常高，可能有問題
        else:
            return 20    # 無股息或數據異常
    
    def evaluate_revenue_growth(self, growth: Optional[float]) -> float:
        """
        評估營收成長率
        正成長為佳，但需合理
        
        Args:
            growth: 營收成長率百分比
            
        Returns:
            float: 評分 (0-100)
        """
        if growth is None:
            return 50  # 數據不可用
        
        if growth >= 20:
            return 95   # 高成長
        elif growth >= 15:
            return 85   # 良好成長
        elif growth >= 10:
            return 75   # 穩定成長
        elif growth >= 5:
            return 65   # 微幅成長
        elif growth >= 0:
            return 50   # 持平
        elif growth >= -5:
            return 35   # 微幅衰退
        elif growth >= -10:
            return 25   # 明顯衰退
        else:
            return 15   # 嚴重衰退
    
    def evaluate_debt_ratio(self, debt_ratio: Optional[float]) -> float:
        """
        評估負債比率
        越低越好，顯示財務健康
        
        Args:
            debt_ratio: 負債比率百分比
            
        Returns:
            float: 評分 (0-100)
        """
        if debt_ratio is None:
            return 60  # 數據不可用，給予略高於中性的評分
        
        if debt_ratio <= 20:
            return 100  # 優秀
        elif debt_ratio <= 30:
            return 85   # 良好
        elif debt_ratio <= 40:
            return 70   # 可接受
        elif debt_ratio <= 50:
            return 55   # 普通
        elif debt_ratio <= 60:
            return 40   # 偏高
        elif debt_ratio <= 70:
            return 25   # 風險較高
        else:
            return 15   # 高風險
    
    def _get_investment_grade(self, score: float) -> str:
        """根據總分獲取投資等級"""
        if score >= 85:
            return "A+ 優秀"
        elif score >= 75:
            return "A 良好"
        elif score >= 65:
            return "B+ 尚可"
        elif score >= 55:
            return "B 普通"
        elif score >= 45:
            return "C+ 偏弱"
        elif score >= 35:
            return "C 較差"
        else:
            return "D 風險"
    
    def _get_risk_level(self, fundamentals: Dict) -> str:
        """評估風險等級"""
        risk_factors = 0
        
        # 檢查各種風險因子
        pe = fundamentals.get('pe_ratio', 0)
        if pe > 30 or pe <= 0:
            risk_factors += 1
            
        roe = fundamentals.get('roe', 0)
        if roe < 5:
            risk_factors += 1
            
        debt_ratio = fundamentals.get('debt_ratio', 0)
        if debt_ratio > 60:
            risk_factors += 1
            
        revenue_growth = fundamentals.get('revenue_growth', 0)
        if revenue_growth < -10:
            risk_factors += 1
        
        if risk_factors == 0:
            return "低風險"
        elif risk_factors <= 1:
            return "中低風險"
        elif risk_factors <= 2:
            return "中風險"
        elif risk_factors <= 3:
            return "中高風險"
        else:
            return "高風險"
    
    def _generate_summary(self, score: float, fundamentals: Dict) -> str:
        """生成分析摘要"""
        grade = self._get_investment_grade(score)
        
        strengths = []
        weaknesses = []
        
        # 分析優勢和劣勢
        pe = fundamentals.get('pe_ratio', 0)
        if pe > 0 and pe <= 15:
            strengths.append("估值合理")
        elif pe > 25:
            weaknesses.append("估值偏高")
            
        roe = fundamentals.get('roe', 0)
        if roe >= 15:
            strengths.append("獲利能力強")
        elif roe < 8:
            weaknesses.append("獲利能力偏弱")
            
        dividend = fundamentals.get('dividend_yield', 0)
        if 3 <= dividend <= 6:
            strengths.append("股息收益穩定")
        elif dividend > 8:
            weaknesses.append("股息異常高")
            
        summary = f"投資等級: {grade}。"
        
        if strengths:
            summary += f" 優勢: {', '.join(strengths)}。"
        if weaknesses:
            summary += f" 需注意: {', '.join(weaknesses)}。"
            
        return summary
    
    def _get_fallback_score(self, symbol: str) -> Dict:
        """API不可用時的備用評分"""
        # 基於股票代號的簡單評估邏輯
        base_score = 55  # 基礎分數
        
        # ETF通常較穩定
        if symbol.startswith('00'):
            base_score = 70
        # 大盤股相對穩定
        elif symbol in ['2330', '2317', '2382', '2454']:
            base_score = 65
            
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'total_score': base_score,
            'investment_grade': self._get_investment_grade(base_score),
            'risk_level': "中風險",
            'scores': {
                'pe_score': 60,
                'roe_score': 60,
                'dividend_score': 60,
                'revenue_growth_score': 50,
                'debt_ratio_score': 60
            },
            'raw_data': {},
            'analysis_summary': f"使用備用評分機制，建議等待數據恢復後重新分析。",
            'is_fallback': True
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """檢查快取是否有效"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_timeout
    
    def _cache_result(self, cache_key: str, result: Dict):
        """存入快取"""
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
    
    def batch_analyze(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        批量分析多檔股票
        
        Args:
            symbols: 股票代號列表
            
        Returns:
            Dict: 各股票的分析結果
        """
        results = {}
        
        self.logger.info(f"📊 開始批量分析 {len(symbols)} 檔股票...")
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_fundamental_score(symbol)
                time.sleep(0.5)  # 避免API請求過於頻繁
            except Exception as e:
                self.logger.error(f"❌ {symbol} 分析失敗: {e}")
                results[symbol] = self._get_fallback_score(symbol)
        
        self.logger.info(f"✅ 批量分析完成，成功分析 {len(results)} 檔股票")
        return results
    
    def compare_stocks(self, symbols: List[str]) -> List[Dict]:
        """
        比較多檔股票的基本面
        
        Args:
            symbols: 要比較的股票代號列表
            
        Returns:
            List[Dict]: 按總分排序的股票列表
        """
        results = self.batch_analyze(symbols)
        
        # 轉換為列表並排序
        comparison = []
        for symbol, analysis in results.items():
            comparison.append({
                'symbol': symbol,
                'total_score': analysis['total_score'],
                'investment_grade': analysis['investment_grade'],
                'risk_level': analysis['risk_level'],
                'analysis_summary': analysis['analysis_summary']
            })
        
        # 按總分排序
        comparison.sort(key=lambda x: x['total_score'], reverse=True)
        
        self.logger.info(f"📊 股票比較完成，最佳標的: {comparison[0]['symbol']} ({comparison[0]['total_score']:.1f}分)")
        
        return comparison

# 使用範例和測試
if __name__ == "__main__":
    # 基本測試
    analyzer = FundamentalAnalyzer()
    
    # 測試單檔股票分析
    print("=== 測試台積電 (2330) 基本面分析 ===")
    result = analyzer.get_fundamental_score("2330")
    print(f"總分: {result['total_score']}")
    print(f"投資等級: {result['investment_grade']}")
    print(f"風險等級: {result['risk_level']}")
    print(f"分析摘要: {result['analysis_summary']}")
    
    print("\n=== 測試多檔股票比較 ===")
    comparison = analyzer.compare_stocks(["2330", "0050", "00878"])
    for i, stock in enumerate(comparison, 1):
        print(f"{i}. {stock['symbol']}: {stock['total_score']:.1f}分 ({stock['investment_grade']})")