"""
增強型 LLM 分析器 - 智能股票分析系統
提供深度市場洞察和專業投資建議
"""
import requests
import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class OptimizedLLMAnalyzer:
    """優化的 LLM 分析器 - 減少 TOKEN 消耗，提升分析品質"""

    def __init__(self, config: Dict[str, Any] = None):
        """初始化分析器"""
        self.config = config or {}

        # 從配置文件或環境變數讀取設定
        self.base_url = self.config.get('ollama_url', 'http://localhost:11435/api/generate')
        self.model = self.config.get('model', 'llama3.1:8b')
        self.timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))

        # TOKEN 控制設定
        self.max_prompt_length = int(os.getenv('MAX_PROMPT_LENGTH', '1000'))
        self.max_response_length = int(os.getenv('MAX_RESPONSE_LENGTH', '500'))

        # 分析深度設定
        self.analysis_depth = os.getenv('ANALYSIS_DEPTH', 'basic')
        self.enable_detailed = os.getenv('ENABLE_DETAILED_ANALYSIS', 'false').lower() == 'true'

        self._test_connection()

    def _test_connection(self):
        """測試 Ollama 連線"""
        try:
            response = requests.get('http://localhost:11435/api/tags', timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Ollama 連線成功，使用模型: {self.model}")
            else:
                logger.warning(f"⚠️ Ollama 連線異常: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Ollama 連線失敗: {e}")

    def analyze_stock_quick(self, stock_data: Dict[str, Any]) -> str:
        """快速股票分析 - 極簡版本，強制只輸出操作建議"""
        
        symbol = stock_data.get('symbol', '')
        price = stock_data.get('price', 0)
        change = stock_data.get('change_percent', 0)
        rsi = stock_data.get('rsi', 50)
        
        # 先用規則判斷，避免 LLM 廢話
        if rsi > 80:
            return f"🟡 RSI超買({rsi:.0f})，注意回調風險"
        elif rsi < 20:
            return f"🟢 RSI超賣({rsi:.0f})，可關注反彈"
        elif abs(change) > 5:
            direction = "大漲" if change > 0 else "大跌"
            advice = "注意獲利了結" if change > 0 else "可逢低關注"
            return f"{direction}{abs(change):.1f}%，{advice}"
        elif abs(change) < 1:
            return f"盤整中，等待方向明確"
        
        # 極簡 prompt - 只要操作建議
        prompt = f"{symbol} {price:.0f}元 {change:+.1f}% RSI{rsi:.0f} 操作建議?"
        
        return self._generate_response(prompt, max_tokens=50)

    def analyze_prediction_results(self, stock_info: Dict[str, Any], 
                                  predictions: List[float],
                                  model_metrics: Dict[str, Any] = None) -> str:
        """分析預測結果 - 極簡版本，只輸出操作建議"""
        
        current_price = stock_info.get('price', 0)
        symbol = stock_info.get('symbol', '')
        
        if not predictions or current_price == 0:
            return "預測數據不足"
        
        # 計算變化率
        day5_change = (predictions[-1] / current_price - 1) * 100
        
        # 簡單規則判斷，避免 LLM 廢話
        if abs(day5_change) < 2:
            return f"預測變動{day5_change:+.1f}%，建議觀望"
        elif day5_change > 5:
            target = predictions[-1] * 1.02
            stop = current_price * 0.95
            return f"預測上漲{day5_change:+.1f}%，可分批進場，目標{target:.0f}，停損{stop:.0f}"
        elif day5_change < -5:
            return f"預測下跌{abs(day5_change):.1f}%，建議減碼或觀望"
        else:
            direction = "上漲" if day5_change > 0 else "下跌"
            return f"預測{direction}{abs(day5_change):.1f}%，謹慎操作"
    
    def analyze_technical_indicators(self, indicators: Dict[str, Any], 
                                   stock_info: Dict[str, Any]) -> str:
        """分析技術指標 - 極簡版本"""
        
        current_price = stock_info.get('price', 0)
        symbol = stock_info.get('symbol', '')
        rsi = indicators.get('rsi', 50)
        ma5 = indicators.get('ma5', current_price)
        ma20 = indicators.get('ma20', current_price)
        
        # 用規則判斷，避免 LLM 廢話
        signals = []
        
        # RSI 信號
        if rsi > 70:
            signals.append(f"RSI超買({rsi:.0f})")
        elif rsi < 30:
            signals.append(f"RSI超賣({rsi:.0f})")
        
        # 均線信號
        if current_price > ma5 > ma20:
            signals.append("多頭排列")
        elif current_price < ma5 < ma20:
            signals.append("空頭排列")
        
        # 給出建議
        if rsi > 80:
            return f"{', '.join(signals)}，建議觀望"
        elif rsi < 20:
            return f"{', '.join(signals)}，可關注進場"
        elif current_price > ma20 * 1.05:
            return f"{', '.join(signals)}，注意高檔風險"
        elif current_price < ma20 * 0.95:
            return f"{', '.join(signals)}，可留意支撐"
        else:
            return f"{', '.join(signals) if signals else '技術中性'}，持續觀察"

    def get_investment_advice(self, analysis_result: str,
                            risk_level: str = "medium") -> str:
        """投資建議 - 簡化版本"""

        prompt = f"""根據分析結果給建議(30字內):
{analysis_result[:100]}

直接說: 進場價位、停損點、目標價。"""

        return self._generate_response(prompt, max_tokens=80)

    def _generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """生成回應 - 優化版本"""

        # 限制 prompt 長度
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[:self.max_prompt_length] + "..."
            logger.warning("Prompt 過長，已截斷")

        # 設定 max_tokens
        if max_tokens is None:
            max_tokens = self.max_response_length

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # 降低隨機性
                    "top_p": 0.7,
                    "num_predict": max_tokens,
                    "stop": ["總結", "結論", "最後", "綜合"]
                }
            }

            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()['response'].strip()

                # 移除廢話開頭
                unwanted_starts = [
                    "根據提供的資料", "根據以上資訊", "從技術分析來看",
                    "綜合來看", "總的來說", "分析如下", "我的分析是"
                ]

                for start in unwanted_starts:
                    if result.startswith(start):
                        result = result[len(start):].lstrip("，。：")

                return result[:max_tokens]  # 確保不超過限制
            else:
                logger.error(f"LLM API 錯誤: {response.status_code}")
                return "分析暫時無法生成"

        except requests.exceptions.Timeout:
            logger.error("LLM 請求超時")
            return "分析請求超時"
        except Exception as e:
            logger.error(f"LLM 生成錯誤: {e}")
            return "分析生成失敗"

    def get_system_status(self) -> Dict[str, Any]:
        """獲取系統狀態"""
        return {
            'model': self.model,
            'base_url': self.base_url,
            'max_prompt_length': self.max_prompt_length,
            'max_response_length': self.max_response_length,
            'analysis_depth': self.analysis_depth
        }

# 向後相容的別名
class GemmaAnalyzer(OptimizedLLMAnalyzer):
    """向後相容的別名"""
    pass

if __name__ == "__main__":
    # 測試
    analyzer = OptimizedLLMAnalyzer()

    test_data = {
        'symbol': '2330',
        'name': '台積電',
        'price': 580.0,
        'change_percent': 2.5,
        'rsi': 65
    }

    print("🧪 測試分析：")
    result = analyzer.analyze_stock_quick(test_data)
    print(f"結果：{result}")
