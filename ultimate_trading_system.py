#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
終極版台股AI交易系統 - 智能動態權重版本
整合改進版動態權重學習，支援多市場情境適應
"""

import sys
import os
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 設定路徑



# 導入核心模組
from improved_dynamic_weight_learner import ImprovedDynamicWeightLearner
from technical_indicators import SimpleTechnicalIndicators
from market_sentiment_engine import MarketSentimentEngine

class UltimateTradingSystem:
    """終極版智能交易系統"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # 載入股票分類配置
        self.stock_config = self._load_stock_config()
        
        # 初始化各個引擎
        self.technical_engine = SimpleTechnicalIndicators()
        self.sentiment_engine = MarketSentimentEngine()
        
        # 🚀 改進版動態權重學習器
        self.weight_learner = ImprovedDynamicWeightLearner()
        
        # 模擬持倉
        self.current_positions = {}
        
        # 智能風控參數
        self.max_total_investment = 1000000  # 100萬
        self.position_limits = {
            'short_term': 0.08,   # 8%倉位
            'long_term': 0.20,    # 20%倉位  
            'hybrid': 0.12        # 12%倉位
        }
        
        # 🆕 動態閾值系統
        self.dynamic_thresholds = {
            'base_buy': 0.55,
            'base_sell': 0.45,
            'volatility_adjustment': True,
            'confidence_boost': True
        }
        
        # 交易統計
        self.trading_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'weight_adaptations': 0,
            'market_conditions': []
        }
        
        self.logger.info('🚀 終極版智能交易系統啟動')
    
    def _setup_logger(self) -> logging.Logger:
        """設定日誌"""
        logger = logging.getLogger('ultimate_trading_system')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_stock_config(self) -> Dict:
        """載入股票分類配置"""
        try:
            config_path = '/mnt/nvme/taiwan-stock-ai-analyzer-main/stock_pool_config_enhanced.json'
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # 處理配置格式問題，創建測試股票池
                if not any(isinstance(v, list) and v for v in config.values()):
                    # 如果配置不正確，使用測試股票池
                    test_config = {
                        'short_term': ['2330', '2454', '2317'],  # 台積電、聯發科、鴻海
                        'long_term': ['2412', '2882', '2303'],   # 中華電、國泰金、聯電
                        'hybrid': ['2409', '6505', '3711']       # 友達、台塑化、日月光
                    }
                    self.logger.info(f'✅ 使用測試股票池: 短期3支 + 長期3支 + 混合3支')
                    return test_config
                else:
                    self.logger.info(f'✅ 載入股票配置成功')
                    return config
                    
        except Exception as e:
            self.logger.error(f'❌ 載入股票配置失敗: {e}')
            # 使用預設測試配置
            return {
                'short_term': ['2330', '2454', '2317'],
                'long_term': ['2412', '2882', '2303'],
                'hybrid': ['2409', '6505', '3711']
            }
    
    def get_intelligent_score(self, symbol: str, category: str) -> Dict:
        """獲取智能綜合評分"""
        try:
            # 模擬AI預測評分
            predictions_score = self._get_mock_ai_prediction(symbol)
            
            # 獲取技術指標評分
            technical_score = self.technical_engine.get_technical_score(symbol)
            
            # 獲取市場情緒評分
            sentiment_score = self.sentiment_engine.get_market_sentiment_score(symbol)
            
            # 🚀 使用改進版動態權重學習器
            try:
                # 獲取當前市場特徵
                market_features = self.weight_learner.collect_market_features_safe()
                
                # 預測最佳權重
                weights = self.weight_learner.predict_weights_enhanced(market_features)
                
                # 計算智能加權評分
                final_score = (weights[0] * predictions_score + 
                              weights[1] * technical_score + 
                              weights[2] * sentiment_score)
                
                # 市場條件分析
                market_condition = self._analyze_market_condition(market_features)
                
                # 記錄智能權重使用情況
                self.logger.info(f"🧠 {symbol} 智能權重: AI={weights[0]:.3f}({weights[0]*100:.1f}%) "
                                f"技術={weights[1]:.3f}({weights[1]*100:.1f}%) "
                                f"情緒={weights[2]:.3f}({weights[2]*100:.1f}%)")
                self.logger.info(f"📊 市場狀態: {market_condition}")
                
                # 統計權重調整
                self.trading_stats['weight_adaptations'] += 1
                self.trading_stats['market_conditions'].append(market_condition)
                                
            except Exception as e:
                # 權重學習失敗時使用分類權重
                self.logger.warning(f"⚠️ 智能權重預測失敗，使用分類權重: {e}")
                
                if category == 'short_term':
                    final_score = (predictions_score * 0.4) + (technical_score * 0.4) + (sentiment_score * 0.2)
                    weights = [0.4, 0.4, 0.2]
                elif category == 'long_term':
                    final_score = (predictions_score * 0.6) + (technical_score * 0.2) + (sentiment_score * 0.2)
                    weights = [0.6, 0.2, 0.2]
                else:  # hybrid
                    final_score = (predictions_score * 0.5) + (technical_score * 0.3) + (sentiment_score * 0.2)
                    weights = [0.5, 0.3, 0.2]
                
                market_condition = "unknown"
            
            return {
                'symbol': symbol,
                'category': category,
                'predictions_score': predictions_score,
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'final_score': final_score,
                'weights_used': weights,
                'market_condition': market_condition,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f'❌ 計算 {symbol} 智能評分失敗: {e}')
            return {
                'symbol': symbol,
                'category': category,
                'final_score': 0.5,
                'error': str(e)
            }
    
    def _analyze_market_condition(self, market_features: Dict) -> str:
        """分析市場狀態"""
        try:
            volatility = market_features.get('volatility', 1.0)
            rsi = market_features.get('market_rsi', 50.0)
            trend_strength = market_features.get('trend_strength', 1.0)
            
            # 市場狀態分類
            if volatility > 2.5:
                if rsi < 30:
                    return "高波動熊市"
                elif rsi > 70:
                    return "高波動牛市"
                else:
                    return "高波動震盪"
            elif volatility < 1.0:
                if trend_strength > 3.0:
                    return "低波動趨勢"
                else:
                    return "低波動整理"
            else:
                if rsi > 60:
                    return "中波動偏多"
                elif rsi < 40:
                    return "中波動偏空"
                else:
                    return "中波動平衡"
                    
        except Exception:
            return "unknown"
    
    def get_dynamic_thresholds(self, market_condition: str, confidence: float) -> tuple:
        """根據市場狀態動態調整買賣閾值"""
        base_buy = self.dynamic_thresholds['base_buy']
        base_sell = self.dynamic_thresholds['base_sell']
        
        # 根據市場狀態調整
        if "高波動" in market_condition:
            # 高波動時提高閾值，更謹慎
            buy_threshold = base_buy + 0.05
            sell_threshold = base_sell + 0.05
        elif "低波動" in market_condition:
            # 低波動時降低閾值，更積極
            buy_threshold = base_buy - 0.03
            sell_threshold = base_sell - 0.03
        elif "趨勢" in market_condition:
            # 趨勢市場時更積極追趨勢
            buy_threshold = base_buy - 0.02
            sell_threshold = base_sell - 0.02
        else:
            buy_threshold = base_buy
            sell_threshold = base_sell
        
        # 根據信心度微調
        if confidence > 0.8:
            buy_threshold -= 0.02
            sell_threshold += 0.02
        elif confidence < 0.4:
            buy_threshold += 0.03
            sell_threshold -= 0.03
        
        return max(0.5, buy_threshold), min(0.5, sell_threshold)
    
    def intelligent_trade_decision(self, symbol: str, category: str) -> Dict:
        """智能交易決策"""
        try:
            # 獲取智能綜合評分
            score_data = self.get_intelligent_score(symbol, category)
            
            # 動態閾值
            market_condition = score_data.get('market_condition', 'unknown')
            confidence = score_data['final_score']
            buy_threshold, sell_threshold = self.get_dynamic_thresholds(market_condition, confidence)
            
            # 獲取當前位置
            current_position = self.current_positions.get(symbol, {'shares': 0, 'avg_price': 0})
            
            # 智能交易決策
            if score_data['final_score'] > buy_threshold:
                result = self._execute_intelligent_buy(symbol, category, score_data, current_position, buy_threshold)
                
            elif score_data['final_score'] < sell_threshold:
                result = self._execute_intelligent_sell(symbol, category, score_data, current_position, sell_threshold)
                
            else:
                # 持有
                result = {
                    'action': 'hold',
                    'symbol': symbol,
                    'category': category,
                    'reason': f'評分 {score_data["final_score"]:.3f} 在持有區間 [{sell_threshold:.3f}, {buy_threshold:.3f}]',
                    'market_condition': market_condition,
                    'score_data': score_data
                }
            
            # 統計交易
            self.trading_stats['total_trades'] += 1
            if result.get('success', False):
                self.trading_stats['successful_trades'] += 1
            
            return result
                
        except Exception as e:
            self.logger.error(f'❌ 智能交易決策 {symbol} 失敗: {e}')
            return {'error': str(e), 'symbol': symbol}
    
    def _execute_intelligent_buy(self, symbol: str, category: str, score_data: Dict, current_position: Dict, threshold: float) -> Dict:
        """執行智能買入"""
        try:
            max_position_value = self.max_total_investment * self.position_limits[category]
            
            # 根據信心度調整買入數量
            confidence = score_data['final_score']
            confidence_multiplier = min(1.5, confidence / 0.5)  # 最多1.5倍
            
            suggested_shares = int(max_position_value * confidence_multiplier / 100000)
            
            if suggested_shares > 0:
                self.logger.info(f'🟢 智能買入: {symbol} ({category}) - 評分: {score_data["final_score"]:.3f} (閾值: {threshold:.3f})')
                
                # 更新模擬持倉
                self.current_positions[symbol] = {
                    'shares': current_position['shares'] + suggested_shares * 1000,
                    'avg_price': 100,
                    'category': category,
                    'buy_time': datetime.now(),
                    'confidence': confidence,
                    'market_condition': score_data.get('market_condition', 'unknown'),
                    'weights_used': score_data.get('weights_used', [0.4, 0.3, 0.3]),
                    'buy_threshold': threshold
                }
                
                return {
                    'action': 'buy',
                    'symbol': symbol,
                    'category': category,
                    'shares': suggested_shares * 1000,
                    'estimated_cost': suggested_shares * 100000,
                    'confidence_multiplier': confidence_multiplier,
                    'score_data': score_data,
                    'success': True
                }
            else:
                return {
                    'action': 'skip',
                    'symbol': symbol,
                    'reason': '超出倉位限制',
                    'score_data': score_data
                }
                
        except Exception as e:
            self.logger.error(f'❌ 執行 {symbol} 智能買入失敗: {e}')
            return {'error': str(e), 'action': 'buy_failed'}
    
    def _execute_intelligent_sell(self, symbol: str, category: str, score_data: Dict, current_position: Dict, threshold: float) -> Dict:
        """執行智能賣出"""
        try:
            if current_position['shares'] > 0:
                self.logger.info(f'🔴 智能賣出: {symbol} ({category}) - 評分: {score_data["final_score"]:.3f} (閾值: {threshold:.3f})')
                
                shares_to_sell = current_position['shares']
                
                # 更新模擬持倉
                if symbol in self.current_positions:
                    del self.current_positions[symbol]
                
                return {
                    'action': 'sell',
                    'symbol': symbol,
                    'category': category,
                    'shares': shares_to_sell,
                    'estimated_revenue': shares_to_sell * 100,
                    'score_data': score_data,
                    'success': True
                }
            else:
                return {
                    'action': 'skip',
                    'symbol': symbol,
                    'reason': '無持倉可賣出',
                    'score_data': score_data
                }
                
        except Exception as e:
            self.logger.error(f'❌ 執行 {symbol} 智能賣出失敗: {e}')
            return {'error': str(e), 'action': 'sell_failed'}
    
    def _get_mock_ai_prediction(self, symbol: str) -> float:
        """模擬AI預測評分"""
        import random
        random.seed(hash(symbol) % 1000)
        
        base_score = 0.5
        code_num = int(''.join(filter(str.isdigit, symbol)) or '2330')
        adjustment = (code_num % 100) / 200 - 0.25
        
        score = base_score + adjustment + random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, score))
    
    def run_ultimate_analysis(self):
        """執行終極版智能分析"""
        self.logger.info('🚀 開始終極版智能市場分析...')
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'results': {},
            'summary': {
                'total_analyzed': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'errors': 0,
                'weight_adaptations': 0,
                'market_conditions': {}
            }
        }
        
        # 分析各類股票
        for category, stocks in self.stock_config.items():
            if not stocks:
                continue
                
            category_results = []
            
            for stock in stocks[:2]:  # 每類分析2支，避免過多
                result = self.intelligent_trade_decision(stock, category)
                category_results.append(result)
                
                # 統計
                results['summary']['total_analyzed'] += 1
                action = result.get('action', 'error')
                if action == 'buy':
                    results['summary']['buy_signals'] += 1
                elif action == 'sell':
                    results['summary']['sell_signals'] += 1
                elif action == 'hold':
                    results['summary']['hold_signals'] += 1
                else:
                    results['summary']['errors'] += 1
                
                # 市場狀態統計
                market_condition = result.get('market_condition', 'unknown')
                if market_condition in results['summary']['market_conditions']:
                    results['summary']['market_conditions'][market_condition] += 1
                else:
                    results['summary']['market_conditions'][market_condition] = 1
                
                time.sleep(0.1)
            
            results['results'][category] = category_results
        
        # 更新統計
        results['summary']['weight_adaptations'] = self.trading_stats['weight_adaptations']
        
        # 輸出結果摘要
        summary = results['summary']
        self.logger.info(f'📊 智能分析完成: 總計{summary["total_analyzed"]} 買入{summary["buy_signals"]} 賣出{summary["sell_signals"]} 持有{summary["hold_signals"]} 錯誤{summary["errors"]}')
        self.logger.info(f'🧠 權重調整: {summary["weight_adaptations"]} 次')
        self.logger.info(f'📈 市場狀態分布: {summary["market_conditions"]}')
        
        return results

def main():
    """主程式"""
    print("🚀 啟動終極版台股AI智能交易系統")
    
    # 初始化系統
    trading_system = UltimateTradingSystem()
    
    # 執行智能分析
    results = trading_system.run_ultimate_analysis()
    
    # 輸出結果
    print("\n📊 終極版交易分析結果:")
    print(json.dumps(results['summary'], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()