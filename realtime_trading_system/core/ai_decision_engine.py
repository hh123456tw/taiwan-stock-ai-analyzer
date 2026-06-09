#\!/usr/bin/env python3
"""
AI決策引擎 - 基於AI預測進行智能投資決策
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json




from config.trading_config import TradingConfig
from .technical_indicators import SimpleTechnicalIndicators
from .market_sentiment_engine import MarketSentimentEngine

class AIDecisionEngine:
    """AI決策引擎 - 增強版支援技術指標與市場情緒"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.logger = self._setup_logger()
        self.decision_history = []
        
        # 初始化技術指標和市場情緒引擎
        self.technical_engine = SimpleTechnicalIndicators()
        self.sentiment_engine = MarketSentimentEngine()
        
        self.logger.info("✅ AI決策引擎已啟用技術指標和市場情緒分析")
        
    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger('AIDecisionEngine')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            return logger
            
        os.makedirs(self.config.DATA_CONFIG['log_dir'], exist_ok=True)
        
        log_file = os.path.join(self.config.DATA_CONFIG['log_dir'], 'ai_decision.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def analyze_market_sentiment(self, candidates: List[Dict]) -> Dict:
        """分析市場情緒"""
        if not candidates:
            return {
                'overall_sentiment': 'NEUTRAL',
                'bullish_ratio': 0,
                'bearish_ratio': 0,
                'avg_confidence': 0,
                'total_candidates': 0
            }
        
        buy_candidates = [c for c in candidates if c['action'] == 'BUY']
        sell_candidates = [c for c in candidates if c['action'] == 'SELL']
        
        total_candidates = len(candidates)
        bullish_ratio = len(buy_candidates) / total_candidates
        bearish_ratio = len(sell_candidates) / total_candidates
        avg_confidence = sum(c['confidence'] for c in candidates) / total_candidates
        
        # 判斷整體市場情緒
        if bullish_ratio > 0.6:
            overall_sentiment = 'BULLISH'
        elif bearish_ratio > 0.6:
            overall_sentiment = 'BEARISH'
        else:
            overall_sentiment = 'NEUTRAL'
        
        sentiment = {
            'overall_sentiment': overall_sentiment,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'avg_confidence': avg_confidence,
            'total_candidates': total_candidates,
            'buy_candidates': len(buy_candidates),
            'sell_candidates': len(sell_candidates)
        }
        
        self.logger.info(
            f'📊 市場情緒分析: {overall_sentiment} '
            f'(看漲{bullish_ratio:.1%}, 看跌{bearish_ratio:.1%}, '
            f'平均信心度{avg_confidence:.3f})'
        )
        
        return sentiment
    
    def calculate_risk_score(self, candidate: Dict, current_portfolio: Dict) -> float:
        """計算風險評分 (0-1, 越低越安全)"""
        risk_factors = []
        
        # 1. 信心度風險 (信心度越低風險越高)
        confidence_risk = 1 - candidate['confidence']
        risk_factors.append(('confidence', confidence_risk, 0.3))
        
        # 2. 價格變動幅度風險 (變動越大風險越高)
        price_change_risk = min(abs(candidate['predicted_change']), 0.1) / 0.1
        risk_factors.append(('price_change', price_change_risk, 0.2))
        
        # 3. 持股集中度風險
        symbol = candidate['symbol']
        current_positions = current_portfolio.get('positions', {})
        if symbol in current_positions:
            # 如果已經持有，增加持股風險
            concentration_risk = 0.3
        else:
            # 新股票，檢查是否會過度集中
            total_positions = len(current_positions)
            concentration_risk = min(total_positions / 10, 1.0) * 0.2
        risk_factors.append(('concentration', concentration_risk, 0.2))
        
        # 4. 市場時間風險 (非交易時間風險較高)
        time_risk = 0 if self.config.is_trading_time() else 0.5
        risk_factors.append(('timing', time_risk, 0.1))
        
        # 5. 模型一致性風險 (單一模型風險較高)
        model_risk = 0.2 if '單' in candidate.get('model', '') else 0.1
        risk_factors.append(('model', model_risk, 0.2))
        
        # 計算加權風險評分
        total_risk = sum(score * weight for _, score, weight in risk_factors)
        
        self.logger.debug(
            f'🎯 {symbol} 風險評分: {total_risk:.3f} '
            f'(信心度風險{confidence_risk:.3f}, 價格風險{price_change_risk:.3f})'
        )
        
        return total_risk
    
    def prioritize_candidates(self, candidates: List[Dict], current_portfolio: Dict, available_capital: float) -> List[Dict]:
        """對候選股票進行優先級排序"""
        if not candidates:
            return []
        
        enhanced_candidates = []
        
        for candidate in candidates:
            # 計算風險評分
            risk_score = self.calculate_risk_score(candidate, current_portfolio)
            
            # 計算投資金額
            investment_amount = self.config.get_investment_amount(
                candidate['confidence'], 
                available_capital
            )
            
            # 計算優先級分數 (信心度高、風險低、投資潛力大的優先)
            priority_score = (
                candidate['confidence'] * 0.4 +  # 信心度權重
                (1 - risk_score) * 0.3 +         # 風險權重 (越低越好)
                min(abs(candidate['predicted_change']), 0.1) * 3 * 0.3  # 收益潛力權重
            )
            
            # 增強候選資訊
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                'risk_score': risk_score,
                'priority_score': priority_score,
                'suggested_investment': investment_amount,
                'risk_level': self._get_risk_level(risk_score),
                'decision_timestamp': datetime.now().isoformat()
            })
            
            enhanced_candidates.append(enhanced_candidate)
        
        # 按優先級分數排序
        enhanced_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        self.logger.info(f'🎯 候選股票優先級排序完成，共 {len(enhanced_candidates)} 檔')
        
        # 記錄前5名
        for i, candidate in enumerate(enhanced_candidates[:5], 1):
            self.logger.info(
                f'{i}. {candidate["symbol"]} {candidate["action"]} - '
                f'優先級{candidate["priority_score"]:.3f}, '
                f'風險{candidate["risk_level"]}, '
                f'建議投資{candidate["suggested_investment"]:,.0f}元'
            )
        
        return enhanced_candidates
    
    def _get_risk_level(self, risk_score: float) -> str:
        """獲取風險等級"""
        if risk_score < 0.2:
            return '低風險'
        elif risk_score < 0.4:
            return '中低風險'
        elif risk_score < 0.6:
            return '中等風險'
        elif risk_score < 0.8:
            return '中高風險'
        else:
            return '高風險'
    
    def should_execute_trade(self, candidate: Dict, current_portfolio: Dict, risk_tolerance: float = 0.6) -> Tuple[bool, str]:
        """決策是否執行交易"""
        reasons = []
        
        # 1. 檢查風險容忍度
        if candidate['risk_score'] > risk_tolerance:
            return False, f'風險評分{candidate["risk_score"]:.3f}超過容忍度{risk_tolerance}'
        
        # 2. 檢查信心度
        if candidate['confidence'] < self.config.AI_CONFIG['min_confidence']:
            return False, f'信心度{candidate["confidence"]:.3f}低於最低要求{self.config.AI_CONFIG["min_confidence"]}'
        
        # 3. 檢查投資金額
        if candidate['suggested_investment'] < self.config.CAPITAL_CONFIG['min_single_investment']:
            return False, f'建議投資金額{candidate["suggested_investment"]:,.0f}元低於最低投資額'
        
        # 4. 檢查是否為交易時間
        if False:  # 強制啟用交易
            return False, '強制交易時間測試模式'
        
        # 5. 檢查持股限制
        current_positions = current_portfolio.get('positions', {})
        if candidate['action'] == 'BUY' and len(current_positions) >= self.config.CAPITAL_CONFIG['max_positions']:
            return False, f'持股檔數已達上限{self.config.CAPITAL_CONFIG["max_positions"]}檔'
        
        # 6. 檢查資金充足性
        available_capital = current_portfolio.get('available_capital', 0)
        if candidate['action'] == 'BUY' and candidate['suggested_investment'] > available_capital:
            return False, f'資金不足，需要{candidate["suggested_investment"]:,.0f}元，可用{available_capital:,.0f}元'
        
        # 7. 檢查是否買得起至少1張
        if candidate['action'] == 'BUY':
            current_price = candidate.get('current_price', candidate.get('predicted_price', 0))
            min_cost_per_lot = current_price * 1000  # 1張的成本
            if candidate['suggested_investment'] < min_cost_per_lot:
                return False, f'資金不足買1張，需要{min_cost_per_lot:,.0f}元，建議投資額僅{candidate["suggested_investment"]:,.0f}元'
        
        # 7. 檢查是否已持有該股票
        symbol = candidate['symbol']
        if candidate['action'] == 'BUY' and symbol in current_positions:
            # 可以考慮加碼，但需要檢查單檔持股比例
            current_value = current_positions[symbol].get('current_value', 0)
            total_value = current_value + candidate['suggested_investment']
            total_capital = current_portfolio.get('total_capital', 400000)
            position_ratio = total_value / total_capital
            
            if position_ratio > self.config.CAPITAL_CONFIG['max_position_ratio']:
                return False, f'加碼後單檔持股比例{position_ratio:.1%}將超過上限{self.config.CAPITAL_CONFIG["max_position_ratio"]:.1%}'
        
        if candidate['action'] == 'SELL' and symbol not in current_positions:
            return False, f'無{symbol}持股，無法賣出'
        
        # 所有檢查通過
        reasons.append(f'風險評分{candidate["risk_score"]:.3f}可接受')
        reasons.append(f'信心度{candidate["confidence"]:.3f}符合要求')
        reasons.append(f'投資金額{candidate["suggested_investment"]:,.0f}元合理')
        
        return True, '; '.join(reasons)
    
    def make_trading_decisions(self, candidates: List[Dict], current_portfolio: Dict) -> List[Dict]:
        """做出交易決策"""
        self.logger.info(f'🧠 開始AI交易決策分析，候選股票 {len(candidates)} 檔')
        
        if not candidates:
            self.logger.warning('⚠️ 無候選股票可供決策')
            return []
        
        # 1. 分析市場情緒
        market_sentiment = self.analyze_market_sentiment(candidates)
        
        # 2. 優先級排序
        available_capital = current_portfolio.get('available_capital', 400000)
        prioritized_candidates = self.prioritize_candidates(candidates, current_portfolio, available_capital)
        
        # 3. 決策執行
        trading_decisions = []
        total_planned_investment = 0
        
        for candidate in prioritized_candidates:
            should_trade, reason = self.should_execute_trade(candidate, current_portfolio)
            
            if should_trade:
                # 檢查總投資限制
                planned_investment = candidate['suggested_investment']
                if total_planned_investment + planned_investment <= available_capital * 0.8:  # 不超過80%資金
                    decision = {
                        'symbol': candidate['symbol'],
                        'action': candidate['action'],
                        'investment_amount': planned_investment,
                        'shares': max((int(planned_investment / candidate['current_price']) // 1000) * 1000, 0),  # 整張，最少0股
                        'target_price': candidate['current_price'],
                        'confidence': candidate['confidence'],
                        'risk_score': candidate['risk_score'],
                        'risk_level': candidate['risk_level'],
                        'priority_score': candidate['priority_score'],
                        'decision_reason': candidate['reason'],
                        'approval_reason': reason,
                        'predicted_return': candidate['predicted_change_pct'],
                        'model': candidate['model'],
                        'decision_timestamp': datetime.now().isoformat(),
                        'market_sentiment': market_sentiment['overall_sentiment']
                    }
                    
                    trading_decisions.append(decision)
                    total_planned_investment += planned_investment
                    
                    self.logger.info(
                        f'✅ 決策通過: {candidate["symbol"]} {candidate["action"]} '
                        f'{planned_investment:,.0f}元 - {reason}'
                    )
                else:
                    self.logger.warning(
                        f'💰 資金限制: {candidate["symbol"]} 將超過80%資金使用限制'
                    )
            else:
                self.logger.info(
                    f'❌ 決策拒絕: {candidate["symbol"]} {candidate["action"]} - {reason}'
                )
        
        # 記錄決策歷史
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'approved_decisions': len(trading_decisions),
            'total_planned_investment': total_planned_investment,
            'market_sentiment': market_sentiment,
            'decisions': trading_decisions
        }
        self.decision_history.append(decision_record)
        
        self.logger.info(
            f'🎯 AI決策完成: {len(trading_decisions)}/{len(candidates)} 筆交易獲得批准, '
            f'計劃投資 {total_planned_investment:,.0f} 元'
        )
        
        return trading_decisions
    
    def get_decision_summary(self) -> Dict:
        """獲取決策摘要"""
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'total_approved': 0,
                'approval_rate': 0,
                'avg_investment_per_decision': 0
            }
        
        recent_decisions = self.decision_history[-10:]  # 最近10次決策
        
        total_candidates = sum(d['total_candidates'] for d in recent_decisions)
        total_approved = sum(d['approved_decisions'] for d in recent_decisions)
        total_investment = sum(d['total_planned_investment'] for d in recent_decisions)
        
        return {
            'total_decisions': len(self.decision_history),
            'recent_candidates': total_candidates,
            'recent_approved': total_approved,
            'approval_rate': total_approved / total_candidates if total_candidates > 0 else 0,
            'avg_investment_per_decision': total_investment / total_approved if total_approved > 0 else 0,
            'last_decision_time': self.decision_history[-1]['timestamp'] if self.decision_history else None
        }
    
    def make_enhanced_trading_decisions(self, candidates: List[Dict], current_portfolio: Dict) -> List[Dict]:
        """增強版交易決策 - 整合技術指標和市場情緒"""
        if not candidates:
            return []
        
        self.logger.info(f"🚀 開始增強版AI決策分析，候選股票 {len(candidates)} 檔")
        
        enhanced_decisions = []
        available_capital = current_portfolio.get("available_capital", 0)
        
        for candidate in candidates:
            try:
                symbol = candidate.get('symbol', candidate.get('stock_code'))
                if not symbol:
                    continue
                
                # 1. 獲取技術指標評分
                technical_score = self.technical_engine.get_technical_score(symbol)
                
                # 2. 獲取市場情緒評分
                sentiment_score = self.sentiment_engine.get_market_sentiment_score(symbol)
                
                # 3. 計算增強版信心度
                original_confidence = candidate.get('confidence', 0.5)
                
                # 綜合評分權重：原始AI(60%) + 技術指標(25%) + 市場情緒(15%)
                enhanced_confidence = (
                    original_confidence * 0.60 +
                    technical_score * 0.25 +
                    sentiment_score * 0.15
                )
                
                # 確保在0-1範圍內
                enhanced_confidence = max(0.0, min(1.0, enhanced_confidence))
                
                # 4. 技術指標和情緒過濾
                if self._should_filter_by_technicals(candidate, technical_score, sentiment_score):
                    self.logger.info(f"⚠️ {symbol} 被技術指標/情緒過濾器排除")
                    continue
                
                # 5. 計算投資金額
                investment_amount = self.config.get_investment_amount(enhanced_confidence, available_capital)
                price = candidate.get('price', candidate.get('target_price', 0))
                quantity = int(investment_amount / price) if price > 0 else 0
                
                if quantity == 0:
                    continue
                
                # 6. 生成增強版決策
                enhanced_decision = {
                    'stock_code': symbol,
                    'symbol': symbol,
                    'action': candidate['action'],
                    'quantity': quantity,
                    'price': price,
                    'investment_amount': investment_amount,
                    'confidence': enhanced_confidence,
                    'original_confidence': original_confidence,
                    'technical_score': technical_score,
                    'sentiment_score': sentiment_score,
                    'reasoning': self._generate_enhanced_reasoning(
                        candidate, technical_score, sentiment_score, enhanced_confidence
                    ),
                    'model': candidate.get('model', 'ensemble'),
                    'predicted_change': candidate.get('predicted_change', 0),
                    'decision_timestamp': datetime.now().isoformat(),
                    'enhancement_version': '2.0'
                }
                
                enhanced_decisions.append(enhanced_decision)
                
                self.logger.info(
                    f"✅ 增強決策: {symbol} {candidate['action']} "
                    f"信心度 {original_confidence:.3f}→{enhanced_confidence:.3f} "
                    f"(技術={technical_score:.3f}, 情緒={sentiment_score:.3f})"
                )
                
            except Exception as e:
                self.logger.error(f"❌ 增強版決策處理失敗 {candidate}: {e}")
                continue
        
        self.logger.info(f"🎯 增強版決策完成，最終批准 {len(enhanced_decisions)} 檔交易")
        return enhanced_decisions
    
    def _should_filter_by_technicals(self, candidate: Dict, technical_score: float, sentiment_score: float) -> bool:
        """基於技術指標和情緒進行過濾"""
        action = candidate['action']
        
        # 買入信號過濾
        if action == 'BUY':
            # 技術指標太差不買入
            if technical_score < 0.3:
                return True
            # 市場情緒太悲觀不買入  
            if sentiment_score < 0.25:
                return True
            # 技術指標和情緒都偏差不買入
            if technical_score < 0.4 and sentiment_score < 0.4:
                return True
        
        # 賣出信號過濾
        elif action == 'SELL':
            # 技術指標太好不賣出
            if technical_score > 0.7:
                return True
            # 市場情緒太樂觀不賣出
            if sentiment_score > 0.75:
                return True
        
        return False
    
    def _generate_enhanced_reasoning(self, candidate: Dict, technical_score: float, 
                                   sentiment_score: float, enhanced_confidence: float) -> str:
        """生成增強版決策理由"""
        action = candidate['action']
        symbol = candidate.get('symbol', '')
        
        # 基礎理由
        base_reason = candidate.get('reason', f'{action}信號')
        
        # 技術分析描述
        if technical_score >= 0.7:
            tech_desc = "技術面強勢"
        elif technical_score >= 0.6:
            tech_desc = "技術面偏多"
        elif technical_score <= 0.3:
            tech_desc = "技術面疲弱"
        elif technical_score <= 0.4:
            tech_desc = "技術面偏空"
        else:
            tech_desc = "技術面中性"
        
        # 情緒分析描述
        if sentiment_score >= 0.7:
            sentiment_desc = "市場情緒樂觀"
        elif sentiment_score >= 0.6:
            sentiment_desc = "市場情緒偏多"
        elif sentiment_score <= 0.3:
            sentiment_desc = "市場情緒悲觀"
        elif sentiment_score <= 0.4:
            sentiment_desc = "市場情緒偏空"
        else:
            sentiment_desc = "市場情緒中性"
        
        # 信心度變化描述
        original_conf = candidate.get('confidence', 0.5)
        if enhanced_confidence > original_conf + 0.05:
            conf_desc = "技術面和情緒面支撐，信心度提升"
        elif enhanced_confidence < original_conf - 0.05:
            conf_desc = "技術面和情緒面不利，信心度下調"
        else:
            conf_desc = "綜合分析維持原判斷"
        
        return f"{base_reason} - {tech_desc}，{sentiment_desc}，{conf_desc}"

# 測試函數
def test_ai_decision_engine():
    """測試AI決策引擎"""
    print('🧪 測試AI決策引擎')
    
    engine = AIDecisionEngine()
    
    # 模擬候選股票
    test_candidates = [
        {
            'symbol': '2330',
            'action': 'BUY',
            'predicted_price': 950,
            'current_price': 930,
            'predicted_change': 0.0215,
            'predicted_change_pct': 2.15,
            'confidence': 0.85,
            'model': '集成(5模型)',
            'reason': 'AI預測上漲2.15%, 信心度0.85'
        },
        {
            'symbol': '2454',
            'action': 'SELL',
            'predicted_price': 1080,
            'current_price': 1100,
            'predicted_change': -0.018,
            'predicted_change_pct': -1.8,
            'confidence': 0.75,
            'model': '集成(4模型)',
            'reason': 'AI預測下跌1.8%, 信心度0.75'
        },
        {
            'symbol': '2882',
            'action': 'BUY',
            'predicted_price': 65,
            'current_price': 62,
            'predicted_change': 0.048,
            'predicted_change_pct': 4.8,
            'confidence': 0.92,
            'model': '集成(5模型)',
            'reason': 'AI預測上漲4.8%, 信心度0.92'
        }
    ]
    
    # 模擬當前投資組合
    test_portfolio = {
        'total_capital': 400000,
        'available_capital': 350000,
        'positions': {
            '1101': {
                'shares': 2000,
                'avg_cost': 120,
                'current_value': 240000
            }
        }
    }
    
    # 執行決策
    decisions = engine.make_trading_decisions(test_candidates, test_portfolio)
    
    print(f'\n📊 決策結果:')
    print(f'批准交易: {len(decisions)} 筆')
    
    for i, decision in enumerate(decisions, 1):
        print(f'{i}. {decision["symbol"]} {decision["action"]} - {decision["investment_amount"]:,.0f}元')
        print(f'   風險等級: {decision["risk_level"]}, 優先級: {decision["priority_score"]:.3f}')
        print(f'   決策原因: {decision["approval_reason"]}')
    
    # 獲取決策摘要
    summary = engine.get_decision_summary()
    print(f'\n📈 決策摘要: {summary}')

if __name__ == '__main__':
    test_ai_decision_engine()
