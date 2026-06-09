"""
增強型 LLM 分析器 - 修復版含情緒分析
"""
from llm_analyzer import OptimizedLLMAnalyzer

# 導入新聞情緒分析器
try:
    from news_sentiment_analyzer import NewsSentimentAnalyzer
    SENTIMENT_AVAILABLE = True
    print("✅ 新聞情緒分析模組已載入")
except ImportError as e:
    SENTIMENT_AVAILABLE = False
    print(f"⚠️ 新聞情緒分析模組載入失敗: {e}")

# 導入市場情緒分析器
try:
    from market_sentiment_analyzer import MarketSentimentAnalyzer
    MARKET_SENTIMENT_AVAILABLE = True
    print("✅ 市場情緒分析模組已載入")
except ImportError as e:
    MARKET_SENTIMENT_AVAILABLE = False
    print(f"⚠️ 市場情緒分析模組載入失敗: {e}")

class EnhancedLLMAnalyzer(OptimizedLLMAnalyzer):
    """增強型 LLM 分析器 - 修復所有方法"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # 初始化情緒分析器
        if SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = NewsSentimentAnalyzer()
                print("✅ 新聞情緒分析器已載入")
            except Exception as e:
                self.sentiment_analyzer = None
                print(f"⚠️ 新聞情緒分析器初始化失敗: {e}")
        else:
            self.sentiment_analyzer = None
            
        # 初始化市場情緒分析器
        if MARKET_SENTIMENT_AVAILABLE:
            try:
                self.market_sentiment_analyzer = MarketSentimentAnalyzer()
                print("✅ 市場情緒分析器已載入")
            except Exception as e:
                self.market_sentiment_analyzer = None
                print(f"⚠️ 市場情緒分析器初始化失敗: {e}")
        else:
            self.market_sentiment_analyzer = None
            
        # 總結初始化狀態
        features = []
        if self.sentiment_analyzer:
            features.append("新聞情緒")
        if self.market_sentiment_analyzer:
            features.append("市場情緒")
            
        if features:
            print(f"✅ 增強型LLM分析器已初始化 (含{', '.join(features)}分析)")
        else:
            print("✅ 增強型LLM分析器已初始化 (基本版)")
        
    def analyze_prediction_ensemble(self, stock_info, predictions_dict, us_analysis=None):
        """分析集成預測結果 - 修復版含美股關聯"""
        try:
            symbol = stock_info.get('symbol', '')
            name = stock_info.get('name', '')
            price = stock_info.get('price', 0)
            
            analysis_parts = []
            
            if predictions_dict:
                models = list(predictions_dict.keys())
                # 計算平均預測
                avg_pred = sum([pred[-1] for pred in predictions_dict.values()]) / len(predictions_dict)
                change_pct = (avg_pred / price - 1) * 100
                
                if change_pct > 2:
                    sentiment = "多數模型偏多"
                elif change_pct < -2:
                    sentiment = "多數模型偏空"
                else:
                    sentiment = "模型分歧較大"
                    
                analysis_parts.append(f"{name}集成分析: {sentiment}，預期變動{change_pct:+.1f}%")
            else:
                analysis_parts.append(f"{name}技術分析: 價格{price:.2f}，建議持續觀察")
            
            # 加入美股關聯分析
            if us_analysis:
                try:
                    from cross_market_predictor import cross_market_predictor
                    us_prediction = cross_market_predictor.predict_tw_with_us_signals(symbol)
                    
                    if us_prediction:
                        direction = us_prediction.get('direction', '震盪')
                        confidence = us_prediction.get('confidence', 0)
                        emoji = us_prediction.get('emoji', '➡️')
                        analysis_text = us_prediction.get('analysis', [])
                        
                        analysis_parts.append(f"🇺🇸 美股關聯: {direction} {emoji} (信心度:{confidence:.0f}%)")
                        
                        # 加入具體的美股分析
                        if analysis_text and len(analysis_text) > 0:
                            analysis_parts.extend(analysis_text[:2])  # 最多加入2條分析
                    else:
                        analysis_parts.append("🇺🇸 美股關聯: 數據暫時無法獲取")
                        
                except Exception as us_e:
                    analysis_parts.append("🇺🇸 美股關聯: 分析服務暫時不可用")
            
            # 加入新聞情緒分析
            if self.sentiment_analyzer:
                try:
                    sentiment_result = self.sentiment_analyzer.analyze_stock_news_sentiment(symbol, days=2)
                    
                    if sentiment_result['total_news'] > 0:
                        sentiment = sentiment_result['sentiment_summary']
                        confidence = sentiment_result['confidence']
                        news_count = sentiment_result['total_news']
                        
                        # 情緒emoji
                        if sentiment == "正面":
                            sentiment_emoji = "📈"
                        elif sentiment == "負面":
                            sentiment_emoji = "📉"
                        else:
                            sentiment_emoji = "📊"
                        
                        analysis_parts.append(f"📰 新聞情緒: {sentiment} {sentiment_emoji} ({news_count}則新聞, 信心度:{confidence:.0%})")
                    else:
                        analysis_parts.append("📰 新聞情緒: 相關新聞不足")
                        
                except Exception as sentiment_e:
                    analysis_parts.append("📰 新聞情緒: 分析服務暫時不可用")
            
            # 加入市場情緒分析
            if self.market_sentiment_analyzer:
                try:
                    market_sentiment = self.market_sentiment_analyzer.analyze_comprehensive_market_sentiment()
                    
                    overall_sentiment = market_sentiment['overall_sentiment']
                    overall_emoji = market_sentiment['overall_emoji']
                    final_score = market_sentiment['final_score']
                    risk_level = market_sentiment['risk_level']
                    
                    # 獲取關鍵指標
                    vix_data = market_sentiment['indicators'].get('vix', {})
                    vix_value = vix_data.get('current_vix', 'N/A')
                    
                    analysis_parts.append(f"🌍 市場情緒: {overall_sentiment} {overall_emoji} (VIX:{vix_value}, 風險:{risk_level})")
                    
                    # 加入關鍵信號
                    key_signals = market_sentiment.get('key_signals', [])
                    if key_signals:
                        analysis_parts.append(f"⚠️ {key_signals[0]}")  # 只顯示第一個關鍵信號
                        
                except Exception as market_sentiment_e:
                    analysis_parts.append("🌍 市場情緒: 分析服務暫時不可用")
            
            result = " | ".join(analysis_parts)
            
            # 新增矛盾分析
            contradiction_analysis = self._analyze_signal_contradictions(predictions_dict, us_analysis)
            if contradiction_analysis:
                result += f" | 🔍 {contradiction_analysis}"
            
            return result
                
        except Exception as e:
            return f"集成分析: 多模型綜合評估"
    
    def analyze_technical_comprehensive(self, indicators, stock_info, market_regime="normal"):
        """全面技術分析 - 增強版本包含完整移動平均線"""
        try:
            current_price = stock_info.get('price', 0)
            symbol = stock_info.get('symbol', '')
            name = stock_info.get('name', symbol)
            
            # 基本技術指標
            rsi = indicators.get('rsi', 50)
            ma5 = indicators.get('ma5', current_price)
            ma10 = indicators.get('ma10', current_price) 
            ma20 = indicators.get('ma20', current_price)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            # 移動平均線分析
            ma_analysis = []
            if current_price > ma5 > ma10 > ma20:
                ma_analysis.append("多頭排列強勢")
                trend_score = 3
            elif current_price > ma5 > ma20:
                ma_analysis.append("短期多頭")
                trend_score = 2
            elif current_price < ma5 < ma10 < ma20:
                ma_analysis.append("空頭排列")
                trend_score = -3
            elif current_price < ma5 < ma20:
                ma_analysis.append("短期空頭")
                trend_score = -2
            else:
                ma_analysis.append("盤整格局")
                trend_score = 0
            
            # RSI 分析
            if rsi > 80:
                rsi_analysis = f"RSI嚴重超買({rsi:.0f})"
                rsi_score = -2
            elif rsi > 70:
                rsi_analysis = f"RSI超買({rsi:.0f})"
                rsi_score = -1
            elif rsi < 20:
                rsi_analysis = f"RSI嚴重超賣({rsi:.0f})"
                rsi_score = 2
            elif rsi < 30:
                rsi_analysis = f"RSI超賣({rsi:.0f})"
                rsi_score = 1
            else:
                rsi_analysis = f"RSI中性({rsi:.0f})"
                rsi_score = 0
            
            # 成交量分析
            if volume_ratio > 2.0:
                volume_analysis = f"爆量({volume_ratio:.1f}倍)"
            elif volume_ratio > 1.5:
                volume_analysis = f"量增({volume_ratio:.1f}倍)"
            elif volume_ratio < 0.5:
                volume_analysis = f"量縮({volume_ratio:.1f}倍)"
            else:
                volume_analysis = f"量平({volume_ratio:.1f}倍)"
            
            # 綜合評分
            total_score = trend_score + rsi_score
            if total_score >= 3:
                overall = "強勢看多"
            elif total_score >= 1:
                overall = "偏多"
            elif total_score <= -3:
                overall = "強勢看空"
            elif total_score <= -1:
                overall = "偏空"
            else:
                overall = "中性觀望"
            
            # 移動平均線數據
            ma_data = f"MA5:{ma5:.2f} MA10:{ma10:.2f} MA20:{ma20:.2f}"
            
            return f"{name} 技術面: {', '.join(ma_analysis)}, {rsi_analysis}, {volume_analysis}, {ma_data}, 綜合研判: {overall}"
            
        except Exception as e:
            return f"技術指標: RSI{indicators.get('rsi', 50):.0f}，MA分析異常，持續觀察"
    
    def analyze_technical_quick(self, indicators, stock_info):
        """快速技術分析 - 補充缺失方法"""
        return self.analyze_technical_indicators(indicators, stock_info)
    
    def get_ai_investment_strategy(self, analysis, strategy_type="balanced", risk_factor=1.0, market_regime="normal"):
        """AI投資策略 - 智能矛盾分析版"""
        try:
            # 分析各種信號
            ai_bearish = "偏空" in analysis or "看跌" in analysis
            tech_bullish = "偏多" in analysis or "多頭" in analysis
            us_bullish = "美股關聯: 看漲" in analysis
            weak_consensus = "弱共識" in analysis or "分歧" in analysis
            
            # 矛盾分析邏輯
            if ai_bearish and tech_bullish:
                if us_bullish:
                    return """建議: 🔄 **訊號分歧，謹慎為上**
• AI模型偏空 vs 技術面偏多 vs 美股偏多
• 短期技術面強勢，但量化模型警示風險
• 可小量試探，嚴格停損，等待方向明確
• 關注美股半導體板塊動向作為確認信號"""
                else:
                    return """建議: ⚠️ **空方信號較強**
• AI模型與技術面分歧，偏向模型判斷
• 技術面多頭可能是假突破
• 建議觀望，等待更明確的進場點"""
            elif not ai_bearish and tech_bullish and us_bullish:
                return """建議: ✅ **多方信號一致**
• AI模型、技術面、美股關聯三重確認
• 可考慮分批進場，設定合理停損
• 密切關注成交量配合"""
            elif weak_consensus:
                return """建議: 🤔 **市場處於轉折點**
• 各種信號分歧嚴重，方向不明
• 建議減少曝險，等待明確突破
• 可設定觀察區間，突破後再進場"""
            else:
                return "建議: 持續關注市場變化，謹慎操作"
                
        except Exception as e:
            return "建議: 分析異常，建議專業諮詢"
    
    def _analyze_signal_contradictions(self, predictions_dict, us_analysis):
        """分析信號矛盾並提供解釋"""
        try:
            if not predictions_dict:
                return None
                
            # 計算AI模型分歧度
            if len(predictions_dict) >= 3:
                pred_changes = []
                for pred in predictions_dict.values():
                    if len(pred) > 0:
                        pred_changes.append(pred[-1])
                
                if len(pred_changes) >= 3:
                    positive_count = sum(1 for change in pred_changes if change > 0)
                    negative_count = len(pred_changes) - positive_count
                    
                    # 判斷是否存在顯著分歧
                    if abs(positive_count - negative_count) <= 1:
                        return "訊號矛盾原因: AI模型間嚴重分歧，市場可能處於轉折點"
                    elif max(pred_changes) - min(pred_changes) > 1.0:  # 假設超過1元的差異算大
                        return "訊號分歧: 不同模型對風險評估差異極大，建議等待方向明確"
            
            return None
            
        except Exception:
            return None
            
    def analyze_stock_quick(self, stock_data):
        """快速股票分析 - 補充方法"""
        try:
            price = stock_data.get('price', 0)
            change = stock_data.get('change_percent', 0)
            rsi = stock_data.get('rsi', 50)
            
            if change > 3:
                return f"今日漲幅較大({change:+.1f}%)，注意高檔風險"
            elif change < -3:
                return f"今日跌幅較大({change:+.1f}%)，可關注反彈"
            elif rsi > 70:
                return f"RSI偏高({rsi:.0f})，建議謹慎"
            elif rsi < 30:
                return f"RSI偏低({rsi:.0f})，可留意機會"
            else:
                return f"價格{price:.2f}，技術面中性"
        except Exception as e:
            return "技術面分析: 建議持續觀察"
