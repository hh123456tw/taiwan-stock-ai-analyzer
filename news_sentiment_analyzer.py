#!/usr/bin/env python3
"""
財經新聞情緒分析器
支援多來源新聞爬取與情緒分析
"""
import requests
import jieba
import re
from snownlp import SnowNLP
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging
import time
import json
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """財經新聞情緒分析器"""
    
    def __init__(self):
        """初始化情緒分析器"""
        self.financial_keywords = {
            'positive': [
                '上漲', '漲停', '獲利', '成長', '突破', '創新高', '看好', '買進',
                '利多', '強勢', '反彈', '回升', '轉強', '樂觀', '推升', '增長',
                '收復', '站上', '攻克', '衝高', '勁揚', '大漲', '飆漲', '暴漲'
            ],
            'negative': [
                '下跌', '跌停', '虧損', '衰退', '跌破', '創新低', '看空', '賣出',
                '利空', '弱勢', '重挫', '下滑', '轉弱', '悲觀', '拖累', '減少',
                '失守', '跌破', '崩跌', '急跌', '暴跌', '大跌', '重跌', '慘跌'
            ]
        }
        
        # 初始化jieba分詞
        jieba.initialize()
        
        # 載入自定義詞典
        for words in self.financial_keywords.values():
            for word in words:
                jieba.add_word(word)
                
        logger.info("✅ 新聞情緒分析器初始化完成")
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """分析文字情緒"""
        try:
            # 清理文字
            cleaned_text = self._clean_text(text)
            
            # SnowNLP基礎情緒分析
            snow = SnowNLP(cleaned_text)
            base_sentiment = snow.sentiments
            
            # 關鍵詞權重分析
            keyword_score = self._analyze_keywords(cleaned_text)
            
            # 綜合評分 (70% SnowNLP + 30% 關鍵詞)
            final_score = base_sentiment * 0.7 + keyword_score * 0.3
            
            # 分類
            if final_score > 0.6:
                sentiment = "正面"
            elif final_score < 0.4:
                sentiment = "負面"
            else:
                sentiment = "中性"
                
            return {
                'sentiment': sentiment,
                'score': final_score,
                'base_score': base_sentiment,
                'keyword_score': keyword_score,
                'confidence': abs(final_score - 0.5) * 2  # 0-1的信心度
            }
            
        except Exception as e:
            logger.error(f"情緒分析失敗: {e}")
            return {
                'sentiment': "中性",
                'score': 0.5,
                'base_score': 0.5,
                'keyword_score': 0.5,
                'confidence': 0.0
            }
    
    def _clean_text(self, text: str) -> str:
        """清理文字"""
        # 移除HTML標籤
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # 移除特殊字符，保留中文、英文、數字
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_keywords(self, text: str) -> float:
        """分析關鍵詞權重"""
        words = jieba.cut(text)
        
        positive_count = 0
        negative_count = 0
        
        for word in words:
            if word in self.financial_keywords['positive']:
                positive_count += 1
            elif word in self.financial_keywords['negative']:
                negative_count += 1
        
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.5  # 中性
        
        # 計算正面比例
        positive_ratio = positive_count / total_keywords
        return positive_ratio
    
    def crawl_yahoo_news(self, stock_symbol: str, days: int = 3) -> List[Dict]:
        """爬取Yahoo財經新聞"""
        news_list = []
        
        try:
            # Yahoo財經新聞URL (範例)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # 模擬搜尋股票相關新聞
            search_url = f"https://tw.stock.yahoo.com/quote/{stock_symbol}/news"
            
            logger.info(f"正在爬取 {stock_symbol} 的Yahoo新聞...")
            
            # 這裡只是示例結構，實際需要根據網站調整
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # 解析新聞 (需要根據實際HTML結構調整)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 示例：假設找到新聞標題和內容
                news_items = soup.find_all('div', class_='news-item')  # 假設class名
                
                for item in news_items[:10]:  # 取前10則
                    try:
                        title = item.find('h3').get_text().strip()
                        content = item.find('p').get_text().strip()
                        
                        news_list.append({
                            'source': 'Yahoo財經',
                            'title': title,
                            'content': content,
                            'url': search_url,
                            'timestamp': datetime.now(),
                            'stock_symbol': stock_symbol
                        })
                    except:
                        continue
                        
        except Exception as e:
            logger.error(f"Yahoo新聞爬取失敗: {e}")
            
        return news_list
    
    def crawl_cnyes_news(self, stock_symbol: str, days: int = 3) -> List[Dict]:
        """爬取鉅亨網新聞"""
        news_list = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # 鉅亨網搜尋URL
            search_url = f"https://news.cnyes.com/search?keyword={stock_symbol}"
            
            logger.info(f"正在爬取 {stock_symbol} 的鉅亨網新聞...")
            
            # 示例實作 - 需要根據實際網站結構調整
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # 這裡需要根據鉅亨網實際結構調整
                news_list.append({
                    'source': '鉅亨網',
                    'title': f'{stock_symbol} 相關新聞',
                    'content': '示例新聞內容',
                    'url': search_url,
                    'timestamp': datetime.now(),
                    'stock_symbol': stock_symbol
                })
                
        except Exception as e:
            logger.error(f"鉅亨網新聞爬取失敗: {e}")
            
        return news_list
    
    def analyze_stock_news_sentiment(self, stock_symbol: str, days: int = 3) -> Dict:
        """分析特定股票的新聞情緒"""
        logger.info(f"開始分析 {stock_symbol} 的新聞情緒...")
        
        # 爬取多來源新聞
        all_news = []
        all_news.extend(self.crawl_yahoo_news(stock_symbol, days))
        all_news.extend(self.crawl_cnyes_news(stock_symbol, days))
        
        if not all_news:
            logger.warning(f"未找到 {stock_symbol} 相關新聞")
            return {
                'stock_symbol': stock_symbol,
                'total_news': 0,
                'sentiment_summary': "無足夠數據",
                'average_score': 0.5,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'confidence': 0.0,
                'news_details': []
            }
        
        # 分析每則新聞情緒
        sentiment_results = []
        total_score = 0
        
        for news in all_news:
            # 分析標題+內容
            full_text = f"{news['title']} {news['content']}"
            sentiment = self.analyze_text_sentiment(full_text)
            
            sentiment_results.append({
                **news,
                **sentiment
            })
            
            total_score += sentiment['score']
        
        # 統計結果
        total_news = len(sentiment_results)
        average_score = total_score / total_news
        
        positive_count = sum(1 for r in sentiment_results if r['sentiment'] == '正面')
        negative_count = sum(1 for r in sentiment_results if r['sentiment'] == '負面')
        neutral_count = total_news - positive_count - negative_count
        
        positive_ratio = positive_count / total_news
        negative_ratio = negative_count / total_news
        neutral_ratio = neutral_count / total_news
        
        # 整體情緒判斷
        if average_score > 0.6:
            overall_sentiment = "正面"
        elif average_score < 0.4:
            overall_sentiment = "負面"
        else:
            overall_sentiment = "中性"
        
        # 信心度（基於新聞數量和分數分散度）
        confidence = min(1.0, total_news / 10) * (abs(average_score - 0.5) * 2)
        
        # 將新聞詳情中的時間戳轉換為字符串
        for news in sentiment_results:
            if 'timestamp' in news and isinstance(news['timestamp'], datetime):
                news['timestamp'] = news['timestamp'].isoformat()
        
        result = {
            'stock_symbol': stock_symbol,
            'total_news': total_news,
            'sentiment_summary': overall_sentiment,
            'average_score': round(average_score, 3),
            'positive_ratio': round(positive_ratio, 3),
            'negative_ratio': round(negative_ratio, 3),
            'neutral_ratio': round(neutral_ratio, 3),
            'confidence': round(confidence, 3),
            'analysis_time': datetime.now().isoformat(),
            'news_details': sentiment_results
        }
        
        logger.info(f"✅ {stock_symbol} 新聞情緒分析完成: {overall_sentiment} (信心度: {confidence:.2f})")
        return result

def main():
    """測試函數"""
    analyzer = NewsSentimentAnalyzer()
    
    # 測試文字情緒分析
    test_texts = [
        "台積電今日強勢上漲，創下歷史新高，投資人信心大增",
        "鴻海股價重挫，跌破重要支撐，市場情緒轉為悲觀",
        "台股盤整震盪，成交量萎縮，投資人觀望氣氛濃厚"
    ]
    
    print("\n=== 文字情緒分析測試 ===")
    for text in test_texts:
        result = analyzer.analyze_text_sentiment(text)
        print(f"文字: {text}")
        print(f"結果: {result['sentiment']} (評分: {result['score']:.3f}, 信心度: {result['confidence']:.3f})")
        print()
    
    # 測試股票新聞情緒分析
    print("=== 股票新聞情緒分析測試 ===")
    result = analyzer.analyze_stock_news_sentiment("2330")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()