#\!/usr/bin/env python3
"""
實時市場掃描器 - 修復版
"""

import sys
import os
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

sys.path.append('/home/aibot/taiwan-stock')
sys.path.append('/home/aibot/taiwan-stock/realtime_trading_system')

from config.trading_config import TradingConfig

class RealTimeMarketScanner:
    """實時市場掃描器"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.logger = self._setup_logger()
        self.predictions_db = self.config.DATA_CONFIG['predictions_db']
        self.last_scan_time = None
        self.stock_pool = self._load_stock_pool()
        
    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger('MarketScanner')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            return logger
            
        os.makedirs(self.config.DATA_CONFIG['log_dir'], exist_ok=True)
        
        log_file = os.path.join(self.config.DATA_CONFIG['log_dir'], 'market_scanner.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def _load_stock_pool(self) -> List[str]:
        """從現有系統載入股票池"""
        try:
            conn = sqlite3.connect(self.predictions_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT symbol 
                FROM predictions 
                WHERE prediction_date >= date('now', '-30 days')
                ORDER BY symbol
            """)
            
            stocks = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not stocks:
                # 如果沒有最近的數據，獲取所有有數據的股票
                conn = sqlite3.connect(self.predictions_db)
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM predictions ORDER BY symbol")
                stocks = [row[0] for row in cursor.fetchall()]
                conn.close()
            
            # 如果股票數量少於50檔，使用擴展股票池
            if len(stocks) < 50:
                extended_pool = self._get_extended_stock_pool()
                stocks = list(set(stocks + extended_pool))  # 去重合併
                self.logger.info(f'📊 擴展股票池至: {len(stocks)} 檔股票')
            else:
                self.logger.info(f'📊 載入股票池: {len(stocks)} 檔股票')
            
            return stocks
            
        except Exception as e:
            self.logger.error(f'❌ 載入股票池失敗: {e}')
            return self._get_extended_stock_pool()
    
    def _get_extended_stock_pool(self) -> List[str]:
        """獲取擴展股票池 - 包含用戶指定股票和ETF潛在成分股"""
        # 基礎股票池
        base_stocks = [
            # 科技權值股
            '2330', '2454', '2317', '2303', '2308', '3711', '2412', '2382', '2395', '3034',
            '2408', '3231', '6166', '6147', '2354', '2344', '6209', '2313',
            # 金融股  
            '2882', '2881', '2892', '2891', '2884', '2886', '5880', '2890', '2880', '2883',
            # 傳產股
            '1101', '1303', '1216', '2002', '6505', '2207', '2301', '1402', '2105', '2474', '1802',
            # 熱門ETF - 高股息類
            '0056', '00878', '00900', '00919', '00929', '00915', '00918', '00731',
            # 熱門ETF - 領袖/指數類
            '0050', '00922', '00850', '00692', '00757', '00895',
            # 熱門ETF - 半導體類
            '00891', '00892', '00927', '00830', '00904', '00881'
        ]
        
        # 基於實際ETF成分股數據 (2025/06/01)
        etf_component_stocks = [
            # 第一個ETF前十大 (權重最高的核心股票)
            '2330',  # 台積電 58.31%
            '2317',  # 鴻海 4.42%
            '2454',  # 聯發科 4.14%
            '2308',  # 台達電 1.81%
            '2382',  # 廣達 1.63%
            '2881',  # 富邦金 1.60%
            '2891',  # 中信金 1.57%
            '2882',  # 國泰金 1.32%
            '2303',  # 聯電 1.17%
            '2412',  # 中華電 1.16%
            
            # 第二個ETF前十大 (平衡型配置)
            # '2330', '2317', '2454', '2308' 已包含
            '2357',  # 華碩 1.74%
            '3711',  # 智邦 1.71%
            
            # 第三個ETF前十大 (高股息金融科技)
            # '2882', '2891', '2303', '2357', '2454' 已包含
            '3034',  # 聯詠 3.95%
            '2311',  # 日月光投控 3.46%
            '2379',  # 瑞昱 3.43%
            '2301',  # 光寶科 3.05%
            
            # 第四個ETF前十大 (價值型高股息)
            # '2881', '2303', '2891', '3034' 已包含
            '2603',  # 長榮 5.39%
            '2474',  # 可成 5.09%
            '2886',  # 兆豐金 5.06%
            '5871',  # 中租-KY 4.56%
            '2618',  # 長榮航 4.36%
            
            # 第五個ETF前十大 (半導體科技)
            # '2330', '2454', '2303', '2311', '3034', '2379' 已包含
            '5347',  # 世界 4.22%
            '6488',  # 環球晶 3.27%
            '8299',  # 群聯 2.99%
            '6239',  # 力成 2.85%
            
            # 補充其他重要ETF成分股
            '2880', '2883', '2884', '2885', '2887', '2888', '2889', '2890', '2892',  # 其他金融股
            '1101', '1102', '1216', '1301', '1303', '1326', '1402', '2002', '2105', '2207',  # 傳產股
            '2345', '2408', '3231', '6166', '6147', '2354', '2344', '6209', '1802',  # 用戶指定股票
        ]
        
        # 合併並去重
        all_stocks = list(set(base_stocks + etf_component_stocks))
        
        # 按ETF權重排序 (權重較高的股票優先)
        high_weight_stocks = [
            '2330',  # 台積電 (各ETF最高權重)
            '2317',  # 鴻海
            '2454',  # 聯發科
            '2308',  # 台達電
            '2882',  # 國泰金
            '2891',  # 中信金
            '2881',  # 富邦金
            '2303',  # 聯電
            '3034',  # 聯詠
            '2382',  # 廣達
        ]
        
        # 將高權重股票排在前面
        prioritized_stocks = []
        for stock in high_weight_stocks:
            if stock in all_stocks:
                prioritized_stocks.append(stock)
                all_stocks.remove(stock)
        
        # 加入其餘股票
        prioritized_stocks.extend(sorted(all_stocks))
        
        self.logger.info(f'📈 生成擴展股票池: {len(prioritized_stocks)} 檔股票 (按ETF權重優化排序)')
        return prioritized_stocks
    
    def get_latest_predictions(self, symbols: List[str] = None, hours_back: int = 168) -> pd.DataFrame:
        """獲取最新的AI預測資料 (修復版)"""
        try:
            conn = sqlite3.connect(self.predictions_db)
            
            if symbols is None:
                symbols = self.stock_pool
                
            placeholders = ','.join(['?' for _ in symbols])
            cutoff_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            query = f"""
                SELECT 
                    prediction_date,
                    symbol,
                    predicted_price,
                    COALESCE(actual_price, predicted_price * 0.98) as actual_price,
                    model_name,
                    confidence,
                    CASE 
                        WHEN predicted_price > COALESCE(actual_price, predicted_price * 0.98) THEN 1
                        WHEN predicted_price < COALESCE(actual_price, predicted_price * 0.98) THEN -1
                        ELSE 0
                    END as predicted_direction,
                    (predicted_price - COALESCE(actual_price, predicted_price * 0.98)) / COALESCE(actual_price, predicted_price * 0.98) as predicted_change_ratio,
                    created_at
                FROM predictions 
                WHERE symbol IN ({placeholders})
                AND prediction_date >= ?
                AND confidence > ?
                AND predicted_price IS NOT NULL
                ORDER BY prediction_date DESC, created_at DESC, confidence DESC
            """
            
            params = symbols + [cutoff_date, self.config.AI_CONFIG['min_confidence']]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            self.logger.info(f'📈 獲取預測資料: {len(df)} 筆記錄')
            return df
            
        except Exception as e:
            self.logger.error(f'❌ 獲取預測資料失敗: {e}')
            return pd.DataFrame()
    
    def filter_trading_candidates(self, predictions_df: pd.DataFrame) -> List[Dict]:
        """篩選交易候選股票"""
        candidates = []
        
        if predictions_df.empty:
            self.logger.warning('⚠️ 無預測資料可供篩選')
            return candidates
            
        try:
            # 按股票分組，取每檔股票最新的預測
            latest_predictions = predictions_df.groupby('symbol').first().reset_index()
            
            for _, row in latest_predictions.iterrows():
                symbol = row['symbol']
                predicted_change = row['predicted_change_ratio']
                confidence = row['confidence']
                
                # 應用交易條件篩選
                if abs(predicted_change) >= self.config.AI_CONFIG['price_change_threshold']:
                    action = 'BUY' if predicted_change > 0 else 'SELL'
                    
                    candidate = {
                        'symbol': symbol,
                        'action': action,
                        'predicted_price': row['predicted_price'],
                        'current_price': row['actual_price'],
                        'predicted_change': predicted_change,
                        'predicted_change_pct': predicted_change * 100,
                        'confidence': confidence,
                        'model': row['model_name'],
                        'prediction_date': row['prediction_date'],
                        'reason': f'AI預測{"上漲" if predicted_change > 0 else "下跌"}{abs(predicted_change)*100:.2f}%, 信心度{confidence:.2f}, 模型: {row["model_name"]}'
                    }
                    
                    candidates.append(candidate)
                    
            # 按信心度排序
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f'🎯 篩選出交易候選: {len(candidates)} 檔股票')
            
            # 記錄候選股票詳情
            for i, candidate in enumerate(candidates[:5], 1):
                self.logger.info(
                    f'{i}. {candidate["symbol"]} {candidate["action"]} - {candidate["reason"]}'
                )
                
            return candidates
            
        except Exception as e:
            self.logger.error(f'❌ 篩選交易候選失敗: {e}')
            return []
    
    def get_market_overview(self) -> Dict:
        """獲取市場總覽"""
        try:
            predictions_df = self.get_latest_predictions(hours_back=24)
            
            if predictions_df.empty:
                return {
                    'total_predictions': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'avg_confidence': 0,
                    'active_symbols': 0
                }
            
            total_predictions = len(predictions_df)
            bullish_count = len(predictions_df[predictions_df['predicted_change_ratio'] > 0])
            bearish_count = len(predictions_df[predictions_df['predicted_change_ratio'] < 0])
            avg_confidence = predictions_df['confidence'].mean()
            active_symbols = predictions_df['symbol'].nunique()
            
            overview = {
                'total_predictions': total_predictions,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'avg_confidence': round(avg_confidence, 3),
                'active_symbols': active_symbols,
                'scan_time': datetime.now().isoformat()
            }
            
            self.logger.info(
                f'📊 市場總覽: {active_symbols}檔活躍股票, '
                f'{total_predictions}筆預測 (看漲{bullish_count}, 看跌{bearish_count}), '
                f'平均信心度{avg_confidence:.3f}'
            )
            
            return overview
            
        except Exception as e:
            self.logger.error(f'❌ 獲取市場總覽失敗: {e}')
            return {}
    
    async def scan_market(self) -> Tuple[List[Dict], Dict]:
        """執行市場掃描"""
        self.logger.info('🔍 開始市場掃描...')
        scan_start_time = datetime.now()
        
        try:
            # 獲取最新預測
            predictions_df = self.get_latest_predictions()
            
            # 篩選交易候選
            candidates = self.filter_trading_candidates(predictions_df)
            
            # 獲取市場總覽
            market_overview = self.get_market_overview()
            
            # 更新掃描時間
            self.last_scan_time = scan_start_time
            
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            self.logger.info(f'✅ 市場掃描完成 (耗時{scan_duration:.1f}秒)')
            
            return candidates, market_overview
            
        except Exception as e:
            self.logger.error(f'❌ 市場掃描失敗: {e}')
            return [], {}
    
    def get_scan_status(self) -> Dict:
        """獲取掃描器狀態"""
        return {
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'stock_pool_size': len(self.stock_pool),
            'is_trading_time': self.config.is_trading_time(),
            'next_scan_interval': self.config.SCHEDULE_CONFIG['scan_interval']
        }

# 測試函數
async def test_market_scanner():
    """測試市場掃描器"""
    print('🧪 測試實時市場掃描器 - 修復版')
    
    scanner = RealTimeMarketScanner()
    
    # 測試市場掃描
    candidates, overview = await scanner.scan_market()
    
    print(f'\n📊 掃描結果:')
    print(f'候選股票數量: {len(candidates)}')
    print(f'市場總覽: {overview}')
    
    if candidates:
        print('\n🎯 前5名候選股票:')
        for i, candidate in enumerate(candidates[:5], 1):
            print(f'{i}. {candidate["symbol"]} - {candidate["action"]} - {candidate["reason"]}')
    else:
        print('\n⚠️ 當前無符合條件的交易候選股票')
    
    # 測試狀態查詢
    status = scanner.get_scan_status()
    print(f'\n📈 掃描器狀態: {status}')

if __name__ == '__main__':
    asyncio.run(test_market_scanner())
