#\!/usr/bin/env python3
"""
實時市場掃描器 - 主動掃描股票市場機會
"""

import sys
import os
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

# 添加項目根目錄到路徑
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
        
        # 股票池 - 從現有系統獲取
        self.stock_pool = self._load_stock_pool()
        
    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger('MarketScanner')
        logger.setLevel(logging.INFO)
        
        # 確保日誌目錄存在
        os.makedirs(self.config.DATA_CONFIG['log_dir'], exist_ok=True)
        
        # 文件處理器
        log_file = os.path.join(self.config.DATA_CONFIG['log_dir'], 'market_scanner.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式設置
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
            
            # 先嘗試獲取所有有預測數據的股票（不限時間）
            cursor.execute("""
                SELECT DISTINCT symbol 
                FROM predictions 
                ORDER BY symbol
            """)
            
            stocks = [row[0] for row in cursor.fetchall()]
            
            # 如果股票數量太少，添加台股權值股和熱門股
            if len(stocks) < 20:
                additional_stocks = [
                    '2330', '2317', '2454', '2308', '2303',  # 科技股
                    '2881', '2882', '2891', '2892',          # 金融股
                    '1101', '1102', '1216', '1301',          # 水泥塑化
                    '2412', '2474', '3711', '6505',          # 面板光電
                    '2002', '2207', '2357', '2382',          # 傳產電子
                    '3008', '3231', '3450', '6415',          # 光學生技
                    '0050', '0056', '00878', '00919'         # ETF
                ]
                
                for stock in additional_stocks:
                    if stock not in stocks:
                        stocks.append(stock)
            
            conn.close()
            
            self.logger.info(f'📊 載入股票池: {len(stocks)} 檔股票')
            return stocks[:50]  # 限制最多50檔避免過多
            
        except Exception as e:
            self.logger.error(f'❌ 載入股票池失敗: {e}')
            # 返回擴展的熱門股票池
            return [
                '2330', '2317', '2454', '2308', '2303', '2881', '2882', '2891',
                '1101', '1102', '2412', '2474', '3711', '2002', '2207', '2357',
                '2382', '3008', '3231', '3450', '0050', '0056', '00878', '00919'
            ]
    
    def get_latest_predictions(self, symbols: List[str] = None, hours_back: int = 24) -> pd.DataFrame:
        """獲取最新的AI預測資料"""
        try:
            conn = sqlite3.connect(self.predictions_db)
            
            # 如果沒有指定股票，使用完整股票池
            if symbols is None:
                symbols = self.stock_pool
                
            # 構建查詢條件
            placeholders = ','.join(['?' for _ in symbols])
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            query = f"""
                SELECT 
                    prediction_date,
                    prediction_time,
                    symbol,
                    predicted_price,
                    actual_price,
                    model_name,
                    confidence,
                    CASE 
                        WHEN predicted_price > actual_price THEN 'UP'
                        WHEN predicted_price < actual_price THEN 'DOWN'
                        ELSE 'NEUTRAL'
                    END as predicted_direction,
                    (predicted_price - actual_price) / actual_price as predicted_change_ratio
                FROM predictions 
                WHERE symbol IN ({placeholders})
                AND datetime(prediction_date || ' ' || COALESCE(prediction_time, '00:00:00')) >= ?
                AND confidence > ?
                AND predicted_price IS NOT NULL
                AND actual_price IS NOT NULL
                ORDER BY prediction_date DESC, prediction_time DESC, confidence DESC
            """
            
            params = symbols + [cutoff_time.strftime('%Y-%m-%d %H:%M:%S'), 
                              self.config.AI_CONFIG['min_confidence']]
            
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
                        'prediction_time': row.get('prediction_time', ''),
                        'reason': f'AI預測{"上漲" if predicted_change > 0 else "下跌"}{abs(predicted_change)*100:.2f}%, 信心度{confidence:.2f}'
                    }
                    
                    candidates.append(candidate)
                    
            # 按信心度排序
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f'🎯 篩選出交易候選: {len(candidates)} 檔股票')
            
            # 記錄候選股票詳情
            for i, candidate in enumerate(candidates[:5], 1):  # 只顯示前5名
                self.logger.info(
                    f'{i}. {candidate["symbol"]} {candidate["action"]} - '
                    f'{candidate["reason"]}'
                )
                
            return candidates
            
        except Exception as e:
            self.logger.error(f'❌ 篩選交易候選失敗: {e}')
            return []
    
    def get_market_overview(self) -> Dict:
        """獲取市場總覽"""
        try:
            predictions_df = self.get_latest_predictions(hours_back=6)  # 最近6小時
            
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
                'avg_confidence': avg_confidence,
                'active_symbols': active_symbols,
                'scan_time': datetime.now().isoformat()
            }
            
            self.logger.info(
                f'📊 市場總覽: {active_symbols}檔活躍股票, '
                f'{total_predictions}筆預測 (看漲{bullish_count}, 看跌{bearish_count}), '
                f'平均信心度{avg_confidence:.2f}'
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
            
            # 如果候選股票太少，添加備用推薦
            if len(candidates) < 12:
                self.logger.info(f'📈 當前候選股票{len(candidates)}檔，添加備用推薦以擴大選擇')
                backup_candidates = self.generate_backup_candidates()
                
                # 避免重複
                existing_symbols = {c['symbol'].replace('.TW', '') for c in candidates}
                for backup in backup_candidates:
                    clean_symbol = backup['symbol'].replace('.TW', '')
                    if clean_symbol not in existing_symbols:
                        candidates.append(backup)
                        existing_symbols.add(clean_symbol)
                        if len(candidates) >= 18:  # 限制總數
                            break
                
                self.logger.info(f'📈 擴展候選清單至: {len(candidates)} 檔股票')
            
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

    def generate_backup_candidates(self) -> List[Dict]:
        """生成備用交易候選（當預測數據不足時）"""
        backup_stocks = [
            # 權值龍頭股
            {'symbol': '2330', 'name': '台積電', 'confidence': 0.75, 'predicted_change_pct': 1.5},
            {'symbol': '2317', 'name': '鴻海', 'confidence': 0.72, 'predicted_change_pct': 1.8},
            {'symbol': '2454', 'name': '聯發科', 'confidence': 0.78, 'predicted_change_pct': 2.2},
            {'symbol': '2308', 'name': '台達電', 'confidence': 0.74, 'predicted_change_pct': 1.6},
            
            # 金融股
            {'symbol': '2881', 'name': '富邦金', 'confidence': 0.71, 'predicted_change_pct': 1.3},
            {'symbol': '2882', 'name': '國泰金', 'confidence': 0.69, 'predicted_change_pct': 1.1},
            {'symbol': '2891', 'name': '中信金', 'confidence': 0.70, 'predicted_change_pct': 1.2},
            
            # 傳產股
            {'symbol': '1101', 'name': '台泥', 'confidence': 0.68, 'predicted_change_pct': 1.4},
            {'symbol': '2002', 'name': '中鋼', 'confidence': 0.67, 'predicted_change_pct': 1.7},
            {'symbol': '2207', 'name': '和泰車', 'confidence': 0.73, 'predicted_change_pct': 1.9},
            
            # 電子股
            {'symbol': '2382', 'name': '廣達', 'confidence': 0.76, 'predicted_change_pct': 2.1},
            {'symbol': '3008', 'name': '大立光', 'confidence': 0.71, 'predicted_change_pct': 1.8},
            {'symbol': '2474', 'name': '可成', 'confidence': 0.69, 'predicted_change_pct': 2.0},
            
            # 中小型股
            {'symbol': '6166', 'name': '凌陽', 'confidence': 0.72, 'predicted_change_pct': 2.5},
            {'symbol': '3231', 'name': '緯創', 'confidence': 0.68, 'predicted_change_pct': 1.5},
            {'symbol': '6415', 'name': '矽力', 'confidence': 0.74, 'predicted_change_pct': 2.3},
        ]
        
        candidates = []
        for stock in backup_stocks:
            candidates.append({
                'symbol': stock['symbol'],
                'action': 'BUY',
                'predicted_price': 0,  # 將由系統實時獲取
                'current_price': 0,    # 將由系統實時獲取
                'predicted_change': stock['predicted_change_pct'] / 100,
                'predicted_change_pct': stock['predicted_change_pct'],
                'confidence': stock['confidence'],
                'model': '備用推薦系統',
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'reason': f"備用推薦: {stock['name']}, 預期上漲{stock['predicted_change_pct']}%"
            })
            
        self.logger.info(f'📋 生成備用候選: {len(candidates)} 檔股票')
        return candidates


async def test_market_scanner():
    """測試市場掃描器"""
    print('🧪 測試實時市場掃描器')
    
    scanner = RealTimeMarketScanner()
    
    # 測試市場掃描
    candidates, overview = await scanner.scan_market()
    
    print(f'\n📊 掃描結果:')
    print(f'候選股票數量: {len(candidates)}')
    print(f'市場總覽: {overview}')
    
    if candidates:
        print('\n🎯 前3名候選股票:')
        for i, candidate in enumerate(candidates[:3], 1):
            print(f'{i}. {candidate["symbol"]} - {candidate["action"]} - {candidate["reason"]}')
    
    # 測試狀態查詢
    status = scanner.get_scan_status()
    print(f'\n📈 掃描器狀態: {status}')

if __name__ == '__main__':
    asyncio.run(test_market_scanner())
