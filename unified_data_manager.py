#!/usr/bin/env python3
"""
統一數據管理系統
解決數據源混亂問題，建立標準化的股價獲取和管理機制
"""

import yfinance as yf
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedDataManager:
    """統一數據管理器 - 所有模組的標準數據源"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or './data/unified_stock_data.db'
        self.predictions_db = './predictions.db'
        self.stock_pool_path = './stock_pool_config.json'
        
        # 確保資料庫存在
        self._init_database()
        
    def _init_database(self):
        """初始化統一數據庫"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_prices (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'yahoo_finance',
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    pe_ratio REAL,
                    dividend_yield REAL,
                    beta REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
    def get_stock_pool(self) -> List[str]:
        """獲取當前股票池"""
        try:
            with open(self.stock_pool_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('active_stocks', [])
        except Exception as e:
            logger.error(f"無法讀取股票池配置: {e}")
            return []
    
    def fetch_latest_prices(self, symbols: List[str] = None, force_update: bool = False) -> Dict[str, Dict]:
        """
        獲取最新股價數據 (統一入口)
        
        Args:
            symbols: 股票代號列表，None則使用完整股票池
            force_update: 是否強制更新數據
            
        Returns:
            Dict: {股票代號: {price_data}}
        """
        if symbols is None:
            symbols = self.get_stock_pool()
            
        logger.info(f"📊 開始獲取 {len(symbols)} 檔股票的最新價格...")
        
        stock_data = {}
        today = datetime.now().date()
        
        for symbol in symbols:
            try:
                # 檢查是否需要更新 (今日是否已有數據)
                if not force_update:
                    existing_data = self._get_cached_price(symbol, today)
                    if existing_data:
                        stock_data[symbol] = existing_data
                        continue
                
                # 從Yahoo Finance獲取數據
                yahoo_symbol = f"{symbol}.TW"
                ticker = yf.Ticker(yahoo_symbol)
                
                # 獲取歷史數據
                hist = ticker.history(period="5d")
                if hist.empty:
                    logger.warning(f"⚠️ {symbol}: 無法獲取歷史數據")
                    continue
                
                # 獲取基本資訊
                info = ticker.info
                
                # 處理最新價格
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                
                price_data = {
                    'symbol': symbol,
                    'current_price': round(float(latest['Close']), 2),
                    'open_price': round(float(latest['Open']), 2),
                    'high_price': round(float(latest['High']), 2),
                    'low_price': round(float(latest['Low']), 2),
                    'prev_close': round(float(prev['Close']), 2),
                    'volume': int(latest['Volume']),
                    'change': round(float(latest['Close'] - prev['Close']), 2),
                    'change_pct': round(float((latest['Close'] - prev['Close']) / prev['Close'] * 100), 2),
                    'date': today.strftime('%Y-%m-%d'),
                    'name': info.get('longName', symbol),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': round(float(info.get('dividendYield', 0) * 100), 2) if info.get('dividendYield') else 0,
                    'market_cap': info.get('marketCap', 0),
                    'one_lot_cost': round(float(latest['Close'] * 1000), 0),  # 台股1張成本
                    'source': 'yahoo_finance',
                    'updated_at': datetime.now().isoformat()
                }
                
                # 儲存到資料庫
                self._save_price_data(symbol, hist, info)
                stock_data[symbol] = price_data
                
                logger.info(f"✅ {symbol}: ${price_data['current_price']} ({price_data['change']:+.2f}, {price_data['change_pct']:+.2f}%)")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: 獲取數據失敗 - {e}")
                continue
                
        logger.info(f"📈 完成! 成功獲取 {len(stock_data)}/{len(symbols)} 檔股票數據")
        return stock_data
    
    def _get_cached_price(self, symbol: str, date) -> Optional[Dict]:
        """檢查是否有快取的價格數據"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT close, open, high, low, volume, date
                    FROM stock_prices 
                    WHERE symbol = ? AND date = ?
                ''', (symbol, date))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'symbol': symbol,
                        'current_price': row[0],
                        'open_price': row[1],
                        'high_price': row[2],
                        'low_price': row[3],
                        'volume': row[4],
                        'date': row[5],
                        'source': 'cache'
                    }
        except Exception as e:
            logger.debug(f"快取檢查失敗: {e}")
        return None
    
    def _save_price_data(self, symbol: str, hist_data: pd.DataFrame, info: Dict):
        """儲存價格數據到資料庫"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 儲存歷史價格
                for date, row in hist_data.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO stock_prices 
                        (symbol, date, open, high, low, close, volume, adj_close, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        date.date(),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row['Close']),  # adj_close
                        'yahoo_finance'
                    ))
                
                # 儲存股票基本資訊
                conn.execute('''
                    INSERT OR REPLACE INTO stock_info
                    (symbol, name, sector, industry, market_cap, pe_ratio, dividend_yield, beta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    info.get('longName', symbol),
                    info.get('sector', ''),
                    info.get('industry', ''),
                    info.get('marketCap', 0),
                    info.get('trailingPE', 0),
                    info.get('dividendYield', 0),
                    info.get('beta', 0)
                ))
                
        except Exception as e:
            logger.error(f"儲存數據失敗: {e}")
    
    def get_training_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        獲取訓練用歷史數據
        
        Args:
            symbol: 股票代號
            days: 歷史天數
            
        Returns:
            DataFrame: 包含OHLCV和技術指標的數據
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT date, open, high, low, close, volume
                    FROM stock_prices 
                    WHERE symbol = ? 
                    ORDER BY date DESC 
                    LIMIT ?
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, days))
                
                if df.empty:
                    logger.warning(f"⚠️ {symbol}: 無歷史數據，嘗試即時獲取...")
                    # 如果沒有數據，嘗試即時獲取
                    self.fetch_latest_prices([symbol], force_update=True)
                    df = pd.read_sql_query(query, conn, params=(symbol, days))
                
                # 確保數據順序 (舊到新)
                df = df.sort_values('date').reset_index(drop=True)
                df['date'] = pd.to_datetime(df['date'])
                
                # 計算技術指標
                df = self._add_technical_indicators(df)
                
                return df
                
        except Exception as e:
            logger.error(f"獲取訓練數據失敗: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術指標"""
        if df.empty:
            return df
            
        try:
            # 移動平均線
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 成交量移動平均
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            
            # 價格相對位置
            df['price_position'] = (df['close'] - df['low'].rolling(window=20).min()) / (
                df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
            )
            
            # MA比率 (用於特徵工程)
            df['ma5_ratio'] = df['close'] / df['ma5']
            df['ma10_ratio'] = df['close'] / df['ma10']
            
        except Exception as e:
            logger.error(f"計算技術指標失敗: {e}")
            
        return df
    
    def update_all_stocks(self) -> Dict:
        """更新所有股票池的數據 (用於定時任務)"""
        logger.info("🔄 開始更新所有股票數據...")
        
        stock_pool = self.get_stock_pool()
        results = self.fetch_latest_prices(stock_pool, force_update=True)
        
        summary = {
            'total_stocks': len(stock_pool),
            'successful_updates': len(results),
            'failed_updates': len(stock_pool) - len(results),
            'update_time': datetime.now().isoformat(),
            'success_rate': len(results) / len(stock_pool) * 100 if stock_pool else 0
        }
        
        logger.info(f"📊 更新完成: {summary['successful_updates']}/{summary['total_stocks']} 成功 ({summary['success_rate']:.1f}%)")
        
        return {
            'summary': summary,
            'data': results
        }

# 使用範例和測試
if __name__ == "__main__":
    print("🚀 統一數據管理系統測試")
    
    # 初始化管理器
    dm = UnifiedDataManager()
    
    # 測試獲取少量股票數據
    test_symbols = ['0050', '0056', '2330', '00927']
    print(f"\n📊 測試獲取 {len(test_symbols)} 檔股票數據...")
    
    data = dm.fetch_latest_prices(test_symbols)
    
    print(f"\n✅ 成功獲取 {len(data)} 檔股票數據:")
    for symbol, info in data.items():
        print(f"   {symbol}: ${info['current_price']} ({info['change']:+.2f}, {info['change_pct']:+.2f}%) - 1張${info['one_lot_cost']:,.0f}")
    
    # 測試獲取訓練數據
    print(f"\n📈 測試獲取 0050 的訓練數據...")
    training_data = dm.get_training_data('0050', days=30)
    print(f"   獲取到 {len(training_data)} 天的數據")
    if not training_data.empty:
        print(f"   最新價格: ${training_data['close'].iloc[-1]:.2f}")
        print(f"   RSI: {training_data['rsi'].iloc[-1]:.1f}")