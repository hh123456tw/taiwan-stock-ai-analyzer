"""
預測追蹤驗證系統
記錄所有預測結果並驗證準確性
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PredictionTracker:
    """預測追蹤器 - 記錄和驗證所有預測"""
    
    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化資料庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 創建預測記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                current_price REAL NOT NULL,
                predicted_change_pct REAL NOT NULL,
                confidence_score REAL,
                model_type TEXT,
                target_date TEXT NOT NULL,
                actual_price REAL,
                actual_change_pct REAL,
                accuracy_score REAL,
                is_verified BOOLEAN DEFAULT FALSE,
                verification_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 創建準確率統計表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0.0,
                avg_error_pct REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, model_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_prediction(self, symbol: str, prediction_data: Dict) -> int:
        """記錄新預測"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_date = (datetime.now() + timedelta(days=prediction_data.get('days_ahead', 3))).strftime("%Y-%m-%d")
        
        cursor.execute("""
            INSERT INTO predictions (
                symbol, prediction_date, prediction_type, predicted_price, 
                current_price, predicted_change_pct, confidence_score, 
                model_type, target_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            prediction_date,
            prediction_data.get('type', 'price'),
            prediction_data.get('predicted_price', 0),
            prediction_data.get('current_price', 0),
            prediction_data.get('predicted_change_pct', 0),
            prediction_data.get('confidence_score', 0.5),
            prediction_data.get('model_type', 'unknown'),
            target_date
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"記錄預測: {symbol} ID={prediction_id}")
        return prediction_id
    
    def verify_predictions(self, symbol: str, current_price: float) -> List[Dict]:
        """驗證到期預測"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 查找需要驗證的預測
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE symbol = ? AND target_date <= ? AND is_verified = FALSE
        """, (symbol, today))
        
        unverified = cursor.fetchall()
        verified_results = []
        
        for row in unverified:
            prediction_id = row[0]
            predicted_price = row[4]
            original_price = row[5]
            predicted_change_pct = row[6]
            
            # 計算實際變化
            actual_change_pct = (current_price - original_price) / original_price * 100
            
            # 計算準確率
            price_error = abs(current_price - predicted_price) / original_price * 100
            direction_correct = (predicted_change_pct > 0) == (actual_change_pct > 0)
            
            # 準確率評分 (0-1)
            accuracy_score = self._calculate_accuracy_score(
                predicted_change_pct, actual_change_pct, price_error
            )
            
            # 更新預測記錄
            cursor.execute("""
                UPDATE predictions 
                SET actual_price = ?, actual_change_pct = ?, accuracy_score = ?, 
                    is_verified = TRUE, verification_date = ?
                WHERE id = ?
            """, (
                current_price, actual_change_pct, accuracy_score, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prediction_id
            ))
            
            verified_results.append({
                'prediction_id': prediction_id,
                'predicted_change': predicted_change_pct,
                'actual_change': actual_change_pct,
                'accuracy_score': accuracy_score,
                'direction_correct': direction_correct,
                'price_error_pct': price_error
            })
        
        conn.commit()
        conn.close()
        
        # 更新準確率統計
        if verified_results:
            self._update_accuracy_stats(symbol, verified_results)
        
        return verified_results
    
    def _calculate_accuracy_score(self, predicted_pct: float, actual_pct: float, price_error: float) -> float:
        """計算準確率評分"""
        # 方향正確性 (40%)
        direction_score = 1.0 if (predicted_pct > 0) == (actual_pct > 0) else 0.0
        
        # 幅度準確性 (60%)
        magnitude_error = abs(predicted_pct - actual_pct)
        magnitude_score = max(0, 1 - magnitude_error / 10)  # 10%以內誤差得滿分
        
        # 價格誤差懲罰
        price_penalty = min(price_error / 5, 0.2)  # 最多扣20%
        
        return max(0, direction_score * 0.4 + magnitude_score * 0.6 - price_penalty)
    
    def _update_accuracy_stats(self, symbol: str, results: List[Dict]):
        """更新準確率統計"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 按模型類型分組統計
        model_stats = {}
        for result in results:
            cursor.execute("""
                SELECT model_type FROM predictions WHERE id = ?
            """, (result['prediction_id'],))
            model_type = cursor.fetchone()[0]
            
            if model_type not in model_stats:
                model_stats[model_type] = []
            model_stats[model_type].append(result)
        
        # 更新各模型統計
        for model_type, model_results in model_stats.items():
            total_new = len(model_results)
            correct_new = sum(1 for r in model_results if r['accuracy_score'] > 0.6)
            avg_accuracy = sum(r['accuracy_score'] for r in model_results) / total_new
            
            cursor.execute("""
                INSERT OR REPLACE INTO accuracy_stats (
                    symbol, model_type, total_predictions, correct_predictions, 
                    accuracy_rate, avg_error_pct, last_updated
                ) VALUES (
                    ?, ?, 
                    COALESCE((SELECT total_predictions FROM accuracy_stats WHERE symbol = ? AND model_type = ?), 0) + ?,
                    COALESCE((SELECT correct_predictions FROM accuracy_stats WHERE symbol = ? AND model_type = ?), 0) + ?,
                    (COALESCE((SELECT correct_predictions FROM accuracy_stats WHERE symbol = ? AND model_type = ?), 0) + ?) * 1.0 / 
                    (COALESCE((SELECT total_predictions FROM accuracy_stats WHERE symbol = ? AND model_type = ?), 0) + ?),
                    ?, ?
                )
            """, (
                symbol, model_type, symbol, model_type, total_new, 
                symbol, model_type, correct_new, symbol, model_type, correct_new,
                symbol, model_type, total_new, avg_accuracy, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        conn.commit()
        conn.close()
    
    def get_model_performance(self, symbol: str, days: int = 30) -> Dict:
        """獲取模型表現統計"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # 獲取各模型統計
        cursor.execute("""
            SELECT model_type, 
                   COUNT(*) as total,
                   AVG(accuracy_score) as avg_accuracy,
                   SUM(CASE WHEN accuracy_score > 0.6 THEN 1 ELSE 0 END) as correct_count,
                   AVG(ABS(predicted_change_pct - actual_change_pct)) as avg_error
            FROM predictions 
            WHERE symbol = ? AND is_verified = TRUE AND prediction_date >= ?
            GROUP BY model_type
        """, (symbol, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        performance = {}
        for row in results:
            model_type, total, avg_accuracy, correct, avg_error = row
            performance[model_type] = {
                'total_predictions': total,
                'accuracy_rate': correct / total if total > 0 else 0,
                'avg_accuracy_score': avg_accuracy or 0,
                'avg_error_pct': avg_error or 0
            }
        
        return performance
    
    def get_recent_predictions(self, symbol: str, limit: int = 10) -> List[Dict]:
        """獲取最近預測記錄"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT prediction_date, model_type, predicted_change_pct, 
                   actual_change_pct, accuracy_score, is_verified
            FROM predictions 
            WHERE symbol = ? 
            ORDER BY prediction_date DESC 
            LIMIT ?
        """, (symbol, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'date': row[0],
                'model': row[1],
                'predicted': row[2],
                'actual': row[3],
                'accuracy': row[4],
                'verified': bool(row[5])
            })
        
        conn.close()
        return results
    
    def generate_performance_report(self, symbol: str) -> str:
        """生成表現報告"""
        performance = self.get_model_performance(symbol)
        recent = self.get_recent_predictions(symbol, 5)
        
        if not performance:
            return f"📊 {symbol} 暫無驗證數據"
        
        report = f"📈 **{symbol} 預測表現回顧**\n\n"
        
        # 各模型表現
        report += "🤖 **各模型準確率**:\n"
        for model, stats in performance.items():
            accuracy = stats['accuracy_rate'] * 100
            avg_error = stats['avg_error_pct']
            total = stats['total_predictions']
            
            report += f"• {model.upper()}: {accuracy:.1f}% ({total}次預測, 平均誤差{avg_error:.1f}%)\n"
        
        # 最佳模型
        best_model = max(performance.items(), key=lambda x: x[1]['accuracy_rate'])
        report += f"\n🏆 **最佳模型**: {best_model[0].upper()} (準確率{best_model[1]['accuracy_rate']*100:.1f}%)\n"
        
        # 最近預測回顧
        if recent:
            report += f"\n📋 **最近5次預測**:\n"
            for pred in recent:
                status = "✅已驗證" if pred['verified'] else "⏳待驗證"
                if pred['verified']:
                    accuracy_icon = "🎯" if pred['accuracy'] > 0.7 else "⚠️" if pred['accuracy'] > 0.4 else "❌"
                    report += f"• {pred['date'][:10]} {pred['model']}: 預測{pred['predicted']:+.1f}% 實際{pred['actual']:+.1f}% {accuracy_icon}\n"
                else:
                    report += f"• {pred['date'][:10]} {pred['model']}: 預測{pred['predicted']:+.1f}% {status}\n"
        
        return report