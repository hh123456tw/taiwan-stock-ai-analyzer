#!/usr/bin/env python3
"""
預測驗證器 - 抓取實際股價來驗證預測準確率
"""

import sqlite3
import pandas as pd
import yfinance as yf
import twstock
from datetime import datetime, timedelta
import numpy as np

class PredictionValidator:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.db')
        self.db_path = db_path
        
    def get_actual_price(self, symbol, date):
        """獲取指定日期的實際股價"""
        try:
            # 嘗試 yfinance (台股)
            ticker = f"{symbol}.TW"
            end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=2)
            start_date = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=5)
            
            stock_data = yf.download(ticker, 
                                   start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d'),
                                   progress=False)
            
            if len(stock_data) > 0:
                # 取最後一個交易日的收盤價
                return float(stock_data['Close'].iloc[-1])
                
        except Exception as e:
            print(f"yfinance 失敗 {symbol}: {e}")
        
        # 備用：使用 twstock
        try:
            stock = twstock.Stock(symbol)
            # 獲取最近的資料
            recent_data = stock.fetch_from(2025, 6)  # 六月資料
            if recent_data:
                # 找到最接近的日期
                target_date = datetime.strptime(date, '%Y-%m-%d')
                for data_point in reversed(recent_data):
                    data_date = data_point.date
                    if data_date <= target_date:
                        return data_point.close
                        
        except Exception as e:
            print(f"twstock 失敗 {symbol}: {e}")
            
        return None
    
    def validate_predictions(self, days_back=7):
        """驗證最近幾天的預測"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 獲取最近的預測
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT id, symbol, prediction_date, predicted_price, model_name, confidence
            FROM predictions 
            WHERE prediction_date >= ?
            ORDER BY prediction_date DESC
            """
            
            predictions_df = pd.read_sql_query(query, conn, params=[cutoff_date])
            conn.close()
            
            if len(predictions_df) == 0:
                print("⚠️ 沒有找到最近的預測資料")
                return None
            
            print(f"🔍 開始驗證 {len(predictions_df)} 筆預測...")
            
            validation_results = []
            
            for _, pred in predictions_df.iterrows():
                print(f"📊 驗證 {pred['symbol']} ({pred['prediction_date']})")
                
                actual_price = self.get_actual_price(pred['symbol'], pred['prediction_date'])
                
                if actual_price is not None:
                    # 計算誤差
                    predicted_price = pred['predicted_price']
                    price_error = abs(actual_price - predicted_price)
                    price_error_pct = (price_error / actual_price) * 100
                    
                    # 計算準確率分數 (誤差越小分數越高)
                    accuracy_score = max(0, 1 - (price_error_pct / 10))  # 10%誤差內給分
                    
                    validation_results.append({
                        'id': pred['id'],
                        'symbol': pred['symbol'],
                        'prediction_date': pred['prediction_date'],
                        'predicted_price': predicted_price,
                        'actual_price': actual_price,
                        'price_error': price_error,
                        'price_error_pct': price_error_pct,
                        'accuracy_score': accuracy_score,
                        'model_name': pred['model_name'],
                        'confidence': pred['confidence']
                    })
                    
                    print(f"  ✅ 預測: {predicted_price:.2f}, 實際: {actual_price:.2f}, 誤差: {price_error_pct:.2f}%, 分數: {accuracy_score:.3f}")
                else:
                    print(f"  ❌ 無法獲取 {pred['symbol']} 的實際價格")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ 驗證失敗: {e}")
            return None
    
    def generate_accuracy_report(self, validation_results):
        """生成準確率報告"""
        if not validation_results:
            return "❌ 沒有驗證結果可以分析"
        
        df = pd.DataFrame(validation_results)
        
        report_lines = []
        report_lines.append("📈 **預測準確率驗證報告**")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # 整體統計
        avg_accuracy = df['accuracy_score'].mean()
        avg_error_pct = df['price_error_pct'].mean()
        best_prediction = df.loc[df['accuracy_score'].idxmax()]
        worst_prediction = df.loc[df['accuracy_score'].idxmin()]
        
        report_lines.append("🎯 **整體表現**")
        report_lines.append(f"• 總驗證次數: {len(df)}")
        report_lines.append(f"• 平均準確率分數: {avg_accuracy:.3f}")
        report_lines.append(f"• 平均價格誤差: {avg_error_pct:.2f}%")
        report_lines.append(f"• 優秀預測 (>0.7分): {len(df[df['accuracy_score'] > 0.7])} 筆")
        report_lines.append(f"• 及格預測 (>0.5分): {len(df[df['accuracy_score'] > 0.5])} 筆")
        report_lines.append("")
        
        # 按股票統計
        stock_stats = df.groupby('symbol').agg({
            'accuracy_score': 'mean',
            'price_error_pct': 'mean',
            'predicted_price': 'count'
        }).round(3)
        stock_stats.columns = ['平均準確率', '平均誤差%', '預測次數']
        
        report_lines.append("📊 **各股票表現**")
        for symbol, stats in stock_stats.iterrows():
            report_lines.append(f"• {symbol}: 準確率 {stats['平均準確率']:.3f}, 誤差 {stats['平均誤差%']:.2f}%, {stats['預測次數']}次預測")
        report_lines.append("")
        
        # 按模型統計
        if df['model_name'].nunique() > 1:
            model_stats = df.groupby('model_name').agg({
                'accuracy_score': 'mean',
                'price_error_pct': 'mean',
                'predicted_price': 'count'
            }).round(3)
            model_stats.columns = ['平均準確率', '平均誤差%', '預測次數']
            
            report_lines.append("🤖 **各模型表現**")
            for model, stats in model_stats.iterrows():
                report_lines.append(f"• {model}: 準確率 {stats['平均準確率']:.3f}, 誤差 {stats['平均誤差%']:.2f}%, {stats['預測次數']}次預測")
            report_lines.append("")
        
        # 最佳和最差預測
        report_lines.append("🏆 **最佳預測**")
        report_lines.append(f"• {best_prediction['symbol']} ({best_prediction['prediction_date']})")
        report_lines.append(f"• 預測: {best_prediction['predicted_price']:.2f}, 實際: {best_prediction['actual_price']:.2f}")
        report_lines.append(f"• 誤差: {best_prediction['price_error_pct']:.2f}%, 分數: {best_prediction['accuracy_score']:.3f}")
        report_lines.append("")
        
        report_lines.append("⚠️ **最差預測**")
        report_lines.append(f"• {worst_prediction['symbol']} ({worst_prediction['prediction_date']})")
        report_lines.append(f"• 預測: {worst_prediction['predicted_price']:.2f}, 實際: {worst_prediction['actual_price']:.2f}")
        report_lines.append(f"• 誤差: {worst_prediction['price_error_pct']:.2f}%, 分數: {worst_prediction['accuracy_score']:.3f}")
        report_lines.append("")
        
        # 詳細列表
        report_lines.append("📋 **詳細驗證結果**")
        for _, row in df.iterrows():
            status = "✅" if row['accuracy_score'] > 0.7 else "⚠️" if row['accuracy_score'] > 0.5 else "❌"
            report_lines.append(f"{status} {row['symbol']} ({row['prediction_date']}): 預測 {row['predicted_price']:.2f} → 實際 {row['actual_price']:.2f} (誤差 {row['price_error_pct']:.2f}%)")
        
        report_lines.append("")
        report_lines.append(f"📅 **報告生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report_lines)
    
    def update_database_with_actual_prices(self, validation_results):
        """將實際價格更新到資料庫"""
        if not validation_results:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 檢查是否需要新增欄位
            cursor.execute("PRAGMA table_info(predictions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'actual_price' not in columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN actual_price REAL")
            if 'price_error_pct' not in columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN price_error_pct REAL")
            if 'accuracy_score' not in columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN accuracy_score REAL")
            if 'is_verified' not in columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN is_verified BOOLEAN DEFAULT 0")
            
            # 更新驗證結果
            for result in validation_results:
                cursor.execute("""
                UPDATE predictions 
                SET actual_price = ?, price_error_pct = ?, accuracy_score = ?, is_verified = 1
                WHERE id = ?
                """, (
                    result['actual_price'],
                    result['price_error_pct'], 
                    result['accuracy_score'],
                    result['id']
                ))
            
            conn.commit()
            conn.close()
            print(f"✅ 已更新 {len(validation_results)} 筆驗證結果到資料庫")
            
        except Exception as e:
            print(f"❌ 更新資料庫失敗: {e}")

def main():
    """主執行函數"""
    print("🔍 啟動預測驗證器")
    
    validator = PredictionValidator()  # 使用預設的開發環境路徑
    
    # 驗證最近7天的預測
    print("📊 開始驗證最近的預測...")
    validation_results = validator.validate_predictions(days_back=7)
    
    if validation_results:
        # 生成報告
        report = validator.generate_accuracy_report(validation_results)
        
        # 保存報告
        import os
        os.makedirs("./results/backtest", exist_ok=True)
        report_file = "./results/backtest/prediction_validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 驗證報告已保存: {report_file}")
        print("\n" + "="*50)
        print(report)
        
        # 更新資料庫
        validator.update_database_with_actual_prices(validation_results)
    else:
        print("⚠️ 沒有可驗證的預測結果")

if __name__ == "__main__":
    main()