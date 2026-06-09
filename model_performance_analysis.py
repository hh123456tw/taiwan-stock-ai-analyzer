#!/usr/bin/env python3
"""
AI模型性能分析 - 檢查訓練問題和預測效果
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def analyze_model_performance():
    """分析AI模型性能"""
    print("🤖 AI模型性能分析")
    print("=" * 60)
    
    # 連接數據庫
    db_path = 'predictions.db'
    conn = sqlite3.connect(db_path)
    
    try:
        # 1. 總體統計
        print("📊 總體統計")
        print("-" * 40)
        
        total_predictions = pd.read_sql_query(
            "SELECT COUNT(*) as total FROM predictions", conn
        ).iloc[0]['total']
        
        verified_predictions = pd.read_sql_query(
            "SELECT COUNT(*) as verified FROM predictions WHERE is_verified = 1", conn
        ).iloc[0]['verified']
        
        avg_accuracy = pd.read_sql_query(
            "SELECT AVG(accuracy_score) as avg_acc FROM predictions WHERE accuracy_score IS NOT NULL", conn
        ).iloc[0]['avg_acc']
        
        print(f"總預測數: {total_predictions}")
        print(f"已驗證預測: {verified_predictions}")
        print(f"平均準確率: {avg_accuracy:.2%}" if avg_accuracy else "平均準確率: N/A")
        
        # 2. 模型性能比較
        print(f"\n🏆 模型性能排行")
        print("-" * 40)
        
        model_performance = pd.read_sql_query("""
            SELECT 
                model_name,
                COUNT(*) as predictions_count,
                AVG(confidence) as avg_confidence,
                AVG(accuracy_score) as avg_accuracy,
                MIN(accuracy_score) as min_accuracy,
                MAX(accuracy_score) as max_accuracy
            FROM predictions 
            WHERE accuracy_score IS NOT NULL 
            GROUP BY model_name 
            ORDER BY avg_accuracy DESC
        """, conn)
        
        for idx, row in model_performance.iterrows():
            print(f"{idx+1}. {row['model_name']}")
            print(f"   預測數: {row['predictions_count']}")
            print(f"   平均信心度: {row['avg_confidence']:.3f}")
            print(f"   平均準確率: {row['avg_accuracy']:.2%}")
            print(f"   準確率範圍: {row['min_accuracy']:.2%} - {row['max_accuracy']:.2%}")
            print()
        
        # 3. 股票預測效果
        print(f"📈 股票預測效果排行")
        print("-" * 40)
        
        stock_performance = pd.read_sql_query("""
            SELECT 
                symbol,
                COUNT(*) as predictions_count,
                AVG(confidence) as avg_confidence,
                AVG(accuracy_score) as avg_accuracy,
                AVG(ABS(price_error_pct)) as avg_error_pct
            FROM predictions 
            WHERE accuracy_score IS NOT NULL 
            GROUP BY symbol 
            HAVING predictions_count >= 2
            ORDER BY avg_accuracy DESC
        """, conn)
        
        for idx, row in stock_performance.iterrows():
            print(f"{idx+1}. {row['symbol']}")
            print(f"   預測數: {row['predictions_count']}")
            print(f"   平均準確率: {row['avg_accuracy']:.2%}")
            print(f"   平均誤差: {row['avg_error_pct']:.2%}")
            print()
        
        # 4. 時間趨勢分析
        print(f"📅 預測時間趨勢")
        print("-" * 40)
        
        time_trend = pd.read_sql_query("""
            SELECT 
                DATE(prediction_date) as pred_date,
                COUNT(*) as daily_predictions,
                AVG(confidence) as avg_confidence,
                AVG(accuracy_score) as avg_accuracy
            FROM predictions 
            WHERE accuracy_score IS NOT NULL 
            GROUP BY DATE(prediction_date) 
            ORDER BY pred_date DESC
            LIMIT 10
        """, conn)
        
        for idx, row in time_trend.iterrows():
            print(f"{row['pred_date']}: {row['daily_predictions']}筆預測, "
                  f"準確率{row['avg_accuracy']:.2%}")
        
        # 5. 問題識別
        print(f"\n⚠️ 潛在問題識別")
        print("-" * 40)
        
        # 低準確率預測
        low_accuracy = pd.read_sql_query("""
            SELECT symbol, model_name, confidence, accuracy_score, prediction_date
            FROM predictions 
            WHERE accuracy_score < 0.8 AND accuracy_score IS NOT NULL
            ORDER BY accuracy_score ASC
            LIMIT 5
        """, conn)
        
        if not low_accuracy.empty:
            print("🔴 低準確率預測 (< 80%):")
            for idx, row in low_accuracy.iterrows():
                print(f"   {row['symbol']} ({row['model_name']}): {row['accuracy_score']:.2%} "
                      f"信心度{row['confidence']:.3f}")
        
        # 高誤差預測
        high_error = pd.read_sql_query("""
            SELECT symbol, model_name, confidence, ABS(price_error_pct) as abs_error
            FROM predictions 
            WHERE ABS(price_error_pct) > 10 AND price_error_pct IS NOT NULL
            ORDER BY abs_error DESC
            LIMIT 5
        """, conn)
        
        if not high_error.empty:
            print("\n🔴 高誤差預測 (> 10%):")
            for idx, row in high_error.iterrows():
                print(f"   {row['symbol']} ({row['model_name']}): {row['abs_error']:.2%}誤差 "
                      f"信心度{row['confidence']:.3f}")
        
        # 6. 建議改進
        print(f"\n💡 改進建議")
        print("-" * 40)
        
        suggestions = []
        
        # 檢查模型多樣性
        model_count = len(model_performance)
        if model_count < 3:
            suggestions.append("增加更多模型類型以提高預測多樣性")
        
        # 檢查整體準確率
        if avg_accuracy and avg_accuracy < 0.9:
            suggestions.append("整體準確率偏低，建議優化模型參數或特徵工程")
        
        # 檢查預測數量分布
        uneven_distribution = stock_performance['predictions_count'].std() > 5
        if uneven_distribution:
            suggestions.append("股票預測數量分布不均，建議平衡各股票的預測頻率")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        else:
            print("✅ 模型性能良好，無明顯問題")
        
        return {
            'total_predictions': total_predictions,
            'verified_predictions': verified_predictions,
            'avg_accuracy': avg_accuracy,
            'model_performance': model_performance,
            'stock_performance': stock_performance,
            'suggestions': suggestions
        }
        
    except Exception as e:
        print(f"❌ 分析過程出錯: {e}")
        return None
        
    finally:
        conn.close()

def check_training_issues():
    """檢查訓練相關問題"""
    print(f"\n🔧 訓練問題檢查")
    print("=" * 60)
    
    issues = []
    
    # 1. 檢查模型文件
    model_dirs = ['models/lstm', 'models/rf', 'models/xgb', 'models/arima', 'models/lightgbm']
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            model_files = os.listdir(model_dir)
            if not model_files:
                issues.append(f"{model_dir} 目錄為空，需要重新訓練")
            else:
                print(f"✅ {model_dir}: {len(model_files)} 個模型文件")
        else:
            issues.append(f"{model_dir} 目錄不存在")
    
    # 2. 檢查數據庫狀態
    if os.path.exists('predictions.db'):
        print("✅ predictions.db 存在")
        
        # 檢查數據新鮮度
        conn = sqlite3.connect('predictions.db')
        try:
            latest_prediction = pd.read_sql_query(
                "SELECT MAX(prediction_date) as latest FROM predictions", conn
            ).iloc[0]['latest']
            
            if latest_prediction:
                latest_date = datetime.strptime(latest_prediction, '%Y-%m-%d')
                days_old = (datetime.now() - latest_date).days
                
                if days_old > 7:
                    issues.append(f"最新預測距今 {days_old} 天，建議更新預測")
                else:
                    print(f"✅ 預測數據較新 (最新: {latest_prediction})")
            else:
                issues.append("數據庫中無預測記錄")
                
        except Exception as e:
            issues.append(f"數據庫查詢錯誤: {e}")
        finally:
            conn.close()
    else:
        issues.append("predictions.db 不存在，需要初始化數據庫")
    
    # 3. 檢查訓練腳本
    training_scripts = [
        'batch_model_trainer.py',
        'complete_training_cycle.py', 
        'auto_stock_training_system.py'
    ]
    
    for script in training_scripts:
        if os.path.exists(script):
            print(f"✅ {script} 存在")
        else:
            issues.append(f"{script} 訓練腳本缺失")
    
    # 輸出問題總結
    if issues:
        print(f"\n⚠️ 發現問題:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print(f"\n✅ 訓練環境正常")
    
    return issues

def generate_performance_report():
    """生成性能報告"""
    print(f"\n📋 生成性能報告")
    print("=" * 60)
    
    # 分析模型性能
    performance_data = analyze_model_performance()
    
    # 檢查訓練問題
    training_issues = check_training_issues()
    
    # 生成報告文件
    report_content = f"""# AI模型性能分析報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 模型性能總結
- 總預測數: {performance_data['total_predictions'] if performance_data else 'N/A'}
- 已驗證預測: {performance_data['verified_predictions'] if performance_data else 'N/A'}
- 平均準確率: {performance_data['avg_accuracy']:.2%} if performance_data and performance_data['avg_accuracy'] else 'N/A'

## 訓練問題
{chr(10).join(f"- {issue}" for issue in training_issues) if training_issues else "無明顯問題"}

## 改進建議
{chr(10).join(f"- {suggestion}" for suggestion in performance_data['suggestions']) if performance_data and performance_data['suggestions'] else "模型性能良好"}
"""
    
    with open('model_performance_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("✅ 報告已生成: model_performance_report.md")
    return report_content

if __name__ == "__main__":
    try:
        # 執行完整分析
        analyze_model_performance()
        check_training_issues()
        generate_performance_report()
        
        print(f"\n🎯 性能分析完成!")
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()