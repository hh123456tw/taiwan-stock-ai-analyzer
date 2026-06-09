#\!/usr/bin/env python3
"""
AI策略分析器 - 深度分析AI預測的交易策略表現
包含多維度績效分析、風險評估和策略優化建議
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class StrategyAnalyzer:
    def __init__(self):
        self.db_path = '/mnt/nvme/taiwan-stock-ai-analyzer-main/predictions.db'
        
    def analyze_model_performance(self, days_back: int = 30) -> Dict:
        """分析各AI模型預測表現"""
        conn = sqlite3.connect(self.db_path)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
        SELECT model_type, predicted_direction, 
               CASE 
                   WHEN actual_price > predicted_price AND predicted_direction = 'UP' THEN 1
                   WHEN actual_price < predicted_price AND predicted_direction = 'DOWN' THEN 1
                   ELSE 0
               END as correct_prediction,
               confidence_score,
               ABS(actual_price - predicted_price) / predicted_price as price_error
        FROM predictions 
        WHERE prediction_date BETWEEN ? AND ?
        AND actual_price IS NOT NULL
        AND predicted_price IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'), 
                                                   end_date.strftime('%Y-%m-%d')))
        conn.close()
        
        if df.empty:
            return {}
            
        # 按模型分組分析
        model_stats = {}
        for model in df['model_type'].unique():
            model_data = df[df['model_type'] == model]
            
            model_stats[model] = {
                'total_predictions': len(model_data),
                'accuracy': model_data['correct_prediction'].mean(),
                'avg_confidence': model_data['confidence_score'].mean(),
                'avg_price_error': model_data['price_error'].mean(),
                'up_predictions': len(model_data[model_data['predicted_direction'] == 'UP']),
                'down_predictions': len(model_data[model_data['predicted_direction'] == 'DOWN'])
            }
            
        return model_stats
        
    def analyze_confidence_impact(self, days_back: int = 30) -> Dict:
        """分析信心度對預測準確率的影響"""
        conn = sqlite3.connect(self.db_path)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
        SELECT confidence_score,
               CASE 
                   WHEN actual_price > predicted_price AND predicted_direction = 'UP' THEN 1
                   WHEN actual_price < predicted_price AND predicted_direction = 'DOWN' THEN 1
                   ELSE 0
               END as correct_prediction
        FROM predictions 
        WHERE prediction_date BETWEEN ? AND ?
        AND actual_price IS NOT NULL
        AND predicted_price IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'), 
                                                   end_date.strftime('%Y-%m-%d')))
        conn.close()
        
        if df.empty:
            return {}
            
        # 按信心度區間分析
        confidence_ranges = [
            (0.5, 0.6, 'Low'),
            (0.6, 0.7, 'Medium'),
            (0.7, 0.8, 'High'),
            (0.8, 1.0, 'Very High')
        ]
        
        confidence_stats = {}
        for min_conf, max_conf, label in confidence_ranges:
            range_data = df[(df['confidence_score'] >= min_conf) & 
                           (df['confidence_score'] < max_conf)]
            
            if len(range_data) > 0:
                confidence_stats[label] = {
                    'range': f'{min_conf:.1f}-{max_conf:.1f}',
                    'count': len(range_data),
                    'accuracy': range_data['correct_prediction'].mean(),
                    'avg_confidence': range_data['confidence_score'].mean()
                }
                
        return confidence_stats
        
    def analyze_symbol_performance(self, days_back: int = 30) -> Dict:
        """分析各股票的預測表現"""
        conn = sqlite3.connect(self.db_path)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
        SELECT symbol,
               COUNT(*) as prediction_count,
               AVG(CASE 
                   WHEN actual_price > predicted_price AND predicted_direction = 'UP' THEN 1
                   WHEN actual_price < predicted_price AND predicted_direction = 'DOWN' THEN 1
                   ELSE 0
               END) as accuracy,
               AVG(confidence_score) as avg_confidence,
               AVG(ABS(actual_price - predicted_price) / predicted_price) as avg_price_error
        FROM predictions 
        WHERE prediction_date BETWEEN ? AND ?
        AND actual_price IS NOT NULL
        AND predicted_price IS NOT NULL
        GROUP BY symbol
        HAVING COUNT(*) >= 3
        ORDER BY accuracy DESC, prediction_count DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'), 
                                                   end_date.strftime('%Y-%m-%d')))
        conn.close()
        
        return df.to_dict('records') if not df.empty else []
        
    def generate_strategy_recommendations(self) -> Dict:
        """生成策略建議"""
        model_perf = self.analyze_model_performance()
        confidence_impact = self.analyze_confidence_impact()
        symbol_perf = self.analyze_symbol_performance()
        
        recommendations = {
            'best_models': [],
            'optimal_confidence_threshold': 0.6,
            'recommended_symbols': [],
            'strategy_suggestions': []
        }
        
        # 找出最佳模型
        if model_perf:
            sorted_models = sorted(model_perf.items(), 
                                 key=lambda x: x[1]['accuracy'], reverse=True)
            recommendations['best_models'] = [model for model, stats in sorted_models 
                                            if stats['accuracy'] > 0.6][:3]
        
        # 找出最佳信心度閾值
        if confidence_impact:
            best_confidence = max(confidence_impact.items(), 
                                key=lambda x: x[1]['accuracy'])
            recommendations['optimal_confidence_threshold'] = best_confidence[1]['avg_confidence']
        
        # 推薦表現好的股票
        if symbol_perf:
            recommendations['recommended_symbols'] = [s['symbol'] for s in symbol_perf[:10]
                                                    if s['accuracy'] > 0.65]
        
        # 策略建議
        recommendations['strategy_suggestions'] = [
            f"優先使用模型: {', '.join(recommendations['best_models'][:2])}",
            f"建議信心度閾值: {recommendations['optimal_confidence_threshold']:.2f}以上",
            f"重點關注股票: {', '.join(recommendations['recommended_symbols'][:5])}",
            "建議每次投入資金不超過總資金的15%",
            "設定停損點為-8%，停利點為+15%"
        ]
        
        return recommendations
        
    def generate_comprehensive_report(self, save_path: str = None) -> str:
        """生成綜合分析報告"""
        print("開始生成AI策略分析報告...")
        
        model_perf = self.analyze_model_performance()
        confidence_impact = self.analyze_confidence_impact()
        symbol_perf = self.analyze_symbol_performance()
        recommendations = self.generate_strategy_recommendations()
        
        report = f"""
=== AI交易策略深度分析報告 ===
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析期間: 最近30天

【模型表現分析】
"""
        
        if model_perf:
            for model, stats in model_perf.items():
                report += f"""
{model} 模型:
  - 預測次數: {stats['total_predictions']}
  - 準確率: {stats['accuracy']:.1%}
  - 平均信心度: {stats['avg_confidence']:.2f}
  - 平均價格誤差: {stats['avg_price_error']:.1%}
  - 看漲預測: {stats['up_predictions']} | 看跌預測: {stats['down_predictions']}
"""
        
        report += "\n【信心度影響分析】\n"
        if confidence_impact:
            for range_name, stats in confidence_impact.items():
                report += f"""
{range_name} 信心度 ({stats['range']}):
  - 預測數量: {stats['count']}
  - 準確率: {stats['accuracy']:.1%}
  - 平均信心度: {stats['avg_confidence']:.2f}
"""
        
        report += "\n【表現優異股票 (Top 10)】\n"
        if symbol_perf:
            for i, stock in enumerate(symbol_perf[:10], 1):
                report += f"{i:2d}. {stock['symbol']} - 準確率: {stock['accuracy']:.1%} ({stock['prediction_count']}次預測)\n"
        
        report += "\n【策略建議】\n"
        for i, suggestion in enumerate(recommendations['strategy_suggestions'], 1):
            report += f"{i}. {suggestion}\n"
            
        report += f"""

【投資建議總結】
✓ 推薦使用模型: {', '.join(recommendations['best_models'][:2])}
✓ 最佳信心度門檻: {recommendations['optimal_confidence_threshold']:.2f}
✓ 重點股票池: {', '.join(recommendations['recommended_symbols'][:8])}

【風險提醒】
• 過去績效不代表未來表現
• 建議分散投資，單一標的勿超過總資金20%
• 務必設定停損停利點
• 定期檢視和調整策略參數
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
        return report

def main():
    """主執行函數"""
    analyzer = StrategyAnalyzer()
    
    # 生成綜合報告
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # 保存報告
    os.makedirs('/mnt/nvme/taiwan-stock-ai-analyzer-main/backtest/reports', exist_ok=True)
    report_path = f'/mnt/nvme/taiwan-stock-ai-analyzer-main/backtest/reports/strategy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    analyzer.generate_comprehensive_report(report_path)
    print(f"\n報告已保存至: {report_path}")

if __name__ == "__main__":
    main()
