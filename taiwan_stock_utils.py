#\!/usr/bin/env python3
"""
台股交易基本函數庫
重要：這些函數基於台股 1張 = 1000股 的基本規則
"""

def taiwan_stock_affordability_check(stock_price, available_cash):
    """
    台股資金門檻檢查
    
    Args:
        stock_price (float): 每股價格
        available_cash (float): 可用資金
    
    Returns:
        dict: 包含是否買得起、成本、最大張數等資訊
    """
    cost_per_lot = stock_price * 1000  # 1張 = 1000股
    transaction_fee = cost_per_lot * 0.001425  # 手續費約0.1425%
    total_cost = cost_per_lot + transaction_fee
    
    return {
        'affordable': available_cash >= total_cost,
        'cost_per_lot': cost_per_lot,
        'max_lots': int(available_cash / cost_per_lot) if cost_per_lot > 0 else 0,
        'total_cost': total_cost,
        'stock_price': stock_price
    }

def select_affordable_stocks(stock_pool, available_cash, max_cost_ratio=0.3):
    """
    根據資金選擇買得起的股票
    
    Args:
        stock_pool (dict): {股票代號: 股價}
        available_cash (float): 可用資金
        max_cost_ratio (float): 單檔最大投資比例
    
    Returns:
        list: 買得起的股票清單
    """
    affordable_stocks = []
    max_single_investment = available_cash * max_cost_ratio
    
    for stock_code, stock_price in stock_pool.items():
        cost_per_lot = stock_price * 1000
        
        if cost_per_lot <= max_single_investment:
            affordable_stocks.append({
                'code': stock_code,
                'price': stock_price,
                'cost_per_lot': cost_per_lot,
                'max_lots': int(max_single_investment / cost_per_lot)
            })
    
    return sorted(affordable_stocks, key=lambda x: x['cost_per_lot'])

def categorize_stocks_by_price(stock_pool):
    """
    依價格門檻分類股票
    
    Returns:
        dict: 低、中、高價股分類
    """
    categories = {
        'low_price': [],    # <10萬/張
        'mid_price': [],    # 10-50萬/張  
        'high_price': []    # >50萬/張
    }
    
    for stock_code, stock_price in stock_pool.items():
        cost_per_lot = stock_price * 1000
        
        if cost_per_lot < 100000:
            categories['low_price'].append({
                'code': stock_code,
                'price': stock_price,
                'cost_per_lot': cost_per_lot
            })
        elif cost_per_lot < 500000:
            categories['mid_price'].append({
                'code': stock_code,
                'price': stock_price,
                'cost_per_lot': cost_per_lot
            })
        else:
            categories['high_price'].append({
                'code': stock_code,
                'price': stock_price,
                'cost_per_lot': cost_per_lot
            })
    
    return categories

def demo_taiwan_stock_basics():
    """示範台股基本觀念"""
    print('📚 台股基本交易觀念示範')
    print('=' * 40)
    
    # 示範股票
    demo_stocks = {
        '2330': 1140,  # 台積電
        '2454': 1350,  # 聯發科
        '0050': 51,    # 元大台灣50
        '2317': 181    # 鴻海
    }
    
    # 示範不同資金水準
    cash_levels = [100000, 500000, 1000000, 2000000]
    
    for cash in cash_levels:
        print(f'\n💰 可用資金: {cash:,}元')
        print('-' * 30)
        
        for code, price in demo_stocks.items():
            result = taiwan_stock_affordability_check(price, cash)
            status = '✅ 買得起' if result['affordable'] else '❌ 買不起'
            print(f'{code}: {price}元/股 → {result["cost_per_lot"]:,}元/張 {status}')
            if result['max_lots'] > 0:
                print(f'     最多可買: {result["max_lots"]}張')

if __name__ == '__main__':
    demo_taiwan_stock_basics()
