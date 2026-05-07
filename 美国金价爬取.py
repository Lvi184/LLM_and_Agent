import requests

def get_usd_gold_price():
    # 美元金价接口 URL
    url = "https://data-asg.goldprice.org/dbXRates/USD"
    
    # 模拟浏览器请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://goldprice.org/",
        "Origin": "https://goldprice.org"
    }
    
    try:
        # 发送 GET 请求
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # 解析 JSON 数据
        data = response.json()
        
        # 提取核心信息
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            print("实时美国金价数据（美元计价）：")
            print(f"黄金价格（盎司）: {item['xauPrice']} USD")
            print(f"白银价格（盎司）: {item['xagPrice']} USD")
            print(f"黄金涨跌额: {item['chgXau']} USD")
            print(f"黄金涨跌幅: {item['pcXau']} %")
            print(f"黄金收盘价: {item['xauClose']} USD")
            print(f"数据更新时间: {data['date']}")
        else:
            print("未获取到有效数据")
    
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    except (KeyError, IndexError) as e:
        print(f"数据解析错误: {e}")

if __name__ == "__main__":
    get_usd_gold_price()