import requests
from lxml import etree
import json
import pandas as pd

def parser():
    url1 = f'https://www.autohome.com.cn/rank/1-1-0-{start_p}_{end_p}-x-{etype}-x/{y}-{m}.html'
    response1 = requests.get(url=url1, headers=headers1)
    tree = etree.HTML(response1.text)
    # 检查 car_model 是否存在
    car_model_list = tree.xpath("//div[@class='tw-text-nowrap tw-text-lg tw-font-medium hover:tw-text-[#ff6600]']/text()")
    if not car_model_list:
        print(f"未找到 car_model，跳过 {url1}")
        return None  # 或者根据需求进行其他处理
    car_model = car_model_list[0]

    # 检查 sales 是否存在
    sales_list = tree.xpath("//span[@class='tw-pt-[5px]  tw-text-[22px] tw-font-[500] tw-leading-none']/text()")
    if not sales_list:
        print(f"未找到 sales，跳过 {url1}")
        return None  # 或者根据需求进行其他处理
    sales = sales_list[0]

    # 检查 price_r 是否存在
    price_r_list = tree.xpath("//div[@class=' tw-font-medium tw-text-[#717887]']/text()")
    if not price_r_list:
        print(f"未找到 price_r，跳过 {url1}")
        return None  # 或者根据需求进行其他处理
    price_r = price_r_list[0].replace('万', '').split('-')
    print(car_model)
    price = (float(price_r[0]) + float(price_r[1])) / 2

    # 检查 series_id 是否存在
    series_id_list = tree.xpath("//div[@data-rank-num='1']//button/@data-series-id")
    if not series_id_list:
        print(f"未找到 series_id，跳过 {url1}")
        return None  # 或者根据需求进行其他处理
    series_id = series_id_list[0]

    url2 = f'https://car-web-api.autohome.com.cn/car/param/getParamConf?mode=1&site=1&seriesid={series_id}'
    response2 = requests.get(url=url2, headers=headers1)
    res_text = json.loads(response2.text)
    clist = res_text['result']['datalist']
    i = 0
    sum_val = 0
    for data in clist:
        sum_val = float(data['paramconflist'][7]['itemname']) + sum_val
        i = i + 1
    range_val = sum_val / i

    # 创建 DataFrame
    data = {
        'car_model': [car_model],
        'month': [month],
        'price': [price],
        'sales': [sales],
        'range': [range_val],
        'power_type': [power_type],
        'brand_tier': [brand_tier]
    }
    df = pd.DataFrame(data)
    return df

headers1 = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
}
l = [[0, 10], [10, 30], [30, 9000]]
a = [4, 5, 6]
d = {
    '2024': ['09', '10', '11', '12'],
    '2025': ['01', '02']
}

all_dfs = []  # 用于存储所有的 DataFrame

for x in l:
    start_p = x[0]
    end_p = x[1]
    if start_p == 0:
        brand_tier = 1
    elif start_p == 10:
        brand_tier = 2
    else:
        brand_tier = 3
    for y in a:
        kind = ['BEV', 'PHEV', 'REEV']
        etype = y
        power_type = kind[etype - 4]
        for z in d:
            y = z
            for m in d[z]:  # 修正此处，应该遍历 d[z] 而不是 z
                month = f'{y}-{m}'
                df = parser()
                if df is not None:
                    all_dfs.append(df)

# 合并所有的 DataFrame
final_df = pd.concat(all_dfs, ignore_index=True)

# 保存为 CSV 文件
final_df.to_csv('car_info.csv', index=False)

print("数据已保存到 car_info.csv 文件中。")



