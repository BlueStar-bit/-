# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 数据准备与预处理
# --------------------------
# 读取真实数据集
df = pd.read_csv(r'.\car_info.csv')
# print(df)
# 数据预处理
# 价格平滑处理（3个月移动平均）
df['smoothed_price'] = df.groupby('car_model')['price'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# 创建销量滞后项（滞后1个月）
df['lag_sales'] = df.groupby('car_model')['sales'].shift(1)

# 对数转换
df['ln_sales'] = np.log(df['sales'] + 1)  # 防止零值
df['ln_price'] = np.log(df['smoothed_price'])

# 标准化处理（续航和品牌）
scaler = StandardScaler()
df[['scaled_range', 'scaled_brand']] = scaler.fit_transform(df[['range', 'brand_tier']])

# --------------------------
# 2. 基础线性回归模型
# --------------------------
# 添加常数项
X = df[['ln_price', 'scaled_range', 'scaled_brand']]
X = sm.add_constant(X)
y = df['ln_sales']

model = sm.OLS(y, X).fit(cov_type='HC1')  # 使用异方差稳健标准误差
print("基础线性回归结果:".center(50, '-'))
print(model.summary())

# 计算VIF检测多重共线性
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF检测:".center(50, '-'))
print(vif_data)

# --------------------------
# 3. 分群回归分析
# --------------------------
print("\n分群回归结果:".center(50, '-'))
# 新增：存储分群回归结果的字典
elasticity_by_power = {}

for name, group in df.groupby('power_type'):
    X_group = group[['ln_price', 'scaled_range']]
    X_group = sm.add_constant(X_group)
    model_group = sm.OLS(group['ln_sales'], X_group).fit()
    
    # 存储回归系数到字典
    elasticity_by_power[name] = model_group.params['ln_price']
    print(f"动力类型 {name} 回归结果:")
    print(f"价格弹性系数: {elasticity_by_power[name]:.3f} (p={model_group.pvalues['ln_price']:.3f})")

# --------------------------
# 5. 策略建议生成
# --------------------------
# 删除原有错误的分组计算代码（约第150-180行）

# 直接使用分群回归的弹性系数
strategy_table = pd.DataFrame({
    'power_type': elasticity_by_power.keys(),
    'price_elasticity': [round(v,3) for v in elasticity_by_power.values()]
})

# 根据动力类型和弹性显著性制定策略
conditions = [
    strategy_table['power_type'] == 'PHEV',
    strategy_table['power_type'] == 'REEV',
    strategy_table['power_type'] == 'BEV'
]

# 修正：动态获取各动力类型对应的弹性系数
choices = [
    '适度降价配合品牌营销（弹性系数：%.2f）' % strategy_table[strategy_table.power_type=='PHEV'].price_elasticity.values[0],
    '探索非价格竞争手段（弹性系数：%.2f）' % strategy_table[strategy_table.power_type=='REEV'].price_elasticity.values[0],
    '维持价格，强化技术优势（弹性系数：%.2f）' % strategy_table[strategy_table.power_type=='BEV'].price_elasticity.values[0]
]

strategy_table['suggested_strategy'] = np.select(conditions, choices, default='需深度分析')

# 生成最终策略表
final_strategy_table = (
    df[['car_model', 'power_type']]
    .drop_duplicates()
    .merge(strategy_table, on='power_type')
    .sort_values(by=['power_type', 'price_elasticity'], ascending=[True, False])
    .reset_index(drop=True)
)

# 优化打印设置
print("\n策略建议:".center(50, '-'))
with pd.option_context('display.max_columns', None, 'display.width', 1000):
    print(final_strategy_table.to_markdown(index=False))

# --------------------------
# 4. 可视化分析
# --------------------------
plt.figure(figsize=(15,10))

# 价格弹性热力图
# 显式选择分组列之后的列
elasticity = df.groupby('car_model')[['ln_sales', 'ln_price']].apply(
    lambda x: sm.OLS(x['ln_sales'], sm.add_constant(x['ln_price'])).fit().params['ln_price'])
plt.subplot(2,2,1)
sns.heatmap(pd.DataFrame(elasticity, columns=['价格弹性']).round(2),
            annot=True, cmap='coolwarm', center=0)
plt.title('各车型价格弹性系数热力图')

# 动态趋势分析
plt.subplot(2,2,2)
for model in df['car_model'].unique():
    subset = df[df['car_model'] == model]
    plt.plot(subset['month'], subset['smoothed_price'], label=model)
plt.xticks(rotation=45)
plt.title('价格动态趋势')
plt.legend(bbox_to_anchor=(1.05,1))

# 雷达图（使用matplotlib实现）
# 创建极坐标子图
ax = plt.subplot(2, 2, 3, projection='polar')
radar_data = df.groupby('car_model')[['scaled_range', 'ln_price', 'ln_sales']].mean().reset_index()
labels = ['续航', '价格', '销量']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合

for index, row in radar_data.iterrows():
    values = row[['scaled_range', 'ln_price', 'ln_sales']].tolist()
    values += values[:1]  # 闭合
    ax.plot(angles, values, label=row['car_model'])

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title('各车型综合指标雷达图')
ax.legend(bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()