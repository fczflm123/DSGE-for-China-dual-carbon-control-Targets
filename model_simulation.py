import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# 设置Matplotlib中文显示字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False            # 正常显示负号

# ========== 1. 参数设置 ==========
# 折现因子/率（年）beta，偏好参数
beta = 0.97   # 折现因子 β，每年约3%贴现率
sigma = 2.0   # 相对风险厌恶系数 σ（决定消费边际效用曲率）
phi   = 2.0   # 劳动供给弹性参数 φ（Frisch 弹性 = 1/φ = 0.5）
alpha = 0.4   # 资本产出弹性 α（资本在产出中的份额约40%）
delta = 0.10  # 资本折旧率 δ（每年10%折旧）
A     = 1.0   # 初始全要素生产率 A（归一化）

# 利用初始数据校准劳动偏好参数 chi。
# 2024年: 假设GDP = 1.0（规模归一化），消费≈0.55，投资≈0.45
Y0 = 1.0
C0 = 0.55
I0 = Y0 - C0
gamma0 = 1.0

# 假设初始资本存量 K0 使投资占比约45%，利用稳态近似 I0 ≈ δ*K0 求 K0
K0 = I0 / delta
# 由生产函数 Y0 = A * K0^alpha * L0^(1-alpha) 求解初始劳动投入 L0
L0 = (Y0 / (A * (K0**alpha))) ** (1/(1 - alpha))

# 用家庭的劳动最优条件校准 chi:
w0 = (1 - alpha) * A * (K0**alpha) * (L0**(-alpha))
chi = (w0 * (C0 ** -sigma)) / (L0 ** phi)

# 将主要参数保存为表格
params = [
    ("beta", beta, "折现因子 β"),
    ("sigma", sigma, "相对风险厌恶系数 σ"),
    ("phi", phi, "劳动供给弹性参数 φ (Frisch=1/φ)"),
    ("alpha", alpha, "资本产出弹性 α"),
    ("delta", delta, "资本折旧率 δ"),
    ("A", A, "全要素生产率 A (初始值)"),
    ("chi", chi, "偏好参数 χ (劳动效用权重)"),
    ("gamma0", gamma0, "初始碳强度 γ0"),
    ("mu_target", 0.8, "2030年碳强度目标 μ"),
    ("tau_loss_coeff", 0.5, "减排产出损失系数(二次) 1/2"),
]
with open("parameters.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["参数", "值", "描述"])
    for name, value, desc in params:
        writer.writerow([name, f"{value:.4f}", desc])

# ========== 2. 定义碳强度情景路径 ==========
years = np.arange(2024, 2031)
T = len(years)
mu_base = np.ones(T)
mu_loose = np.ones(T)
mu_tight = np.ones(T)

# 基准: 2025-2030线性降低，总降幅20%
for i, year in enumerate(years):
    if year >= 2025:
        mu_base[i] = 1.0 - 0.2 * ((year - 2024) / 6.0)

# 先松后紧: 2025-2027累计降约0.05，2028-2030再降0.15
for i, year in enumerate(years):
    if 2025 <= year <= 2027:
        mu_loose[i] = 1.0 - (0.05 * (year - 2024) / 3.0)
    elif year >= 2028:
        mu_loose[i] = 0.95 - (0.15 * (year - 2027) / 3.0)

# 先紧后松: 2025-2027累计降0.15，2028-2030仅降0.05
for i, year in enumerate(years):
    if 2025 <= year <= 2027:
        mu_tight[i] = 1.0 - (0.15 * (year - 2024) / 3.0)
    elif year >= 2028:
        mu_tight[i] = 0.85 - (0.05 * (year - 2027) / 3.0)

# ========== 3. 定义模型求解函数 ==========
def simulate_path(mu_series):
    results = {
        "Year": [], "GDP": [], "HighCarbonOutput": [], "LowCarbonOutput": [],
        "Consumption": [], "Investment": [], "Labor": [], "Capital": [], "CarbonIntensity": []
    }
    # 初始化状态（从2024年数据开始）
    C = C0
    K = K0
    L = L0
    
    for i, year in enumerate(years):
        mu = mu_series[i]
        # 减排产出损失系数
        tau = 1.0 - 0.5 * ((1.0 - mu) ** 2)
        # 更新劳动供给
        L = ((1 - alpha) * tau * A * (K ** alpha) / (chi * (C ** sigma))) ** (1.0 / (phi + alpha))
        if L > 1.0:
            L = 1.0
        
        # 计算产出和要素回报
        Y = tau * A * (K ** alpha) * (L ** (1 - alpha))
        Y_high = mu * Y
        Y_low = Y - Y_high
        w = (1 - alpha) * tau * A * (K ** alpha) * (L ** -alpha)
        R = alpha * tau * A * (K ** (alpha - 1)) * (L ** (1 - alpha))
        
        # 资源分配
        I = Y - C
        K_next = (1 - delta) * K + I
        
        # 保存当期结果
        results["Year"].append(year)
        results["GDP"].append(Y)
        results["HighCarbonOutput"].append(Y_high)
        results["LowCarbonOutput"].append(Y_low)
        results["Consumption"].append(C)
        results["Investment"].append(I)
        results["Labor"].append(L)
        results["Capital"].append(K)
        results["CarbonIntensity"].append(mu)
        
        # 计算下一期消费 C_next（Euler方程）
        if i < T - 1:
            mu_next = mu_series[i + 1]
            
            def euler_residual(C_next):
                tau_next = 1.0 - 0.5 * ((1.0 - mu_next) ** 2)
                L_next = ((1 - alpha) * tau_next * A * (K_next ** alpha) /
                          (chi * (C_next ** sigma))) ** (1.0 / (phi + alpha))
                L_next = 1.0 if L_next > 1.0 else L_next
                R_next = alpha * tau_next * A * (K_next ** (alpha - 1)) * (L_next ** (1 - alpha))
                return beta * ((C / C_next) ** sigma) * (R_next + 1 - delta) - 1.0
            
            C_low, C_high = 1e-8, Y
            for _ in range(50):
                C_mid = 0.5 * (C_low + C_high)
                res = euler_residual(C_mid)
                if res > 0:
                    C_low = C_mid
                else:
                    C_high = C_mid
            C_next = 0.5 * (C_low + C_high)
        else:
            C_next = C
        
        # 更新状态
        C = C_next
        K = K_next
    
    df = pd.DataFrame(results)
    return df

# 运行模拟三种情景
df_base = simulate_path(mu_base)
df_loose = simulate_path(mu_loose)
df_tight = simulate_path(mu_tight)

# 输出结果CSV
df_base.to_csv("results_baseline.csv", index=False, float_format="%.6f")
df_loose.to_csv("results_loose_tight.csv", index=False, float_format="%.6f")
df_tight.to_csv("results_tight_loose.csv", index=False, float_format="%.6f")

# ========== 4. 结果可视化 ==========

# 提取绘图区间的数据 (2024-2030)
years_plot = df_base["Year"]

# 三种情景
scenarios_ch = {
    "基准情景": df_base,
    "先松后紧情景": df_loose,
    "先紧后松情景": df_tight
}
scenarios_en = {
    "Baseline Scenario": df_base,
    "Loose-then-tight": df_loose,
    "Tight-then-loose": df_tight
}

styles_ch = {
    "基准情景":       {"color": "blue",  "linestyle": "-"},
    "先松后紧情景":   {"color": "red",   "linestyle": "--"},
    "先紧后松情景":   {"color": "green", "linestyle": "-."}
}
styles_en = {
    "Baseline Scenario":  {"color": "blue",  "linestyle": "-"},
    "Loose-then-tight":   {"color": "red",   "linestyle": "--"},
    "Tight-then-loose":   {"color": "green", "linestyle": "-."}
}

# 为简化，可将要绘制的变量信息组织为列表，便于循环
# (变量名, 中文标题, 中文y轴, 英文标题, 英文y轴, 文件名前缀, DataFrame列名)
plot_vars = [
    ("GDP", "GDP（国内生产总值）", "指数",
     "GDP", "Index", "gdp", "GDP"),
    ("CarbonIntensity", "碳排放强度", "单位GDP排放（相对值）",
     "Carbon Intensity", "Relative Emission", "intensity", "CarbonIntensity"),
    ("Consumption", "居民消费", "指数",
     "Consumption", "Index", "consumption", "Consumption"),
    ("Investment", "全社会投资", "指数",
     "Investment", "Index", "investment", "Investment"),
    ("HighCarbonOutput", "高碳产出", "指数",
     "High-Carbon Output", "Index", "high_carbon", "HighCarbonOutput"),
    ("LowCarbonOutput", "低碳产出", "指数",
     "Low-Carbon Output", "Index ", "low_carbon", "LowCarbonOutput"),
    ("Capital", "资本存量", "指数",
     "Capital Stock", "Index", "capital", "Capital"),
    ("Labor", "劳动投入", "劳动力（占比）",
     "Labor Supply", "Labor (fraction)", "labor", "Labor")
]

# 按需求：每个变量输出两张图（中文 + 英文），且每张图仅包含该变量
for (var_key, title_ch, ylabel_ch,
     title_en, ylabel_en, file_prefix, df_col) in plot_vars:
    
    # ========== 中文版本 ==========
    plt.figure(figsize=(6,5))
    for name, df in scenarios_ch.items():
        plt.plot(df["Year"], df[df_col], label=name, **styles_ch[name])
    plt.title(title_ch)
    plt.xlabel("年份")
    plt.ylabel(ylabel_ch)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename_ch = f"{file_prefix}_scenarios_ch.png"
    plt.savefig(filename_ch)
    plt.show()
    
    # ========== 英文版本 ==========
    plt.figure(figsize=(6,5))
    for name, df in scenarios_en.items():
        plt.plot(df["Year"], df[df_col], label=name, **styles_en[name])
    plt.title(title_en)
    plt.xlabel("Year")
    plt.ylabel(ylabel_en)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename_en = f"{file_prefix}_scenarios_en.png"
    plt.savefig(filename_en)
    plt.show()
