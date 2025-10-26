import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, font_manager
import matplotlib.patheffects as path_effects

# ================== 字体配置 ==================
rcParams['svg.fonttype'] = 'none'  # 保留文本为矢量字体
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

# 字体路径（可根据系统调整）
font_paths = [
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    r"C:\Windows\Fonts\times.ttf",   # Times New Roman
]
for fp in font_paths:
    try:
        font_manager.fontManager.addfont(fp)
    except:
        print(f"⚠ 未找到字体：{fp}")

rcParams['font.sans-serif'] = ['SimSun']
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.family'] = ['sans-serif']

# ================== 数据准备 ==================
labels = ['C', 'D', 'E', 'F', 'G', 'H']
a_0 = [10.082, 6.555, 9.832, 5.482, 23.449, 18.739]
a_1 = [10.08, 6.56, 9.83, 5.48, 23.45, 18.74]
b_0 = [11, 10, 10, 12, 11, 12]
b_1 = [11.996, 12.674, 11.444, 12.073, 11.923, 12.037]
r0 = [-0.08896, -0.03302, -0.06042, -0.04521, -0.1107, -0.1125]
r1 = [-0.08216, -0.03, -0.07883, -0.04887, -0.10012, -0.1415]
t0 = [38.29, 88.56, 50.16, 93.00, 11.15, 10.88]
t1 = [40, 96, 53, 96, 10, 9]

def compute_error(v0, v1):
    return np.abs(np.array(v1) - np.array(v0)) / 2

errors = {
    'a': compute_error(a_0, a_1),
    'b': compute_error(b_0, b_1),
    'r': compute_error(r0, r1),
    't': compute_error(t0, t1)
}

# ================== 绘图函数 ==================
def plot_bar(ax, ylabel, y0, y1, error):
    ax.bar(x - width / 2, y0, width, label='原始数据', color='skyblue', alpha=0.85)
    ax.bar(x + width / 2, y1, width, yerr=error, label='拟合数据',
           color='orange', capsize=4, alpha=0.85)

    # X 轴刻度
    xticks = ax.set_xticks(x)
    labels_obj = ax.set_xticklabels(labels, fontsize=14, fontname='Times New Roman')
    for lbl in labels_obj:
        lbl.set_path_effects([path_effects.withStroke(linewidth=.8, foreground='black')])

    # Y 轴刻度
    ax.tick_params(axis='y', labelsize=13, width=2, length=6, labelcolor='black')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_path_effects([path_effects.withStroke(linewidth=.8, foreground='black')])

    # 图例
    leg = ax.legend(prop={'size': 12, 'family': 'SimSun'}, loc='upper right')
    for text in leg.get_texts():
        text.set_path_effects([path_effects.withStroke(linewidth=.8, foreground='black')])

    # Y 轴标签
    label_obj = ax.set_ylabel(ylabel, fontsize=15, fontname='SimSun')
    label_obj.set_path_effects([path_effects.withStroke(linewidth=.8, foreground='black')])

    ax.grid(True, linestyle='--', alpha=0.5)

# ================== 绘图 ==================
x = np.arange(len(labels))
width = 0.35
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

plot_bar(
    axs[0, 0],
    '毒素累积速率 $\\alpha$ ($\\mu$g·L$^{-1}$·d$^{-1}$)\nToxin Accumulation Rate $\\alpha$',
    a_0, a_1, errors['a']
)
plot_bar(
    axs[0, 1],
    '环境承载力 $b$ ($\\times10^{3}$ cells·mL$^{-1}$)\nEnvironmental Carrying Capacity $b$',
    b_0, b_1, errors['b']
)
plot_bar(
    axs[1, 0],
    '种群增长率 $r$ (d$^{-1}$)\nPopulation Growth Rate $r$',
    r0, r1, errors['r']
)
plot_bar(
    axs[1, 1],
    '毒素显著时间 $t_{0}$ (d)\nSignificant Toxin Time $t_{0}$',
    t0, t1, errors['t']
)

plt.tight_layout()
plt.savefig('图4.svg', dpi=600, bbox_inches='tight', format='svg')
plt.show()
