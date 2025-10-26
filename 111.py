import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, font_manager, patheffects

# ================== 1. 字体配置 ==================
rcParams['svg.fonttype'] = 'none'  # 保留文本为矢量字体（可复制）
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

# 注册字体
font_paths = [
    r"C:\Windows\Fonts\simsun.ttc",        # 宋体
    r"C:\Windows\Fonts\times.ttf",         # Times New Roman
]
for fp in font_paths:
    try:
        font_manager.fontManager.addfont(fp)
    except:
        print(f"⚠ 未找到字体：{fp}")

rcParams['font.sans-serif'] = ['SimSun']
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.family'] = ['serif']
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'

# 定义统一的文字描边样式
text_effect = [patheffects.withStroke(linewidth=0.7, foreground='black')]

# ================== 2. 数据准备 ==================
a = [10.082, 6.555, 9.832, 5.482, 23.449, 18.739]
b = [11.996, 12.674, 11.444, 12.073, 11.923, 12.037]
r = [-0.08896, -0.03302, -0.06042, -0.04521, -0.1107, -0.1125]
t0 = [38.29, 88.56, 50.16, 93.00, 11.15, 10.88]
T = [0.72, 0.21, 0.63, 0.15, 0.98, 0.85]
labels = ['C', 'D', 'E', 'F', 'G', 'H']

sorted_indices = np.argsort(T)
sorted_T = np.array(T)[sorted_indices]
sorted_labels = np.array(labels)[sorted_indices]

# ================== 3. 创建画布 ==================
fig, (ax1, ax2_main) = plt.subplots(1, 2, figsize=(15, 6))

# ===== 左图：Bar图 =====
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b']
bars = ax1.bar(labels, T, color=bar_colors, edgecolor='black', linewidth=0.8)

for i, (bar, value) in enumerate(zip(bars, T)):
    height = bar.get_height()
    txt = ax1.text(bar.get_x() + bar.get_width()/2,
                   height + 0.02,
                   f'T={value}',
                   ha='center',
                   va='bottom',
                   color='black',
                   fontsize=11,
                   fontweight='bold',
                   fontname='Times New Roman')
    txt.set_path_effects(text_effect)

label_a = ax1.text(0.02, 1.02, '(a)', transform=ax1.transAxes,
                   fontsize=22, fontweight='bold', va='top', fontname='Times New Roman')
label_a.set_path_effects(text_effect)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.tick_params(axis='both', colors='black', width=0.8, labelsize=12)

for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontname('Times New Roman')
    tick.set_path_effects(text_effect)

ylabel = ax1.set_ylabel('毒性强度 T\nToxicity Intensity T',
                        color='black', fontsize=18, fontweight='bold', fontname='SimSun')
ylabel.set_path_effects(text_effect)
xlabel = ax1.set_xlabel('组别\nGroup',
                        color='black', fontsize=18, fontweight='bold', fontname='SimSun')
xlabel.set_path_effects(text_effect)

# ===== 右图：多y轴 =====
label_b = ax2_main.text(0.02, 1.02, '(b)', transform=ax2_main.transAxes,
                        fontsize=22, fontweight='bold', va='top', fontname='Times New Roman')
label_b.set_path_effects(text_effect)

xlabel2 = ax2_main.set_xlabel('毒性强度 T\nToxicity Intensity T',
                              color='black', fontsize=18, fontweight='bold', fontname='SimSun')
xlabel2.set_path_effects(text_effect)

l1, = ax2_main.plot(sorted_T, np.array(a)[sorted_indices], 'b', label='a', marker="^", linewidth=0.8)
l2, = ax2_main.plot(sorted_T, np.array(b)[sorted_indices], 'c', label='b', marker="D", linewidth=0.8)

ax2_main.spines['left'].set_color('b')
ax2_main.tick_params(axis='y', colors='black', width=0.8, labelsize=12)
for tick in ax2_main.get_yticklabels() + ax2_main.get_xticklabels():
    tick.set_fontweight('bold')
    tick.set_fontname('Times New Roman')
    tick.set_path_effects(text_effect)
ylabel2 = ax2_main.set_ylabel('参数a或参数b',
                              color='black', fontsize=18, fontweight='bold', fontname='SimSun')
ylabel2.set_path_effects(text_effect)

# r轴
ax2_r = ax2_main.twinx()
l3, = ax2_r.plot(sorted_T, np.array(r)[sorted_indices], 'g', label='r', marker='o', linewidth=0.8)
ax2_r.spines['right'].set_color('g')
ax2_r.tick_params(axis='y', colors='g', width=0.8, labelsize=12)
text_effect3 = [patheffects.withStroke(linewidth=0.7, foreground='green')]
for tick in ax2_r.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontname('Times New Roman')
    tick.set_path_effects(text_effect3)
ylabel3 = ax2_r.set_ylabel('参数r', color='g', va='top',
                           fontsize=18, fontweight='bold', fontname='SimSun')
text_effect3 = [patheffects.withStroke(linewidth=0.7, foreground='green')]
ylabel3.set_path_effects(text_effect3)

# t0轴
ax2_t0 = ax2_main.twinx()
ax2_t0.spines["right"].set_position(("axes", 1.12))
l4, = ax2_t0.plot(sorted_T, np.array(t0)[sorted_indices], 'r', label='t0', marker='s', linewidth=0.8)
ax2_t0.spines['right'].set_color('r')
ax2_t0.tick_params(axis='y', colors='r', width=0.8, labelsize=12)
text_effect4 = [patheffects.withStroke(linewidth=0.7, foreground='red')]
for tick in ax2_t0.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontname('Times New Roman')
    tick.set_path_effects(text_effect4)
ylabel4 = ax2_t0.set_ylabel(r'参数$t_0$', color='r', va='bottom',
                            fontsize=18, fontweight='bold', fontname='SimSun')

ylabel4.set_path_effects(text_effect4)

for ax in [ax2_main, ax2_r, ax2_t0]:
    ax.spines['top'].set_visible(False)
ax2_main.spines['bottom'].set_color('black')
ax2_main.tick_params(axis='x', colors='black', width=0.8)

legend = ax2_main.legend(handles=[l1, l2, l3, l4],
                         labels=['a', 'b', 'r', r'$t_0$'],
                         loc='upper left', frameon=False,
                         bbox_to_anchor=(0.32, 0.98),
                         prop={'size': 12, 'weight': 'bold', 'family': 'Times New Roman'})

plt.tight_layout()
plt.savefig('图5.svg', dpi=600, bbox_inches='tight', format='svg')
plt.show()
