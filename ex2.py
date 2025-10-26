import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------------------
# 数据（使用之前的 d2 和 time 数组）
# -------------------------------
time = [0, 12, 24, 36, 48, 60, 72, 84, 96]
d2 = [
    [2.0435345783628294, 1.5467654572979292, 1.6582256506561588, 1.6860015194641327, 2.107501899330166,
     1.391042466423581, 0.6845960470469842, 0.25643048888961406, 0.0],
    [1.6587776735495454, 1.1912284289379829, 0.7283301407104642, 0.3790699867907069, 0.37066561540274323,
     0.42592783311306187, 0.17200691238876503, 0.34401382477753006, 0.0],
    [2.8587329590103145, 2.722997167694354, 2.393586349699053, 1.836156592009196, 2.295195740011495, 1.3486328216158354,
     0.6960586860965252, 0.9065431378265857, 0.0],
    [0.4034352126590787, 0.4513664283223703, 0.48878380719463155, 0.19630505942730636, 0.24538132428413295,
     0.10697181878298848, 0.05802586138896153, 0.47632011533290664, 0.0],
    [4.047006916207546, 4.625150761380053, 4.590077649732354, 4.418309126979308, 3.169405335660317, 1.5592071142137565,
     0.4553288568753459, 1.1151669547183385, 0.0],
    [4.0409872781579965, 4.618271175037711, 3.032388809958165, 3.287050518445874, 4.108813148057343, 2.5880236489717166,
     0.8217626296114685, 3.000869793586084, 0.0]
]
time = np.array(time)
d2 = np.array(d2)

# 假设您已经有了误差数据，存储在 errors 数组中
# 例如，假设每个数据点的误差为 0.1
errors = np.full_like(d2, 0.1)

# -------------------------------
# 字体设置（中文字体和全局字体）
# -------------------------------
# 注意：请根据实际情况调整字体路径
font_path = r"C:\Windows\Fonts\simhei.ttf"
chinese_font = fm.FontProperties(fname=font_path, size=16)

plt.rc('font', family="Times New Roman", size=16)

# -------------------------------
# 创建 3 行 2 列的子图，采用自定义排列顺序
# -------------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 15), sharex=True)
axs = axs.flatten()
# 定义子图排列顺序（例如，将序列 0,1,3,4,2,5 分别放入 6 个子图）
order = [0, 1, 3, 4, 2, 5]

# -------------------------------
# 遍历每个序列，进行绘图
# -------------------------------
for idx, i in enumerate(order):
    ax = axs[idx]
    # 绘制真实数据（使用蓝色倒三角，仅显示数据点）
    ax.plot(time, d2[i], color='blue', marker='v', linestyle=' ', markersize=8, label="真实数据 Real Data")

    # 多项式拟合（这里采用 4 次多项式）
    coeffs = np.polyfit(time, d2[i], 4)
    poly_func = np.poly1d(coeffs)
    fitted_curve = poly_func(time)
    # 绘制拟合曲线（蓝色虚线）
    ax.plot(time, fitted_curve, color='blue', linestyle='--', linewidth=2, label="拟合曲线 Fitted Curve")

    # 添加误差条
    ax.errorbar(time, d2[i], yerr=errors[i], fmt=' ', color='black', capsize=5, label="误差条 Error Bars")

    # 标注真实数据中的最小值
    min_val = np.min(d2[i])
    min_idx = np.argmin(d2[i])
    ax.annotate(
        f"{min_val:.4f}",
        (time[min_idx], d2[i, min_idx]),
        xytext=(-20, 10),
        textcoords='offset points',
        fontsize=12,
        color='red',
        arrowprops=dict(facecolor='red', arrowstyle='->')
    )

    # 去除右边和上边的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置 y 轴标签（中英文）
    ax.set_ylabel("次级生产力/(mg C/L·day)", fontproperties=chinese_font)
    # 可选：设置子图标题

# 为底部子图添加 x 轴标签
for ax in axs[4:]:
    ax.set_xlabel("时间/h", fontproperties=chinese_font)

# 调整子图间距
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 在图形上方添加统一图例（共用真实数据与拟合曲线图例）
fig.legend(
    labels=["真实数据 Real Data", "拟合曲线 Fitted Curve", "误差条 Error Bars"],
    loc="upper center",
    fontsize=15,
    ncol=3,
    frameon=False,
    prop=chinese_font
)

plt.show()
