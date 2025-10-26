import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from matplotlib import rcParams, font_manager as fm
import matplotlib.patheffects as path_effects

def simulation_kt(ax, t_data, N_data, chinese_font, is_last=False, plot_style=None, subplot_label=None):
    # 定义模型
    def K_t(t, b, alpha, t0):
        return b / (1 + np.exp(alpha * (t - t0)))

    def logistic_growth(N, t, r, b, alpha, t0):
        K = K_t(t, b, alpha, t0)
        return r * N * (1 - N / K)

    def solve_logistic(t, r, b, alpha, t0, N0):
        N = odeint(logistic_growth, N0, t, args=(r, b, alpha, t0))
        return N.flatten()

    def fit_func(t, r, b, alpha, t0, N0):
        return solve_logistic(t, r, b, alpha, t0, N0)

    # 拟合参数
    initial_guess = [0.1, 10, 0.1, 50, 8.25]
    params, _ = curve_fit(fit_func, t_data, N_data, p0=initial_guess, maxfev=10000)
    r_fit, b_fit, alpha_fit, t0_fit, N0_fit = params
    N_fitted = fit_func(t_data, r_fit, b_fit, alpha_fit, t0_fit, N0_fit)

    # 绘制数据
    ax.scatter(t_data, N_data, label='真实数据', color='red', marker='o', zorder=5)
    ax.plot(t_data, N_fitted, 'b--', label='拟合曲线')

    if plot_style:
        if plot_style[0] is not None:
            ax.plot(t_data, plot_style[0], color='#2CA02C', marker='o',
                    linestyle='-', label='对照组A', markersize=8)
        if plot_style[1] is not None:
            ax.plot(t_data, plot_style[1], color='#2CA02C', marker='s',
                    linestyle='-', label='对照组B', markersize=8)

    # 文字样式（加粗 + 轮廓）
    text_effect = [path_effects.withStroke(linewidth=0.8, foreground='black')]

    if is_last:
        txt = ax.set_xlabel('时间/(h)', fontproperties=chinese_font, size=18, weight='bold')
        txt.set_path_effects(text_effect)
    txt = ax.set_ylabel('种群数量/(ind.)', fontproperties=chinese_font, size=18, weight='bold')
    txt.set_path_effects(text_effect)

    # 坐标刻度
    ax.tick_params(axis='both', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(10)
        label.set_color('#2C3E50')
        label.set_fontweight('bold')
        label.set_path_effects(text_effect)

    # 子图标注 (a) (b)...
    if subplot_label:
        txt = ax.text(0.05, 1, subplot_label, transform=ax.transAxes,
                      fontproperties=fm.FontProperties(family='Times New Roman', weight='bold'),
                      size=22)
        txt.set_path_effects(text_effect)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)


def display_results():
    t_data = np.array([0, 12, 24, 36, 48, 60, 72, 84, 96])
    datasets = [
        [10, 8.25, 8, 7, 4.5, 3, 2.5, 2.25, 0.16],
        [10, 10, 9.66, 9.16, 9, 8.5, 7.33, 7.33, 6.83],
        [10, 7.83, 7, 6.83, 4.5, 4.33, 1.66, 0.666, 0.5],
        [10, 9, 8.66, 8.5, 7.66, 7.5, 6.33, 6.33, 5.16],
        [10, 5.83, 5.5, 4, 2.16, 1.16, 0.66, 0.16, 0.333],
        [10, 6.5, 7.83, 6.66, 5, 6, 4.33, 1.16, 0.566],
    ]
    control_A = [11, 10, 10, 12, 11, 12, 11, 11, 12]
    control_B = [11, 11, 10, 10, 12, 12, 12, 12, 10]

    # 字体设置
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.unicode_minus'] = False

    font_path = r"C:\Windows\Fonts\simsun.ttc"
    chinese_font = fm.FontProperties(fname=font_path, size=20, weight='bold')

    fig, axs = plt.subplots(3, 2, figsize=(16, 9))
    plt.subplots_adjust(hspace=0.6, wspace=0.2)

    for i, ax in enumerate(axs.flat):
        if i < len(datasets):
            plot_style = None
            if i in [2, 3, 5]:
                plot_style = [control_A, None]
            elif i in [0, 1, 4]:
                plot_style = [None, control_B]
            simulation_kt(ax, t_data, datasets[i], chinese_font,
                          is_last=(i >= 4),
                          plot_style=plot_style,
                          subplot_label=f'({chr(97 + i)})')

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='真实数据',
                   markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], color='blue', linestyle='--', label='拟合曲线'),
        plt.Line2D([0], [0], marker='o', color='#2CA02C', linestyle='-',
                   markersize=10, label='对照组A'),
        plt.Line2D([0], [0], marker='s', color='#2CA02C', linestyle='-',
                   markersize=10, label='对照组B')
    ]

    fig.legend(handles=legend_elements,
               loc='upper center',
               ncol=4,
               frameon=False,
               prop=chinese_font,
               bbox_to_anchor=(0.5, 1))

    # 输出SVG，保留可复制文字
    plt.savefig('图3.svg', dpi=600, bbox_inches='tight', format='svg')
    plt.show()


if __name__ == "__main__":
    display_results()
