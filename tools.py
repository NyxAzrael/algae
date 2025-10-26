
import json
import os
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PIL.ImageFilter import GaussianBlur
from PIL import Image
from matplotlib import rcParams
import matplotlib.font_manager as fm
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from win32comext.propsys.propsys import PROPVARIANTType


def resize_pic():
    path = "basesrc/back.jpg"

    image = Image.open(path)
    image = image.resize((320,320))
    save_path = os.path.join(path)
    image.save(save_path)


def convert_label_json(json_dir, save_dir, classes):
    files = os.listdir(json_dir)
    # 删选出json文件
    jsonFiles = []
    for file in files:
        if os.path.splitext(file)[1] == ".json":
            jsonFiles.append(file)
    # 获取类型
    classes = classes.split(',')

    # 获取json对应中对应元素
    for json_path in tqdm(jsonFiles):
        path = os.path.join(json_dir, json_path)
        with open(path, 'r') as loadFile:
            print(loadFile)
            json_dict = json.load(loadFile)
        h, w = json_dict['imageHeight'], json_dict['imageWidth']
        txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
        txt_file = open(txt_path, 'w')

        for shape_dict in json_dict['shapes']:
            label = shape_dict['label']
            label_index = classes.index(label)
            points = shape_dict['points']
            points_nor_list = []
            for point in points:
                points_nor_list.append(point[0] / w)
                points_nor_list.append(point[1] / h)
            points_nor_list = list(map(lambda x: str(x), points_nor_list))
            points_nor_str = ' '.join(points_nor_list)
            label_str = str(label_index) + ' ' + points_nor_str + '\n'
            txt_file.writelines(label_str)


def blur_now(image:Image):
    return image.filter(GaussianBlur(radius=1))

def video_creator(frames):
    clip = ImageSequenceClip(frames,fps=10,with_mask=False,load_images=True)
    clip.write_videofile("1.mp4",codec="libx264",audio=False)

def logistic_simulation(r,K0,alpha,t0,P0,t_span):

    def K(t):
        return K0 / (1 + np.exp(alpha * (t - t0)))

    def logistic(t, P):
        return r * P * (1 - P / K(t))

    t_eval = np.linspace(*t_span, 500)  # 评估时间点，用于绘图
    sol = solve_ivp(logistic, t_span, [P0], t_eval=t_eval, method='RK45')


    plt.plot(sol.t, sol.y[0], label="Population P(t)")
    plt.xlabel('Time t')
    plt.ylabel('Population P(t)')
    plt.title('Solution of Logistic Equation with Varying K(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from matplotlib import rcParams, font_manager as fm


def simulation_kt(ax, t_data, N_data, is_last=False, plot_style=None):
    # Define K(t) formula
    def K_t(t, b, alpha, t0):
        return b / (1 + np.exp(alpha * (t - t0)))

    # Differential equation dN/dt = r * N(t) * (1 - N(t) / K(t))
    def logistic_growth(N, t, r, b, alpha, t0):
        K = K_t(t, b, alpha, t0)
        dNdt = r * N * (1 - N / K)
        return dNdt

    # Solving the logistic growth equation using odeint
    def solve_logistic(t, r, b, alpha, t0, N0):
        N = odeint(logistic_growth, N0, t, args=(r, b, alpha, t0))
        return N.flatten()

    # Curve fitting function
    def fit_func(t, r, b, alpha, t0, N0):
        return solve_logistic(t, r, b, alpha, t0, N0)

    # Initial guess for parameters [r, b, alpha, t0, N0]
    initial_guess = [0.1, 10, 0.1, 50, 8.25]

    # Curve fitting to find the best parameters
    params, covariance = curve_fit(fit_func, t_data, N_data, p0=initial_guess, maxfev=10000)

    # Extracting the fitted parameters
    r_fit, b_fit, alpha_fit, t0_fit, N0_fit = params

    # Generate fitted curve using the best-fit parameters
    N_fitted = fit_func(t_data, r_fit, b_fit, alpha_fit, t0_fit, N0_fit)

    # Plotting on the provided axis
    ax.scatter(t_data, N_data, label='Data真实数据', color='red', marker='o')
    ax.plot(t_data, N_fitted, label='Curve拟合曲线', color='blue', linestyle='--')

    # Plot the additional data (new arrays) based on the `plot_style`
    if plot_style:
        # Check if the first element is not None before plotting
        if plot_style[0] is not None:
            ax.plot(t_data, plot_style[0], label='New Data (Array 1)', color='green', marker='s', linestyle='-',
                    markersize=8)  # Green squares
        # Check if the second element is not None before plotting
        if plot_style[1] is not None:
            ax.plot(t_data, plot_style[1], label='New Data (Array 2)', color='green', marker='o', linestyle='-',
                    markersize=6)  # Green circles

    if is_last:
        ax.set_xlabel('时间（小时）\nTime (h)', fontsize=12, family="SimHei")
    ax.set_ylabel('种群密度\nPopulation Density', fontsize=12, family="SimHei")  # Chinese and English label
    ax.legend(prop=chinese_font, frameon=False, fontsize=3, markerscale=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust y-axis to keep min and max values but reduce the number of ticks
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max)  # Ensures min and max limits are the same
    ax.locator_params(axis='y', nbins=5)  # Reduce the number of y-axis ticks


def display_results():
    # Provided data
    t2 = np.array([0, 12, 24, 36, 48, 60, 72, 84, 96])
    d2 = [
        [0.0, -0.0025643048888961406, -0.006845960470469842, -0.01391042466423581, -0.02107501899330166,
         -0.016860015194641326, -0.016582256506561588, -0.015467654572979293, -0.020435345783628295],
        [0.0, -0.0034401382477753004, -0.0017200691238876502, -0.004259278331130619, -0.0037066561540274323,
         -0.0037906998679070694, -0.0072833014071046415, -0.01191228428937983, -0.016587776735495455],
        [0.0, -0.009065431378265857, -0.006960586860965253, -0.013486328216158354, -0.02295195740011495,
         -0.01836156592009196, -0.02393586349699053, -0.027229971676943538, -0.038587329590103144],
        [0.0, 0.004763201153329066, -0.0005802586138896153, -0.0010697181878298848, -0.0024538132428413295,
         -0.0019630505942730635, -0.0048878380719463155, -0.004513664283223703, -0.004034352126590787],
        [0.0, 0.0011151669547183385, 0.0004553288568753459, -0.0015592071142137565, -0.0031694053356603173,
         -0.004418309126979308, -0.004590077649732354, -0.004625150761380053, -0.004047006916207546],
        [0.0, -0.003000869793586084, -0.008217626296114685, -0.0025880236489717167, -0.004108813148057343,
         -0.003287050518445874, -0.0030323888099581652, -0.004618271175037711, -0.004040987278157997],
    ]

    # New data arrays
    new_data1 = [
        0.0, 0.0014537738278873613, 0.0007268869139436807, -0.0024955176002377203, -0.001474355643504316,
        -0.002132124745469266, -0.00023197013358238082, -0.0005163106750011039, -0.0015430228338091194
    ]
    new_data2 = [
        0.0, -0.0015047319789368945, -0.0015185674966856817, 0.0014248137329875164, 0.0010686102997406373,
        0.0008548882397925099, 0.0, 0.0, -0.0009643445420015792
    ]

    # Setting up a 3x2 grid for subplots with narrower width and increased spacing
    fig, axes = plt.subplots(3, 2, figsize=(9, 6))  # Reduced width for narrower plots
    axes = axes.flatten()

    # Loop through each dataset and subplot
    for i, data in enumerate(d2):
        is_last = False
        plot_style = None

        # Assigning the new data arrays to the specified plots
        if i == 2 or i == 3 or i == 5:
            plot_style = [new_data1, None]  # Green square lines in subplots 3, 4, 6
        elif i == 0 or i == 1 or i == 4:
            plot_style = [None, new_data2]  # Green circle lines in subplots 1, 2, 5

        if i == 4 or i == 5:
            is_last = True

        simulation_kt(axes[i], t2, data, is_last=is_last, plot_style=plot_style)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.6, wspace=0.3)  # Increase hspace and wspace for more space between plots
    # Show the plot
    plt.show()


if __name__ == "__main__":
    rcParams['font.sans-serif'] = ['SimSun']  # 中文宋体
    rcParams['font.family'] = 'Times New Roman'  # 英文 Times New Roman
    rcParams['axes.unicode_minus'] = False
    font_path = r"C:\Windows\Fonts\simhei.ttf"
    chinese_font = fm.FontProperties(fname=font_path, size=10)

    # Call the function to display the results
    display_results()

