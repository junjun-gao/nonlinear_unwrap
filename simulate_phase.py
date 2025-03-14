import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import random
from until import *

class SimulatedPhase:

    def __init__(self, param_file):
        self.param_file = param_file

    def generate_sub_phase(self, check_num):
        # 设定随机种子，确保实验的可重复性
        np.random.seed(check_num)
        # 卫星参数导入
        wavelength = self.param_file["sentinel-1"]["wavelength"]
        sigma_Bn = self.param_file["sentinel-1"]["Bn"]
        Bn = np.random.normal(0, sigma_Bn, self.param_file["Nifg"])
        incidence_angle = self.param_file["sentinel-1"]["incidence_angle"] * np.pi / 180
        R = self.param_file["sentinel-1"]["H"] / np.cos(incidence_angle)

        # 高程值
        terrain_height = self.param_file["param_simulation"]["height"]

        # 生成形变值
        max_value = self.param_file["param_simulation"]["displace_max"]
        min_value = self.param_file["param_simulation"]["displace_min"]
        x_points = np.array([0, 1, 2, 3, 4])
        y_points = np.array([np.random.randint(min_value, max_value) for _ in range(5)])
        # y_points = np.sort(np.array([np.random.randint(min_value, max_value) for _ in range(5)]))
        # 生成拉格朗日插值多项式
        poly = lagrange(x_points, y_points)
        # 生成 30 个等间距点
        x_points = np.linspace(min(x_points), max(x_points), self.param_file["Nifg"])
        displace = poly(x_points)
        # plot(displace)

        displace = np.array([x * 1e-3 for x in displace])           # 转换单位为： mm -> m

        # 10° 的噪声？？？？
        self.noise = np.random.normal(0, np.pi * self.param_file["noise_level"] / 180, self.param_file["Nifg"])

        # 生成真实相位
        terrain_phase  = (4 * np.pi * Bn * terrain_height) / (wavelength * R * np.sin(incidence_angle))
        displace_phase = (4 * np.pi * displace) / (wavelength)
        phase_true = terrain_phase + displace_phase + self.noise

        # 获取缠绕相位
        complex_signal = np.exp(1j * phase_true)
        obs_phase = np.angle(complex_signal)

        return obs_phase, Bn, displace
