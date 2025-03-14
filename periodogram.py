import matplotlib.pyplot as plt
import numpy as np

class Periodogram:
    """Periodogram method for temporal phase unwrapping"""

    def __init__(self, param) -> None:
        self.param_file  = param

    def calculate_terrain_phase(self, Bn):
        # 读取卫星数据
        wavelength = self.param_file["sentinel-1"]["wavelength"]
        incidence_angle = self.param_file["sentinel-1"]["incidence_angle"] * np.pi / 180
        R = self.param_file["sentinel-1"]["H"] / np.cos(incidence_angle)

        h_start = self.param_file["search_range"]["h_range"][0]
        h_end   = self.param_file["search_range"]["h_range"][1]
        h_step  = self.param_file["search_range"]["h_step"]
        height_search_space   = np.arange(h_start, h_end + 1, 1) * h_step

        # 计算高程相位
        terrain_phase  = (4 * np.pi * height_search_space.reshape(-1, 1) * Bn.reshape(1, -1)) / (wavelength * R * np.sin(incidence_angle))

        return terrain_phase

    def remove_displace_phase(self, obs_phase, d_est):
        # 读取卫星数据
        wavelength = self.param_file["sentinel-1"]["wavelength"]
        displace = d_est

        # 计算 displace_phase
        displace_phase = (4 * np.pi * displace) / (wavelength)

        # 去除 displace_phase 并缠绕
        phase = obs_phase - displace_phase
        complex_signal = np.exp(1j * phase)
        phase = np.angle(complex_signal)

        return phase

    def remove_terrain_phase(self, obs_phase, h_est, Bn):
        # 卫星参数导入
        wavelength = self.param_file["sentinel-1"]["wavelength"]
        incidence_angle = self.param_file["sentinel-1"]["incidence_angle"] * np.pi / 180
        R = self.param_file["sentinel-1"]["H"] / np.cos(incidence_angle)

        # 生成真实相位
        terrain_phase  = (4 * np.pi * Bn * h_est) / (wavelength * R * np.sin(incidence_angle))

        # 相减,然后缠绕
        displac_phase = obs_phase - terrain_phase
        e_displac_phase = np.exp(1j * displac_phase)
        displac_phase = np.angle(e_displac_phase)

        return displac_phase

    def calculate_terrain(self, terrain_phase, Bn):
        # 获取卫星数据
        wavelength = self.param_file["sentinel-1"]["wavelength"]
        incidence_angle = self.param_file["sentinel-1"]["incidence_angle"] * np.pi / 180
        R = self.param_file["sentinel-1"]["H"] / np.cos(incidence_angle)

        # 确定搜索空间
        h_start = self.param_file["search_range"]["h_range"][0]
        h_end   = self.param_file["search_range"]["h_range"][1]
        h_step  = self.param_file["search_range"]["h_step"]
        height_search_space   = np.arange(h_start, h_end + 1, 1) * h_step

        # 生成真实相位
        terrain_phase_search  = (4 * np.pi * Bn.reshape(-1, 1) * height_search_space.reshape(1, -1)) / (wavelength * R * np.sin(incidence_angle))

        # 一维搜索,得到最大值对应的 h 分量值
        e_terrain_phase = np.exp(1j * terrain_phase)
        e_terrain_phase_search = np.exp(-1j * terrain_phase_search)
        objective_function = np.mean(e_terrain_phase.reshape(-1, 1) * e_terrain_phase_search, axis=0)
        result = height_search_space[np.argmax(np.abs(objective_function))]

        return result

    def calculate_displace(self, displac_phase):
        # 一维解缠
        # ! 加一个低通滤波器试试，找个效果好的
        phase_diff = np.diff(displac_phase, append=displac_phase[-1])
        phase_jumps = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        unwrapped_phase = np.cumsum(phase_jumps) + displac_phase[0]

        # 计算 est_displace
        wavelength = self.param_file["sentinel-1"]["wavelength"]
        d_est = (unwrapped_phase * wavelength) / (4 * np.pi)
        return d_est

    def estimation(self, obs_phase, Bn):
        # 步骤一
        # 根据 h 分量的搜索空间，计算高程相位，并从观测相位中减去
        terrain_phase = self.calculate_terrain_phase(Bn)
        displac_phase = obs_phase - terrain_phase

        # 步骤二:
        # 沿着 h 分量的方向进行累加，得到 displac_phase。
        # 注意：累加的时候，用 e 底，这样可以避免缠绕相位相减过程中的 造成的 大于 2 pi 的情况
        e_displac_phase = np.sum(np.exp(1j * displac_phase), axis=0)
        displac_phase = np.angle(e_displac_phase)
        # displac_phase = np.sum(displac_phase, axis=0)
        # displac_phase = np.angle(np.exp(1j * displac_phase))

        # 步骤三：
        # 根据 displac_phase 相位，计算 displac 值
        d_est = self.calculate_displace(displac_phase)

        # 步骤四:
        # 根据估计到的 d_est 得到 terrain_phase
        terrain_phase = self.remove_displace_phase(obs_phase, d_est)

        # 线性搜索,计算 h 分量
        h_est = self.calculate_terrain(terrain_phase, Bn)

        # 迭代
        for i in range(self.param_file["iterative_times"]):
            # 根据估计到的 h_est,去除 terrain_phase , 从而得到 displac_phase
            displac_phase = self.remove_terrain_phase(obs_phase, h_est, Bn)
            # 根据 displac_phase 相位，计算 displac 值
            d_est = self.calculate_displace(displac_phase)
            # 根据估计到的 d_est, 去除 displace_phase , 从而得到 terrain_phase
            terrain_phase = self.remove_displace_phase(obs_phase, d_est)
            # 线性搜索,计算 h_est
            h_est = self.calculate_terrain(terrain_phase, Bn)

        return d_est, h_est