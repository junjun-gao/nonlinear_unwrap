#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
periodogram_temporal_unwrap.py

独立的一维 Periodogram 时间域解缠模块。

功能：
1. 对单条时间序列相位进行残余高程误差和形变序列估计
2. 对 Delaunay 网络中的所有 arc 进行批量时间域解缠

输入：
    obs_phase : shape = (n_time,)
        单条缠绕观测相位序列，单位 rad

    bperp : shape = (n_time,)
        每个时相对应的垂直基线，单位 m

输出：
    d_est : shape = (n_time,)
        估计形变量时间序列，单位 m

    h_est : float
        估计高程误差，单位 m

注意：
    这里沿用了你给出的相位模型：

        height_phase = 4*pi*h*Bperp / (lambda*R*sin(theta))
        deformation_phase = 4*pi*d / lambda

    如果你生成观测相位时使用的是负号：
        deformation_phase = -4*pi*d / lambda

    那么需要把 displacement_sign 设置为 -1。
"""

import numpy as np


def wrap_phase(phase):
    """
    将相位缠绕到 [-pi, pi]
    """
    return np.angle(np.exp(1j * phase))


class PeriodogramTemporalUnwrapper(object):
    """
    一维 Periodogram 时间域解缠类。

    param 格式示例：

    param = {
        "sentinel-1": {
            "wavelength": 0.055465,
            "incidence_angle": 34.0,
            "H": 800000.0
        },
        "search_range": {
            "h_range": [-20, 20],
            "h_step": 0.1
        },
        "iterative_times": 2,
        "displacement_sign": 1
    }

    说明：
        H 这里按照你原始代码使用：
            R = H / cos(incidence_angle)

        如果你想直接使用斜距 R，也可以把 H 设置为：
            H = R * cos(incidence_angle)
    """

    def __init__(self, param):
        self.param = param

        self.wavelength = self.param["sentinel-1"]["wavelength"]
        self.incidence_angle = (
            self.param["sentinel-1"]["incidence_angle"] * np.pi / 180.0
        )
        self.H = self.param["sentinel-1"]["H"]

        self.R = self.H / np.cos(self.incidence_angle)

        self.h_start = self.param["search_range"]["h_range"][0]
        self.h_end = self.param["search_range"]["h_range"][1]
        self.h_step = self.param["search_range"]["h_step"]

        self.height_search_space = np.arange(
            self.h_start,
            self.h_end + self.h_step,
            self.h_step
        )

        self.iterative_times = self.param.get("iterative_times", 2)

        # 形变相位符号
        # 你的原始代码是 +4*pi*d/lambda，所以默认是 +1
        # 如果你的模拟观测相位生成时用的是 -4*pi*d/lambda，则设置为 -1
        self.displacement_sign = self.param.get("displacement_sign", 1)

    def lowpass_filter_complex(self, z, window_size=5):
        """
        对复数相位序列进行低通滤波，避免直接滤波缠绕相位造成跳变错误。
        """

        z = np.asarray(z, dtype=complex)

        kernel = np.ones(window_size) / window_size

        real_filtered = np.convolve(z.real, kernel, mode="same")
        imag_filtered = np.convolve(z.imag, kernel, mode="same")

        z_filtered = real_filtered + 1j * imag_filtered

        return z_filtered

    def calculate_terrain_phase_search(self, bperp):
        """
        根据高程搜索空间，计算所有候选高程误差对应的地形相位。

        返回：
            terrain_phase: shape = (n_height, n_time)
        """

        bperp = np.asarray(bperp, dtype=float)

        terrain_phase = (
            4.0
            * np.pi
            * self.height_search_space.reshape(-1, 1)
            * bperp.reshape(1, -1)
            / (
                self.wavelength
                * self.R
                * np.sin(self.incidence_angle)
            )
        )

        return terrain_phase

    def calculate_single_terrain_phase(self, h_est, bperp):
        """
        根据一个 h_est 计算对应的地形相位。

        返回：
            terrain_phase: shape = (n_time,)
        """

        bperp = np.asarray(bperp, dtype=float)

        terrain_phase = (
            4.0
            * np.pi
            * bperp
            * h_est
            / (
                self.wavelength
                * self.R
                * np.sin(self.incidence_angle)
            )
        )

        return terrain_phase

    def calculate_single_displacement_phase(self, d_est):
        """
        根据形变量计算形变相位。

        d_est 单位：m
        """

        d_est = np.asarray(d_est, dtype=float)

        displacement_phase = (
            self.displacement_sign
            * 4.0
            * np.pi
            * d_est
            / self.wavelength
        )

        return displacement_phase

    def remove_displacement_phase(self, obs_phase, d_est):
        """
        从观测相位中去除形变相位，得到主要由高程误差造成的相位。
        """

        obs_phase = np.asarray(obs_phase, dtype=float)

        displacement_phase = self.calculate_single_displacement_phase(d_est)

        terrain_phase = obs_phase - displacement_phase
        terrain_phase = wrap_phase(terrain_phase)

        return terrain_phase

    def remove_terrain_phase(self, obs_phase, h_est, bperp):
        """
        从观测相位中去除地形残差相位，得到形变相位。
        """

        obs_phase = np.asarray(obs_phase, dtype=float)

        terrain_phase = self.calculate_single_terrain_phase(h_est, bperp)

        displacement_phase = obs_phase - terrain_phase
        displacement_phase = wrap_phase(displacement_phase)

        return displacement_phase

    def calculate_terrain(self, terrain_phase, bperp):
        """
        一维搜索估计高程误差 h。

        输入：
            terrain_phase: shape = (n_time,)
                去除形变后的地形残差相位，仍然是缠绕相位

            bperp: shape = (n_time,)
                垂直基线

        输出：
            h_est: float
                估计高程误差，单位 m
        """

        terrain_phase = np.asarray(terrain_phase, dtype=float)
        bperp = np.asarray(bperp, dtype=float)

        terrain_phase_search = self.calculate_terrain_phase_search(bperp)

        e_terrain_phase = np.exp(1j * terrain_phase)
        e_terrain_phase_search = np.exp(-1j * terrain_phase_search)

        objective_function = np.mean(
            e_terrain_phase.reshape(-1, 1) * e_terrain_phase_search.T,
            axis=0
        )

        h_est = self.height_search_space[
            np.argmax(np.abs(objective_function))
        ]

        return h_est

    def calculate_displacement(self, displacement_phase):
        """
        根据去除地形残差后的形变相位，进行一维时间解缠，并转换成形变量。

        输入：
            displacement_phase: shape = (n_time,)
                形变缠绕相位

        输出：
            d_est: shape = (n_time,)
                形变量，单位 m
        """

        displacement_phase = np.asarray(displacement_phase, dtype=float)

        # 这里保留你原来的解缠逻辑
        phase_diff = np.diff(
            np.concatenate([displacement_phase, displacement_phase[-1:]])
        )

        phase_jumps = (phase_diff + np.pi) % (2.0 * np.pi) - np.pi

        unwrapped_phase = np.cumsum(phase_jumps) + displacement_phase[0]

        d_est = (
            unwrapped_phase
            * self.wavelength
            / (
                self.displacement_sign
                * 4.0
                * np.pi
            )
        )

        return d_est

    def estimate_initial_displacement(self, obs_phase, bperp):
        """
        初始估计形变量。

        对所有高程候选相位进行去除，然后沿高程搜索方向累加，
        得到一个近似的形变相位，再进行时间解缠。
        """

        terrain_phase_search = self.calculate_terrain_phase_search(bperp)

        displacement_phase_candidates = (
            obs_phase.reshape(1, -1) - terrain_phase_search
        )

        e_displacement_phase = np.sum(
            np.exp(1j * displacement_phase_candidates),
            axis=0
        )

        # =====================================================
        # 在复数域对时间序列进行低通滤波
        # =====================================================
        e_displacement_phase = self.lowpass_filter_complex(
            e_displacement_phase,
            window_size=5
        )

        displacement_phase = np.angle(e_displacement_phase)

        d_est = self.calculate_displacement(displacement_phase)

        return d_est

    def estimation(self, obs_phase, bperp):
        """
        单条时间序列相位解缠接口。

        输入：
            obs_phase: shape = (n_time,)
                缠绕观测相位，单位 rad

            bperp: shape = (n_time,)
                垂直基线，单位 m

        输出：
            d_est: shape = (n_time,)
                估计形变量，单位 m

            h_est: float
                估计高程误差，单位 m
        """

        obs_phase = np.asarray(obs_phase, dtype=float)
        bperp = np.asarray(bperp, dtype=float)

        if obs_phase.ndim != 1:
            raise ValueError("obs_phase 必须是一维数组。")

        if bperp.ndim != 1:
            raise ValueError("bperp 必须是一维数组。")

        if obs_phase.shape[0] != bperp.shape[0]:
            raise ValueError("obs_phase 和 bperp 长度必须一致。")

        # ----------------------------------------------------
        # 步骤 1：初始估计形变量
        # ----------------------------------------------------
        d_est = self.estimate_initial_displacement(obs_phase, bperp)

        # ----------------------------------------------------
        # 步骤 2：去除形变相位，估计高程误差
        # ----------------------------------------------------
        terrain_phase = self.remove_displacement_phase(obs_phase, d_est)

        h_est = self.calculate_terrain(terrain_phase, bperp)

        # ----------------------------------------------------
        # 步骤 3：迭代优化
        # ----------------------------------------------------
        for _ in range(self.iterative_times):

            displacement_phase = self.remove_terrain_phase(
                obs_phase,
                h_est,
                bperp
            )

            d_est = self.calculate_displacement(displacement_phase)

            terrain_phase = self.remove_displacement_phase(
                obs_phase,
                d_est
            )

            h_est = self.calculate_terrain(
                terrain_phase,
                bperp
            )

        return d_est, h_est



def unwrap_arcs_periodogram(
    phase_wrapped,
    arcs,
    bperp,
    param
):
    """
    批量 arc 时间域解缠接口。

    输入：
        phase_wrapped: shape = (n_points, n_time)
            每个 PS 点的缠绕观测相位

        arcs: shape = (n_arcs, 2)
            Delaunay 网络边，每行是 [p1, p2]

        bperp: shape = (n_time,)
            垂直基线序列，单位 m

        param: dict
            Periodogram 参数

    输出：
        net: dict
            包含 arc 时间解缠结果
    """

    phase_wrapped = np.asarray(phase_wrapped, dtype=float)
    arcs = np.asarray(arcs, dtype=int)
    bperp = np.asarray(bperp, dtype=float)

    n_points, n_time = phase_wrapped.shape
    n_arcs = arcs.shape[0]

    if bperp.shape[0] != n_time:
        raise ValueError("bperp 长度必须和相位时相数量一致。")

    unwrapper = PeriodogramTemporalUnwrapper(param)

    arc_phase_wrapped = np.zeros((n_arcs, n_time), dtype=float)
    arc_phase_unwrapped = np.zeros((n_arcs, n_time), dtype=float)

    # 新增：保存整数模糊度
    arc_ambiguity = np.zeros((n_arcs, n_time), dtype=int)

    arc_delta_h = np.zeros(n_arcs, dtype=float)
    arc_res_std = np.zeros(n_arcs, dtype=float)

    for idx in range(n_arcs):
        p1 = arcs[idx, 0]
        p2 = arcs[idx, 1]

        # ----------------------------------------------------
        # 1. arc 差分观测相位
        #    value[p2] - value[p1]
        # ----------------------------------------------------
        diff_phase = wrap_phase(
            phase_wrapped[p2, :] - phase_wrapped[p1, :]
        )

        # ----------------------------------------------------
        # 2. Periodogram 估计该 arc 的形变序列和高程误差
        # ----------------------------------------------------
        d_est, h_est = unwrapper.estimation(
            obs_phase=diff_phase,
            bperp=bperp
        )

        # ----------------------------------------------------
        # 3. 根据估计的 d 和 h 构造模型相位
        # ----------------------------------------------------
        terrain_phase = unwrapper.calculate_single_terrain_phase(
            h_est,
            bperp
        )

        displacement_phase = unwrapper.calculate_single_displacement_phase(
            d_est
        )

        diff_phase_model = terrain_phase + displacement_phase

        # ----------------------------------------------------
        # 4. 根据模型相位反推整数模糊度
        #
        #    diff_phase_model ≈ diff_phase + 2πk
        #
        #    所以：
        #    k = round((diff_phase_model - diff_phase) / 2π)
        # ----------------------------------------------------
        k = np.round(
            (diff_phase_model - diff_phase) / (2.0 * np.pi)
        ).astype(int)

        # ----------------------------------------------------
        # 5. 用整数模糊度修正原始观测缠绕相位
        #
        #    这一步才是真正的解缠相位：
        #    diff_phase_unwrapped = diff_phase + 2πk
        # ----------------------------------------------------
        diff_phase_unwrapped = diff_phase + 2.0 * np.pi * k

        # ----------------------------------------------------
        # 6. 残差评价
        #    注意这里用解缠相位和模型相位比较
        # ----------------------------------------------------
        residual = diff_phase_unwrapped - diff_phase_model

        arc_phase_wrapped[idx, :] = diff_phase
        arc_phase_unwrapped[idx, :] = diff_phase_unwrapped
        arc_ambiguity[idx, :] = k

        arc_delta_h[idx] = h_est
        arc_res_std[idx] = np.std(residual)

    net = {
        "arcs": arcs,
        "arc_phase_wrapped": arc_phase_wrapped,
        "arc_phase_unwrapped": arc_phase_unwrapped,
        "arc_ambiguity": arc_ambiguity,
        "arc_delta_h": arc_delta_h,
        "arc_res_std": arc_res_std,
    }

    return net