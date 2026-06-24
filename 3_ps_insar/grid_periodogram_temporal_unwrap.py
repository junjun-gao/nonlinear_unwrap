#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
grid_periodogram_temporal_unwrap.py

传统二维网格 Periodogram 时间域解缠模块。

用于对比算法：

    同时在 height-search-space 和 velocity-search-space 上做二维搜索，
    找到使 temporal coherence 最大的高程误差和线性速度。

主要接口：

    unwrap_arcs_grid_periodogram(
        phase_wrapped,
        arcs,
        bperp,
        time_year,
        param
    )

输出 net 字典，可以直接接入后续：

    spatial_unwrap_and_solve_linear_periodogram()
    export_results()

相位模型：

    phi = h * h2ph + v * v2ph

其中：

    h2ph = 4*pi*Bperp / (lambda*R*sin(theta))
    v2ph = displacement_sign * 4*pi*time_year / lambda

如果你生成观测相位时使用：
    deformation_phase = -4*pi*d/lambda

则 displacement_sign = -1。

如果使用：
    deformation_phase =  4*pi*d/lambda

则 displacement_sign = 1。
"""

import numpy as np
import os
from multiprocessing import Pool, cpu_count


def wrap_phase(phase):
    """
    将相位缠绕到 [-pi, pi]
    """
    return np.angle(np.exp(1j * phase))


class GridPeriodogramTemporalUnwrapper(object):
    """
    传统二维网格 Periodogram 时间解缠类。

    param 示例：

    GRID_PERIODOGRAM_PARAM = {
        "sentinel-1": {
            "wavelength": 0.055465,
            "incidence_angle": 34.0,
            "H": 800000.0 * np.cos(np.deg2rad(34.0))
        },

        "search_range": {
            "h_range": [-20, 20],
            "h_step": 0.1,

            "v_range": [-0.05, 0.05],
            "v_step": 0.0005
        },

        "period_est_mode": "grid-period",
        "displacement_sign": 1
    }

    注意：
        h_range 单位：m
        h_step  单位：m

        v_range 单位：m/year
        v_step  单位：m/year
    """

    def __init__(self, param):
        self.param = param

        self.wavelength = self.param["sentinel-1"]["wavelength"]
        self.incidence_angle = (
            self.param["sentinel-1"]["incidence_angle"] * np.pi / 180.0
        )

        self.H = self.param["sentinel-1"]["H"]

        # 沿用你原始代码的写法：
        # R = H / cos(incidence_angle)
        self.R = self.H / np.cos(self.incidence_angle)

        self.displacement_sign = self.param.get("displacement_sign", 1)

        self.height_search_space = self._build_height_search_space()
        self.velocity_search_space = self._build_velocity_search_space()

    def _build_height_search_space(self):
        """
        构建高程误差搜索空间，单位 m。
        """
        h_start = self.param["search_range"]["h_range"][0]
        h_end = self.param["search_range"]["h_range"][1]
        h_step = self.param["search_range"]["h_step"]

        return np.arange(h_start, h_end + h_step, h_step, dtype=float)

    def _build_velocity_search_space(self):
        """
        构建速度搜索空间，单位 m/year。

        注意：
        你给的原始代码里这里写成了：

            v_start = h_range[0]
            v_end   = h_range[1]

        这个应该是笔误。这里改成 v_range。
        """
        if "v_range" not in self.param["search_range"]:
            raise ValueError(
                "search_range 中缺少 v_range。"
                "例如：'v_range': [-0.05, 0.05]"
            )

        v_start = self.param["search_range"]["v_range"][0]
        v_end = self.param["search_range"]["v_range"][1]
        v_step = self.param["search_range"]["v_step"]

        return np.arange(v_start, v_end + v_step, v_step, dtype=float)

    def compute_h2ph(self, bperp):
        """
        计算高程误差到相位的转换系数。

        h2ph(t) = 4*pi*Bperp(t)/(lambda*R*sin(theta))

        返回：
            h2ph: shape = (n_time,)
        """
        bperp = np.asarray(bperp, dtype=float)

        h2ph = (
            4.0
            * np.pi
            * bperp
            / (
                self.wavelength
                * self.R
                * np.sin(self.incidence_angle)
            )
        )

        return h2ph

    def compute_v2ph(self, time_year):
        """
        计算线性速度到相位的转换系数。

        如果 v 单位是 m/year，time_year 单位是 year：

            d(t) = v * time_year

        所以：

            phi_def(t) = displacement_sign * 4*pi*v*time_year/lambda

        因此：

            v2ph(t) = displacement_sign * 4*pi*time_year/lambda

        返回：
            v2ph: shape = (n_time,)
        """
        time_year = np.asarray(time_year, dtype=float)

        v2ph = (
            self.displacement_sign
            * 4.0
            * np.pi
            * time_year
            / self.wavelength
        )

        return v2ph

    @staticmethod
    def linear_search(exp_phase, p2ph, search_space):
        """
        一维搜索，用于 linear-period 模式。

        exp_phase:
            观测相位复数形式

        p2ph:
            参数到相位转换系数，shape = (n_time,)

        search_space:
            参数搜索空间
        """

        sub_e = np.exp(
            -1j * (
                p2ph.reshape(-1, 1)
                * search_space.reshape(1, -1)
            )
        )

        objective_function = np.mean(
            exp_phase.reshape(-1, 1) * sub_e,
            axis=0
        )

        result = search_space[np.argmax(np.abs(objective_function))]

        return result

    def linear_periodogram(self, arc_phase, h2ph, v2ph):
        """
        一维连续搜索 Periodogram。

        这里保留为可选功能，不作为主要对比算法。
        """

        arc_phase = np.asarray(arc_phase, dtype=float)

        delta_phase_h = (
            self.height_search_space.reshape(-1, 1)
            * h2ph.reshape(1, -1)
        )

        sub_phase_h = arc_phase.reshape(1, -1) - delta_phase_h

        e_phase_sub_h = np.sum(np.exp(1j * sub_phase_h), axis=0)

        v_est = self.linear_search(
            e_phase_sub_h,
            v2ph,
            self.velocity_search_space
        )

        phase_sub_v = arc_phase - v_est * v2ph

        e_phase_sub_v = np.exp(1j * phase_sub_v)

        h_est = self.linear_search(
            e_phase_sub_v,
            h2ph,
            self.height_search_space
        )

        return h_est, v_est

    def grid_periodogram(self, arc_phase, h2ph, v2ph):
        """
        二维网格 Periodogram 搜索。

        输入：
            arc_phase: shape = (n_time,)
                单条 arc 的差分缠绕观测相位

            h2ph: shape = (n_time,)
                高程误差到相位转换系数

            v2ph: shape = (n_time,)
                速度到相位转换系数

        输出：
            h_est: float
                arc 上的相对高程误差，单位 m

            v_est: float
                arc 上的相对线性速度，单位 m/year

            gamma_max: float
                最大 temporal coherence
        """

        arc_phase = np.asarray(arc_phase, dtype=float)

        n_time = arc_phase.size
        n_h = self.height_search_space.size
        n_v = self.velocity_search_space.size

        # ----------------------------------------------------
        # 构建二维搜索相位空间
        #
        # search_phase_space shape:
        #     (n_time, n_v * n_h)
        #
        # 参数排列顺序：
        #     velocity 外层，height 内层
        # ----------------------------------------------------

        velocity_phase = (
            v2ph.reshape(-1, 1)
            * self.velocity_search_space.reshape(1, -1)
        )

        height_phase = (
            h2ph.reshape(-1, 1)
            * self.height_search_space.reshape(1, -1)
        )

        search_phase_space = (
            np.repeat(velocity_phase, n_h, axis=1)
            + np.tile(height_phase, (1, n_v))
        )

        # ----------------------------------------------------
        # 计算目标函数
        # ----------------------------------------------------
        sub_phase = arc_phase.reshape(-1, 1) - search_phase_space

        gamma_temporal = np.sum(
            np.exp(1j * sub_phase),
            axis=0
        ) / float(n_time)

        abs_gamma = np.abs(gamma_temporal)

        max_index = np.argmax(abs_gamma)

        v_index, h_index = np.unravel_index(
            max_index,
            (n_v, n_h),
            order="C"
        )

        h_est = self.height_search_space[h_index]
        v_est = self.velocity_search_space[v_index]
        gamma_max = abs_gamma[max_index]

        return h_est, v_est, gamma_max

    def estimate(self, arc_phase, bperp, time_year, mode="grid-period"):
        """
        单条 arc 的 Periodogram 估计接口。

        输入：
            arc_phase:
                arc 差分缠绕相位，shape = (n_time,)

            bperp:
                垂直基线，shape = (n_time,)

            time_year:
                时间基线，shape = (n_time,)

            mode:
                "grid-period" 或 "linear-period"

        输出：
            h_est, v_est, gamma_max
        """

        arc_phase = np.asarray(arc_phase, dtype=float)
        bperp = np.asarray(bperp, dtype=float)
        time_year = np.asarray(time_year, dtype=float)

        if arc_phase.ndim != 1:
            raise ValueError("arc_phase 必须是一维数组。")

        if bperp.ndim != 1:
            raise ValueError("bperp 必须是一维数组。")

        if time_year.ndim != 1:
            raise ValueError("time_year 必须是一维数组。")

        if not (
            arc_phase.size == bperp.size == time_year.size
        ):
            raise ValueError("arc_phase、bperp、time_year 长度必须一致。")

        h2ph = self.compute_h2ph(bperp)
        v2ph = self.compute_v2ph(time_year)

        if mode == "grid-period":
            h_est, v_est, gamma_max = self.grid_periodogram(
                arc_phase,
                h2ph,
                v2ph
            )

        elif mode == "linear-period":
            h_est, v_est = self.linear_periodogram(
                arc_phase,
                h2ph,
                v2ph
            )

            model_phase = h_est * h2ph + v_est * v2ph
            gamma_max = np.abs(
                np.mean(np.exp(1j * (arc_phase - model_phase)))
            )

        else:
            raise ValueError("mode 只能是 'grid-period' 或 'linear-period'。")

        return h_est, v_est, gamma_max


def _grid_periodogram_worker(args):
    """
    单条 arc 的 grid-periodogram 计算函数。

    作用：
        1. 计算 arc 的差分缠绕相位
        2. 用 grid-periodogram 估计 h_est 和 v_est
        3. 用线性模型相位反推整数模糊度 k
        4. 用 k 修正原始缠绕相位，得到真正的 arc 解缠相位
    """

    idx, p1, p2, phase_wrapped, bperp, time_year, param, mode = args

    unwrapper = GridPeriodogramTemporalUnwrapper(param)

    h2ph = unwrapper.compute_h2ph(bperp)
    v2ph = unwrapper.compute_v2ph(time_year)

    # --------------------------------------------------------
    # 1. arc 差分缠绕相位
    # --------------------------------------------------------
    diff_phase = wrap_phase(
        phase_wrapped[p2, :] - phase_wrapped[p1, :]
    )

    # --------------------------------------------------------
    # 2. grid-periodogram 估计 h 和 v
    # --------------------------------------------------------
    h_est, v_est, gamma_max = unwrapper.estimate(
        arc_phase=diff_phase,
        bperp=bperp,
        time_year=time_year,
        mode=mode
    )

    # --------------------------------------------------------
    # 3. 根据 h 和 v 构造线性模型相位
    # --------------------------------------------------------
    diff_phase_model = h_est * h2ph + v_est * v2ph

    # --------------------------------------------------------
    # 4. 根据模型相位反推整数模糊度
    # --------------------------------------------------------
    k = np.round(
        (diff_phase_model - diff_phase) / (2.0 * np.pi)
    ).astype(int)

    # --------------------------------------------------------
    # 5. 用整数模糊度修正原始观测缠绕相位
    # --------------------------------------------------------
    diff_phase_unwrapped = diff_phase + 2.0 * np.pi * k

    # --------------------------------------------------------
    # 6. 用解缠相位和模型相位之间的差异评价质量
    # --------------------------------------------------------
    residual = diff_phase_unwrapped - diff_phase_model

    return {
        "idx": idx,
        "diff_phase": diff_phase,
        "diff_phase_model": diff_phase_model,
        "diff_phase_unwrapped": diff_phase_unwrapped,
        "k": k,
        "h_est": h_est,
        "v_est": v_est,
        "gamma_max": gamma_max,
        "res_std": np.std(residual),
    }

def unwrap_arcs_grid_periodogram(
    phase_wrapped,
    arcs,
    bperp,
    time_year,
    param,
    mode="grid-period",
    n_jobs=1,
    chunk_size=50
):
    """
    批量 arc 时间域解缠接口，支持多 CPU 并行。

    输入：
        phase_wrapped: ndarray, shape = (n_points, n_time)
            每个 PS 点的缠绕观测相位

        arcs: ndarray, shape = (n_arcs, 2)
            Delaunay 网络边，每行是 [p1, p2]

        bperp: ndarray, shape = (n_time,)
            垂直基线，单位 m

        time_year: ndarray, shape = (n_time,)
            时间基线，单位 year

        param: dict
            搜索参数和雷达参数

        mode:
            "grid-period" 或 "linear-period"

        n_jobs:
            使用的 CPU 数量。
            n_jobs=1 表示单进程。
            n_jobs=-1 表示使用所有 CPU。

        chunk_size:
            multiprocessing 的 chunksize。
            arc 很多时可以设为 50、100、200。

    输出：
        net: dict
            可直接接入后续空间网络求解。
    """

    phase_wrapped = np.asarray(phase_wrapped, dtype=float)
    arcs = np.asarray(arcs, dtype=int)
    bperp = np.asarray(bperp, dtype=float)
    time_year = np.asarray(time_year, dtype=float)

    n_points, n_time = phase_wrapped.shape
    n_arcs = arcs.shape[0]

    if bperp.size != n_time:
        raise ValueError("bperp 长度必须和相位时相数一致。")

    if time_year.size != n_time:
        raise ValueError("time_year 长度必须和相位时相数一致。")

    if n_jobs == -1:
        n_jobs = cpu_count()

    if n_jobs < 1:
        n_jobs = 1

    print("开始 grid-periodogram 时间解缠")
    print("点数:", n_points)
    print("arc 数量:", n_arcs)
    print("时相数:", n_time)
    print("使用 CPU 数:", n_jobs)

    arc_phase_wrapped = np.zeros((n_arcs, n_time), dtype=float)
    arc_phase_unwrapped = np.zeros((n_arcs, n_time), dtype=float)
    arc_ambiguity = np.zeros((n_arcs, n_time), dtype=int)

    arc_delta_h = np.zeros(n_arcs, dtype=float)
    arc_delta_v = np.zeros(n_arcs, dtype=float)
    arc_gamma = np.zeros(n_arcs, dtype=float)
    arc_res_std = np.zeros(n_arcs, dtype=float)

    # --------------------------------------------------------
    # 单进程模式
    # --------------------------------------------------------
    if n_jobs == 1:
        unwrapper = GridPeriodogramTemporalUnwrapper(param)

        h2ph = unwrapper.compute_h2ph(bperp)
        v2ph = unwrapper.compute_v2ph(time_year)

        for idx in range(n_arcs):
            if idx % 1000 == 0:
                print("已处理 arc: {}/{}".format(idx, n_arcs))

            p1 = arcs[idx, 0]
            p2 = arcs[idx, 1]

            diff_phase = wrap_phase(
                phase_wrapped[p2, :] - phase_wrapped[p1, :]
            )

            h_est, v_est, gamma_max = unwrapper.estimate(
                arc_phase=diff_phase,
                bperp=bperp,
                time_year=time_year,
                mode=mode
            )

            diff_phase_model = h_est * h2ph + v_est * v2ph

            # 根据线性模型反推整数模糊度
            k = np.round(
                (diff_phase_model - diff_phase) / (2.0 * np.pi)
            ).astype(int)

            # 用整数模糊度修正原始观测缠绕相位
            diff_phase_unwrapped = diff_phase + 2.0 * np.pi * k

            # 残差仍然用模型和解缠结果之间的差值评价
            residual = diff_phase_unwrapped - diff_phase_model

            arc_phase_wrapped[idx, :] = diff_phase
            arc_phase_unwrapped[idx, :] = diff_phase_unwrapped
            arc_ambiguity[idx, :] = k

            arc_delta_h[idx] = h_est
            arc_delta_v[idx] = v_est
            arc_gamma[idx] = gamma_max
            arc_res_std[idx] = np.std(residual)

    # --------------------------------------------------------
    # 多进程模式
    # --------------------------------------------------------
    else:
        tasks = [
            (
                idx,
                int(arcs[idx, 0]),
                int(arcs[idx, 1]),
                phase_wrapped,
                bperp,
                time_year,
                param,
                mode,
            )
            for idx in range(n_arcs)
        ]

        processed = 0

        with Pool(processes=n_jobs) as pool:
            for res in pool.imap_unordered(
                _grid_periodogram_worker,
                tasks,
                chunksize=chunk_size
            ):
                idx = res["idx"]

                arc_phase_wrapped[idx, :] = res["diff_phase"]

                # 这里必须使用：
                #   arc_phase_wrapped + 2π * arc_ambiguity
                # 得到的真正解缠相位
                arc_phase_unwrapped[idx, :] = res["diff_phase_unwrapped"]

                # 保存整数模糊度
                arc_ambiguity[idx, :] = res["k"]

                arc_delta_h[idx] = res["h_est"]
                arc_delta_v[idx] = res["v_est"]
                arc_gamma[idx] = res["gamma_max"]
                arc_res_std[idx] = res["res_std"]

                processed += 1

                if processed % 1000 == 0:
                    print("已处理 arc: {}/{}".format(processed, n_arcs))

    net = {
        "arcs": arcs,
        "arc_phase_wrapped": arc_phase_wrapped,
        "arc_phase_unwrapped": arc_phase_unwrapped,
        "arc_ambiguity": arc_ambiguity,
        "arc_delta_h": arc_delta_h,
        "arc_delta_v": arc_delta_v,
        "arc_gamma": arc_gamma,
        "arc_res_std": arc_res_std,
    }

    print("grid-periodogram 时间解缠完成")
    print("arc_delta_h 范围: {:.4f} ~ {:.4f} m".format(
        np.min(arc_delta_h), np.max(arc_delta_h)
    ))
    print("arc_delta_v 范围: {:.6f} ~ {:.6f} m/year".format(
        np.min(arc_delta_v), np.max(arc_delta_v)
    ))

    return net