#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_simulated_phase_unwrap.py

用于仿真实验的 PS-InSAR 简化解缠流程：

1. 读取已经生成好的观测相位 CSV
2. 从 observed_phase_txx_rad 读取缠绕观测相位
3. 从基线 CSV 读取 Bperp，并计算 K_h
4. 构建 Delaunay 网络
5. 对每条 arc 做时间方向解缠
6. 通过网络积分恢复每个 PS 点的解缠相位
7. 分离高程误差和形变量
8. 导出估计结果 CSV

"""

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from periodogram_temporal_unwrap import unwrap_arcs_periodogram


# ============================================================
# 1. 基础函数
# ============================================================

def wrap_phase(phi):
    """
    将相位缠绕到 [-pi, pi]
    """
    return np.angle(np.exp(1j * phi))


def get_time_columns(df, prefix, suffix):
    """
    根据字段名前缀和后缀提取时间序列列名，并按 t01, t02, ... 排序。

    例如：
        observed_phase_t01_rad
        observed_phase_t02_rad
        ...
    """
    pattern = re.compile(r"{}t(\d+)_{}".format(prefix, suffix))

    cols = []
    for col in df.columns:
        m = pattern.match(col)
        if m is not None:
            cols.append((int(m.group(1)), col))

    cols = sorted(cols, key=lambda x: x[0])
    return [c for _, c in cols]


def read_baseline_csv(baseline_csv, n_time):
    """
    读取垂直基线 CSV。

    你的基线 CSV 格式：
        bperp_m,image_id,time_day,time_year

    返回：
        Bperp: shape = (n_time,)
        t_years: shape = (n_time,)
    """

    df_base = pd.read_csv(baseline_csv)

    if "bperp_m" not in df_base.columns:
        raise ValueError("基线 CSV 中缺少 bperp_m 字段。")

    if "time_year" not in df_base.columns:
        raise ValueError("基线 CSV 中缺少 time_year 字段。")

    Bperp = df_base["bperp_m"].values.astype(float)
    t_years = df_base["time_year"].values.astype(float)

    if len(Bperp) != n_time:
        raise ValueError(
            "基线数量与观测相位时相数量不一致。Bperp 数量 = {}, 相位时相数量 = {}".format(
                len(Bperp), n_time
            )
        )

    if len(t_years) != n_time:
        raise ValueError(
            "time_year 数量与观测相位时相数量不一致。time_year 数量 = {}, 相位时相数量 = {}".format(
                len(t_years), n_time
            )
        )


    print("读取基线 CSV 完成")
    print("基线格式: bperp_m,image_id,time_day,time_year")
    print("Bperp shape:", Bperp.shape)
    print("time_year shape:", t_years.shape)
    print("Bperp 范围: {:.4f} ~ {:.4f} m".format(np.min(Bperp), np.max(Bperp)))
    print("time_year 范围: {:.6f} ~ {:.6f} year".format(np.min(t_years), np.max(t_years)))

    return Bperp, t_years


def compute_kh_from_bperp(Bperp, wavelength, R, inc_angle_deg):
    """
    根据垂直基线计算高程误差相位系数 K_h。

    公式：
        K_h = 4 * pi * Bperp / (lambda * R * sin(theta))

    单位：
        Bperp: m
        wavelength: m
        R: m
        theta: rad
        K_h: rad/m
    """

    inc_angle = np.deg2rad(inc_angle_deg)

    K_h = 4.0 * np.pi * Bperp / (
        wavelength * R * np.sin(inc_angle)
    )

    print("K_h 计算完成")
    print("K_h shape:", K_h.shape)
    print("K_h 范围: {:.8f} ~ {:.8f} rad/m".format(np.min(K_h), np.max(K_h)))

    return K_h


# ============================================================
# 2. 读取模拟观测相位 CSV
# ============================================================

def read_observed_phase_csv(
    csv_file,
    baseline_csv,
    wavelength=0.056,
    R=800000.0,
    inc_angle_deg=35.0
):
    """
    读取你的观测相位 CSV 和基线 CSV。

    观测相位 CSV 必须包含：
        point_id
        x
        y
        delta_h_true_m
        observed_phase_t01_rad ... observed_phase_t60_rad

    基线 CSV 必须包含垂直基线列，例如：
        bperp_m

    """

    df = pd.read_csv(csv_file)

    required_cols = ["x", "y", "delta_h_true_m"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError("CSV 中缺少必要字段: {}".format(col))

    observed_cols = get_time_columns(
        df,
        prefix="observed_phase_",
        suffix="rad"
    )

    deformation_cols = get_time_columns(
        df,
        prefix="deformation_",
        suffix="mm"
    )

    height_phase_cols = get_time_columns(
        df,
        prefix="height_phase_",
        suffix="rad"
    )

    if len(observed_cols) == 0:
        raise ValueError("没有找到 observed_phase_txx_rad 字段。")

    n_time = len(observed_cols)

    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)

    # 观测相位：已经是缠绕后的相位
    phase_wrapped = df[observed_cols].values.astype(float)

    # 从基线 CSV 读取 Bperp 和 time_year
    Bperp, t_years = read_baseline_csv(
        baseline_csv=baseline_csv,
        n_time=n_time
    )

    # 根据 Bperp 计算 K_h
    K_h = compute_kh_from_bperp(
        Bperp=Bperp,
        wavelength=wavelength,
        R=R,
        inc_angle_deg=inc_angle_deg
    )

    true_def_mm = None
    if len(deformation_cols) == n_time:
        true_def_mm = df[deformation_cols].values.astype(float)

    data = {
        "df": df,
        "x": x,
        "y": y,
        "phase_wrapped": phase_wrapped,
        "Bperp": Bperp,
        "K_h": K_h,
        "t_years": t_years,
        "wavelength": wavelength,
        "R": R,
        "inc_angle_deg": inc_angle_deg,
        "observed_cols": observed_cols,
        "height_phase_cols": height_phase_cols,
        "deformation_cols": deformation_cols,
        "true_def_mm": true_def_mm,
    }

    print("读取观测相位 CSV 完成")
    print("点数:", len(x))
    print("时相数:", n_time)

    return data


# ============================================================
# 3. 构建 Delaunay 网络
# ============================================================

def create_delaunay_network(x, y, max_dist=None):
    """
    根据 PS 点坐标构建 Delaunay 网络。
    """

    points = np.column_stack([x, y])

    tri = Delaunay(points)

    edges = set()

    for simplex in tri.simplices:
        i, j, k = simplex

        pairs = [
            (i, j),
            (j, k),
            (k, i),
        ]

        for a, b in pairs:
            if a > b:
                a, b = b, a
            edges.add((int(a), int(b)))

    arcs = np.array(sorted(edges), dtype=int)

    if max_dist is not None:
        dist = np.linalg.norm(points[arcs[:, 1]] - points[arcs[:, 0]], axis=1)
        arcs = arcs[dist <= max_dist]

    print("Delaunay 网络构建完成")
    print("arc 数量:", arcs.shape[0])

    return arcs


# ============================================================
# 4. 时间解缠：对每条 arc 沿时间方向解缠
# ============================================================

# 不同的算法放在不同的函数里，方便后续替换和比较。

# ============================================================
# 5. 选择参考点
# ============================================================

def choose_reference_point(x, y, method="center", ref_idx=None):
    """
    选择参考点。
    """

    if method == "manual":
        if ref_idx is None:
            raise ValueError("method='manual' 时必须给出 ref_idx")
        return int(ref_idx)

    if method == "first":
        return 0

    if method == "center":
        cx = np.mean(x)
        cy = np.mean(y)

        dist2 = (x - cx) ** 2 + (y - cy) ** 2

        return int(np.argmin(dist2))

    raise ValueError("未知参考点选择方法: {}".format(method))


# ============================================================
# 6. 网络积分：由 arc 相对量恢复 PS 点量
# ============================================================

def build_incidence_matrix(arcs, n_points, ref_idx):
    """
    构建设计矩阵 A。
    """

    n_arcs = arcs.shape[0]

    rows = []
    cols = []
    vals = []

    full_to_reduced = -np.ones(n_points, dtype=int)

    reduced_idx = 0
    for i in range(n_points):
        if i == ref_idx:
            continue
        full_to_reduced[i] = reduced_idx
        reduced_idx += 1

    for row in range(n_arcs):
        p1 = arcs[row, 0]
        p2 = arcs[row, 1]

        if p1 != ref_idx:
            rows.append(row)
            cols.append(full_to_reduced[p1])
            vals.append(-1.0)

        if p2 != ref_idx:
            rows.append(row)
            cols.append(full_to_reduced[p2])
            vals.append(1.0)

    A = coo_matrix(
        (vals, (rows, cols)),
        shape=(n_arcs, n_points - 1)
    ).tocsr()

    return A


def integrate_arc_values(arcs, arc_values, n_points, ref_idx, weights=None):
    """
    将 arc 上的相对值积分成 PS 点上的值。
    """

    A = build_incidence_matrix(arcs, n_points, ref_idx)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.nanmax(weights)

        sqrt_w = np.sqrt(weights)

        A_used = A.multiply(sqrt_w[:, None])
    else:
        sqrt_w = None
        A_used = A

    arc_values = np.asarray(arc_values, dtype=float)

    if arc_values.ndim == 1:
        if sqrt_w is not None:
            b = arc_values * sqrt_w
        else:
            b = arc_values

        sol_reduced = lsqr(A_used, b)[0]

        point_values = np.zeros(n_points, dtype=float)
        mask = np.arange(n_points) != ref_idx
        point_values[mask] = sol_reduced
        point_values[ref_idx] = 0.0

        return point_values

    n_time = arc_values.shape[1]
    point_values = np.zeros((n_points, n_time), dtype=float)

    mask = np.arange(n_points) != ref_idx

    for t in range(n_time):
        if sqrt_w is not None:
            b = arc_values[:, t] * sqrt_w
        else:
            b = arc_values[:, t]

        sol_reduced = lsqr(A_used, b)[0]

        point_values[mask, t] = sol_reduced
        point_values[ref_idx, t] = 0.0

    return point_values


# ============================================================
# 7. 空间解缠 / 网络求解
# ============================================================

def spatial_unwrap_and_solve(data, net, ref_idx):
    """
    空间解缠 + 网络求解。

    非线性形变版本：

    1. 根据 arc 上的时间解缠相位，恢复每个 PS 点的解缠相位
    2. 根据 arc_delta_h，恢复每个 PS 点的高程误差
    3. 从 PS 点解缠相位中扣除高程误差相位
    4. 得到每个 PS 点的非线性形变时间序列
    """

    arcs = net["arcs"]
    arc_phase_unwrapped = net["arc_phase_unwrapped"]
    arc_delta_h = net["arc_delta_h"]
    arc_res_std = net["arc_res_std"]

    K_h = data["K_h"]
    wavelength = data["wavelength"]

    n_points = data["x"].shape[0]

    weights = 1.0 / (arc_res_std ** 2 + 1e-6)

    # --------------------------------------------------------
    # 1. arc 解缠相位 -> PS 点解缠相位
    # --------------------------------------------------------
    ps_phase_unwrapped = integrate_arc_values(
        arcs=arcs,
        arc_values=arc_phase_unwrapped,
        n_points=n_points,
        ref_idx=ref_idx,
        weights=weights
    )

    ps_phase_unwrapped = ps_phase_unwrapped - ps_phase_unwrapped[ref_idx, :]

    # --------------------------------------------------------
    # 2. arc 高程误差差值 -> PS 点高程误差
    # --------------------------------------------------------
    h_est = integrate_arc_values(
        arcs=arcs,
        arc_values=arc_delta_h,
        n_points=n_points,
        ref_idx=ref_idx,
        weights=weights
    )

    h_est = h_est - h_est[ref_idx]

    # --------------------------------------------------------
    # 3. 去除高程误差相位
    # --------------------------------------------------------
    height_phase_est = h_est.reshape(-1, 1) * K_h.reshape(1, -1)

    deformation_phase = ps_phase_unwrapped - height_phase_est

    # --------------------------------------------------------
    # 4. 形变相位 -> 形变量
    # --------------------------------------------------------
    displacement_sign = data.get("displacement_sign", 1)

    ph2m = wavelength / (displacement_sign * 4.0 * np.pi)

    deformation_m = deformation_phase * ph2m

    deformation_m = deformation_m - deformation_m[ref_idx, :]

    result = {
        "ps_phase_unwrapped": ps_phase_unwrapped,
        "h_est_m": h_est,
        "v_est_m_per_year": np.zeros(n_points),
        "deformation_m": deformation_m,
        "residual_phase": np.zeros_like(ps_phase_unwrapped),
        "ref_idx": ref_idx,
    }

    print("空间解缠和非线性网络求解完成")
    print("h_est 范围: {:.4f} ~ {:.4f} m".format(
        np.min(h_est), np.max(h_est)
    ))
    print("deformation 范围: {:.4f} ~ {:.4f} mm".format(
        np.min(deformation_m * 1000.0),
        np.max(deformation_m * 1000.0)
    ))

    return result

# ============================================================
# 8. 结果导出
# ============================================================

def export_results(data, net, result, output_csv, network_csv=None):
    """
    导出 PS 点估计结果和网络 arc 结果。
    """

    df = data["df"]
    deformation_m = result["deformation_m"]
    ps_phase_unwrapped = result["ps_phase_unwrapped"]

    n_points, n_time = deformation_m.shape

    out = pd.DataFrame()

    for col in ["point_id", "type", "x", "y", "delta_h_true_m", "final_deformation_mm"]:
        if col in df.columns:
            out[col] = df[col]

    out["ref_idx"] = result["ref_idx"]

    out["h_est_m_relative"] = result["h_est_m"]
    out["v_est_m_per_year_relative"] = result["v_est_m_per_year"]
    out["v_est_mm_per_year_relative"] = result["v_est_m_per_year"] * 1000.0

    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        h_true_rel = h_true - h_true[result["ref_idx"]]

        out["delta_h_true_relative_m"] = h_true_rel
        out["delta_h_error_m"] = out["h_est_m_relative"] - h_true_rel

    true_def_cols = data["deformation_cols"]
    for col in true_def_cols:
        out[col] = df[col]

    for t in range(n_time):
        time_id = t + 1

        out["phase_unwrapped_est_t{:02d}_rad".format(time_id)] = ps_phase_unwrapped[:, t]
        out["deformation_est_t{:02d}_m".format(time_id)] = deformation_m[:, t]
        out["deformation_est_t{:02d}_mm".format(time_id)] = deformation_m[:, t] * 1000.0

    if data["true_def_mm"] is not None:
        true_def_mm = data["true_def_mm"]

        true_def_rel_mm = true_def_mm - true_def_mm[result["ref_idx"], :]

        for t in range(n_time):
            time_id = t + 1
            est_mm = deformation_m[:, t] * 1000.0
            err_mm = est_mm - true_def_rel_mm[:, t]

            out["deformation_true_relative_t{:02d}_mm".format(time_id)] = true_def_rel_mm[:, t]
            out["deformation_error_t{:02d}_mm".format(time_id)] = err_mm

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("PS 点结果已导出:")
    print(output_csv)

    if network_csv is not None:
        arc_out = pd.DataFrame()
        arc_out["p1"] = net["arcs"][:, 0]
        arc_out["p2"] = net["arcs"][:, 1]
        arc_out["arc_delta_h_m"] = net["arc_delta_h"]

        if "arc_delta_v" in net:
            arc_out["arc_delta_v_m_per_year"] = net["arc_delta_v"]
            arc_out["arc_delta_v_mm_per_year"] = net["arc_delta_v"] * 1000.0

        arc_out["arc_res_std_rad"] = net["arc_res_std"]

        os.makedirs(os.path.dirname(network_csv), exist_ok=True)
        arc_out.to_csv(network_csv, index=False)

        print("Delaunay 网络结果已导出:")
        print(network_csv)


# ============================================================
# 9. 精度评价
# ============================================================

def print_accuracy_report(data, result):
    """
    根据 CSV 中保存的真值，输出整体精度评价和不同形变类型的精度评价。
    """

    df = data["df"]
    ref_idx = result["ref_idx"]

    print("\n================ 精度评价 ================")

    # ============================================================
    # 1. 高程误差整体精度
    # ============================================================
    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        h_true_rel = h_true - h_true[ref_idx]

        h_est = result["h_est_m"]
        h_err = h_est - h_true_rel

        rmse_h = np.sqrt(np.mean(h_err ** 2))
        mae_h = np.mean(np.abs(h_err))

        print("高程误差 RMSE: {:.6f} m".format(rmse_h))
        print("高程误差 MAE : {:.6f} m".format(mae_h))

    # ============================================================
    # 2. 形变量整体精度
    # ============================================================
    if data["true_def_mm"] is not None:
        true_def = data["true_def_mm"]
        true_def_rel = true_def - true_def[ref_idx, :]

        est_def = result["deformation_m"] * 1000.0

        def_err = est_def - true_def_rel

        rmse_def = np.sqrt(np.mean(def_err ** 2))
        mae_def = np.mean(np.abs(def_err))

        print("形变量 RMSE: {:.6f} mm".format(rmse_def))
        print("形变量 MAE : {:.6f} mm".format(mae_def))

    print("参考点 index:", ref_idx)
    print("==========================================\n")

    # ============================================================
    # 3. 不同形变类型精度评价
    # ============================================================
    if "type" not in df.columns:
        print("CSV 中没有 type 字段，无法按形变类型统计精度。")
        return

    print("\n============ 不同形变类型精度评价 ============")

    deformation_types = df["type"].unique()

    for deformation_type in deformation_types:

        mask = df["type"].values == deformation_type
        n_points_type = np.sum(mask)

        print("\n---------- {} ----------".format(deformation_type))
        print("点数: {}".format(n_points_type))

        # ========================================================
        # 3.1 当前类型的高程误差精度
        # ========================================================
        if "delta_h_true_m" in df.columns:

            h_err_type = h_err[mask]

            rmse_h_type = np.sqrt(np.mean(h_err_type ** 2))
            mae_h_type = np.mean(np.abs(h_err_type))

            print("高程误差 RMSE: {:.6f} m".format(rmse_h_type))
            print("高程误差 MAE : {:.6f} m".format(mae_h_type))

        # ========================================================
        # 3.2 当前类型的形变量精度
        # ========================================================
        if data["true_def_mm"] is not None:

            def_err_type = def_err[mask, :]

            rmse_def_type = np.sqrt(np.mean(def_err_type ** 2))
            mae_def_type = np.mean(np.abs(def_err_type))

            print("形变量 RMSE: {:.6f} mm".format(rmse_def_type))
            print("形变量 MAE : {:.6f} mm".format(mae_def_type))

    print("\n============================================\n")


# ============================================================
# 10. 主函数：参数直接写在这里
# ============================================================

def main():
    """
    主函数。

    你只需要改这里的 INPUT_CSV、BASELINE_CSV、OUTPUT_DIR 等参数。
    """

    # --------------------------------------------------------
    # 1. 输入观测相位 CSV 地址
    # --------------------------------------------------------
    INPUT_CSV = "/data/test/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/2_generate_observed_phase/observed_phase_csv/observed_phase_noise_0p2rad.csv"

    # --------------------------------------------------------
    # 2. 输入基线 CSV 地址
    # --------------------------------------------------------
    BASELINE_CSV = "/data/test/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/2_generate_observed_phase/observed_phase_csv/simulation_baseline_80_images.csv"

    # --------------------------------------------------------
    # 3. 输出目录
    # --------------------------------------------------------
    OUTPUT_DIR = "/data/test/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/4_result_csv"

    OUTPUT_CSV = os.path.join(
        OUTPUT_DIR,
        "simulated_unwrapped_noise_0p2rad_result.csv"
    )

    NETWORK_CSV = os.path.join(
        OUTPUT_DIR,
        "simulated_delaunay_network.csv"
    )



    PERIODOGRAM_PARAM = {
        "sentinel-1": {
            "wavelength": 0.056,  # m
            "incidence_angle": 35.0,

            # 注意：
            # 这里沿用你原始代码的写法：
            # R = H / cos(incidence_angle)
            #
            # 如果你想让斜距 R = 800000 m，
            # 那么 H 应该设置成：
            # H = R * cos(incidence_angle)
            "H": 800000.0 * np.cos(np.deg2rad(35.0))
        },

        "search_range": {
            # 高程误差搜索范围，单位 m
            "h_range": [-30, 30],

            # 搜索步长，单位 m
            "h_step": 0.1
        },

        # 迭代次数
        "iterative_times": 3,

        # 形变相位符号
        # 如果生成观测相位时用的是：
        #     deformation_phase =  4*pi*d/lambda
        # 设置为 1
        #
        # 如果生成观测相位时用的是：
        #     deformation_phase = -4*pi*d/lambda
        # 设置为 -1
        "displacement_sign": 1
    }

    # --------------------------------------------------------
    # 4. 雷达和几何参数
    # --------------------------------------------------------
    WAVELENGTH = 0.056      # m
    SLANT_RANGE = 800000.0  # m
    INC_ANGLE_DEG = 35.0    # degree

    # --------------------------------------------------------
    # 5. Delaunay 网络参数
    # --------------------------------------------------------
    MAX_ARC_DIST = None

    # --------------------------------------------------------
    # 6. 参考点设置
    # --------------------------------------------------------
    REF_METHOD = "manual"
    REF_IDX = 11341

    # --------------------------------------------------------
    # 开始运行
    # --------------------------------------------------------
    data = read_observed_phase_csv(
        csv_file=INPUT_CSV,
        baseline_csv=BASELINE_CSV,
        wavelength=WAVELENGTH,
        R=SLANT_RANGE,
        inc_angle_deg=INC_ANGLE_DEG
    )

    arcs = create_delaunay_network(
        x=data["x"],
        y=data["y"],
        max_dist=MAX_ARC_DIST
    )

    net = unwrap_arcs_periodogram(
        phase_wrapped=data["phase_wrapped"],
        arcs=arcs,
        bperp=data["Bperp"],
        param=PERIODOGRAM_PARAM
    )

    ref_idx = choose_reference_point(
        x=data["x"],
        y=data["y"],
        method=REF_METHOD,
        ref_idx=REF_IDX
    )

    print("参考点 index:", ref_idx)

    result = spatial_unwrap_and_solve(
        data=data,
        net=net,
        ref_idx=ref_idx
    )

    export_results(
        data=data,
        net=net,
        result=result,
        output_csv=OUTPUT_CSV,
        network_csv=NETWORK_CSV
    )

    print_accuracy_report(
        data=data,
        result=result
    )

    print("全部完成。")


if __name__ == "__main__":
    main()