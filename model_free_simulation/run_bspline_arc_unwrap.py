#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized, lsqr


# ============================================================
# 1. Parameters
# ============================================================

PHASE_NOISE_STD_LIST = [0.2, 0.5, 0.8]
HEIGHT_STD = 20.0
HEIGHT_STEP = 0.1
ITERATIVE_TIMES = 3
SPLINE_DEGREE = 3
SPLINE_LAMBDA = 0.05
MAX_AMBIGUITY_REFINE = 30
DISPLACEMENT_SIGN = 1
BATCH_SIZE = 512
MAX_ARC_DIST = None
PLOT_RANDOM_SEED = 42
OUTPUT_FOLDER = "bspline_arc_unwrap_results"


# ============================================================
# 2. Basic functions
# ============================================================

def wrap_phase(phase):
    return np.angle(np.exp(1j * phase))

def get_time_columns(df, prefix, suffix):
    pattern = re.compile(r"{}t(\d+)_{}".format(re.escape(prefix), re.escape(suffix)))
    cols = []
    for col in df.columns:
        match = pattern.fullmatch(col)
        if match is not None: cols.append((int(match.group(1)), col))
    return [col for _, col in sorted(cols, key=lambda x: x[0])]


# ============================================================
# 3. Locate input files
# ============================================================

def locate_input_files():
    cwd = os.getcwd()
    candidate_dirs = [os.path.join(cwd, "observed_phase_csv"), cwd]

    for input_dir in candidate_dirs:
        observed_files = []
        for noise_std in PHASE_NOISE_STD_LIST:
            tag = str(noise_std).replace(".", "p")
            path = os.path.join(input_dir, "observed_phase_noise_{}rad.csv".format(tag))
            if os.path.isfile(path): observed_files.append((noise_std, path))

        baseline_files = sorted(glob.glob(os.path.join(input_dir, "simulation_baseline_*_images.csv")))
        if len(observed_files) == len(PHASE_NOISE_STD_LIST) and len(baseline_files) > 0:
            return input_dir, observed_files, baseline_files[0]

    raise IOError("Cannot find observed phase CSV files and baseline CSV.")


# ============================================================
# 4. Read baseline information
# ============================================================

def read_baseline_csv(baseline_csv, n_time):
    df = pd.read_csv(baseline_csv)
    if "bperp_m" not in df.columns: raise ValueError("Baseline CSV does not contain bperp_m.")
    if len(df) != n_time: raise ValueError("Baseline number {} != phase acquisition number {}.".format(len(df), n_time))

    bperp = df["bperp_m"].values.astype(float)
    time_year = df["time_year"].values.astype(float) if "time_year" in df.columns else np.arange(n_time, dtype=float) / max(n_time - 1, 1)
    wavelength = float(df["wavelength_m"].iloc[0]) if "wavelength_m" in df.columns else 0.0555
    incidence_angle_deg = float(df["incidence_angle_deg"].iloc[0]) if "incidence_angle_deg" in df.columns else 35.0
    slant_range = float(df["slant_range_m"].iloc[0]) if "slant_range_m" in df.columns else 800000.0
    return bperp, time_year, wavelength, incidence_angle_deg, slant_range

def normalized_time(time_year):
    t = np.asarray(time_year, dtype=float).copy()
    t -= np.nanmin(t)
    return t / np.nanmax(t) if np.nanmax(t) > 0 else np.zeros_like(t)


# ============================================================
# 5. Delaunay network
# ============================================================

def create_delaunay_network(x, y, max_dist=None):
    points = np.column_stack((x, y))
    tri = Delaunay(points)
    edges = set()

    for simplex in tri.simplices:
        i, j, k = [int(v) for v in simplex]
        for a, b in ((i, j), (j, k), (k, i)):
            if a > b: a, b = b, a
            edges.add((a, b))

    arcs = np.asarray(sorted(edges), dtype=int)
    if max_dist is not None:
        dist = np.sqrt(np.sum((points[arcs[:, 1]] - points[arcs[:, 0]]) ** 2, axis=1))
        arcs = arcs[dist <= max_dist]
    return arcs

def choose_reference_point(x, y):
    cx, cy = np.mean(x), np.mean(y)
    return int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))


# ============================================================
# 6. B-spline basis
# ============================================================

def bspline_basis(time_norm, n_basis, degree):
    time_norm = np.asarray(time_norm, dtype=float).reshape(-1)
    if n_basis <= 0: return np.zeros((len(time_norm), 0), dtype=float)

    degree = min(int(degree), int(n_basis) - 1)
    interior_count = n_basis - degree - 1
    interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1] if interior_count > 0 else np.asarray([], dtype=float)
    knots = np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))
    basis = np.zeros((len(time_norm), n_basis), dtype=float)

    for i in range(n_basis): basis[:, i] = ((knots[i] <= time_norm) & (time_norm < knots[i + 1])).astype(float)
    basis[time_norm == 1.0, -1] = 1.0

    for p in range(1, degree + 1):
        next_basis = np.zeros_like(basis)
        for i in range(n_basis):
            left_denom = knots[i + p] - knots[i]
            if left_denom > 0: next_basis[:, i] += ((time_norm - knots[i]) / left_denom) * basis[:, i]

            if i < n_basis - 1:
                right_denom = knots[i + p + 1] - knots[i + 1]
                if right_denom > 0: next_basis[:, i] += ((knots[i + p + 1] - time_norm) / right_denom) * basis[:, i + 1]
        basis = next_basis

    row_sum = np.sum(basis, axis=1)
    good = row_sum > 0
    basis[good, :] = basis[good, :] / row_sum[good, None]
    return basis

def choose_basis_count(n_obs):
    if n_obs < 6: return 0
    return min(max(4, int(np.ceil(float(n_obs) / 6.0)) + 3), n_obs - 3)

def deformation_basis(time_norm, n_basis, degree):
    basis = bspline_basis(time_norm, n_basis, degree)
    return basis[:, 1:] if basis.shape[1] > 0 else basis


# ============================================================
# 7. Precompute B-spline least-squares operator
# ============================================================

def build_spline_fit_operator(time_norm, height_system, weight, spline_degree, spline_lambda):
    n_obs = len(time_norm)
    n_basis = choose_basis_count(n_obs)
    basis = deformation_basis(time_norm, n_basis, spline_degree)
    design = np.column_stack((np.ones(n_obs), height_system, basis))
    w_sqrt = np.sqrt(weight)
    lhs = design * w_sqrt[:, None]
    n_coeff = basis.shape[1]

    if n_coeff >= 3 and spline_lambda > 0:
        roughness = np.diff(np.eye(n_coeff), n=2, axis=0)
        regularization = np.column_stack((np.zeros((roughness.shape[0], 2)), spline_lambda * roughness))
        lhs_aug = np.vstack((lhs, regularization))
    else:
        lhs_aug = lhs

    pinv_lhs = np.linalg.pinv(lhs_aug)
    fit_operator = pinv_lhs[:, :n_obs] * w_sqrt[None, :]
    return design, fit_operator


# ============================================================
# 8. Complex low-pass filter
# ============================================================

def lowpass_filter_complex_batch(z, window_size=5):
    z = np.asarray(z, dtype=complex)
    window_size = max(1, min(int(window_size), z.shape[1]))
    if window_size == 1: return z.copy()

    left = (window_size - 1) // 2
    right = window_size - 1 - left
    zpad = np.pad(z, ((0, 0), (left, right)), mode="constant")
    out = np.zeros_like(z, dtype=complex)

    for k in range(window_size): out += zpad[:, k:k + z.shape[1]]
    return out / float(window_size)


# ============================================================
# 9. Temporal displacement estimation
# ============================================================

def displacement_from_wrapped_phase(displacement_phase, wavelength, displacement_sign):
    displacement_phase = np.asarray(displacement_phase, dtype=float)
    phase_jumps = np.zeros_like(displacement_phase)
    phase_jumps[:, 1:] = wrap_phase(np.diff(displacement_phase, axis=1))
    unwrapped_phase = displacement_phase[:, [0]] + np.cumsum(phase_jumps, axis=1)
    return unwrapped_phase * wavelength / (displacement_sign * 4.0 * np.pi)

def weighted_std_batch(values, weight):
    w = weight.reshape(1, -1)
    mu = np.sum(values * w, axis=1) / np.sum(w)
    return np.sqrt(np.sum(((values - mu[:, None]) ** 2) * w, axis=1) / np.sum(w))

def weighted_coherence_batch(residual, weight):
    w = weight.reshape(1, -1)
    return np.abs(np.sum(w * np.exp(1j * residual), axis=1) / np.sum(w))


# ============================================================
# 10. B-spline temporal arc unwrapping
# ============================================================

def unwrap_arcs_bspline(phase_wrapped, arcs, bperp, time_year, wavelength, incidence_angle_deg, slant_range):
    n_time = phase_wrapped.shape[1]
    n_arcs = arcs.shape[0]

    incidence_angle_rad = np.deg2rad(incidence_angle_deg)
    height_system = 4.0 * np.pi * bperp / (wavelength * slant_range * np.sin(incidence_angle_rad))

    height_search_space = np.arange(-2.0 * HEIGHT_STD, 2.0 * HEIGHT_STD + HEIGHT_STEP * 0.5, HEIGHT_STEP)
    height_phase_search = height_search_space[:, None] * height_system[None, :]
    height_search_phasor = np.exp(-1j * height_phase_search)
    initial_height_sum = np.sum(height_search_phasor, axis=0)

    weight = np.ones(n_time, dtype=float)
    time_norm = normalized_time(time_year)
    design, fit_operator = build_spline_fit_operator(time_norm, height_system, weight, SPLINE_DEGREE, SPLINE_LAMBDA)

    arc_phase_unwrapped = np.zeros((n_arcs, n_time), dtype=float)
    arc_ambiguity = np.zeros((n_arcs, n_time), dtype=np.int16)
    arc_delta_h = np.zeros(n_arcs, dtype=float)
    arc_res_std = np.zeros(n_arcs, dtype=float)
    arc_coherence = np.zeros(n_arcs, dtype=float)

    for batch_id, start in enumerate(range(0, n_arcs, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n_arcs)
        batch_arcs = arcs[start:end]

        obs = wrap_phase(phase_wrapped[batch_arcs[:, 1], :] - phase_wrapped[batch_arcs[:, 0], :])

        e_disp = np.exp(1j * obs) * initial_height_sum[None, :]
        displacement_phase = np.angle(lowpass_filter_complex_batch(e_disp, min(5, n_time)))
        d_est = displacement_from_wrapped_phase(displacement_phase, wavelength, DISPLACEMENT_SIGN)

        terrain_phase = wrap_phase(obs - DISPLACEMENT_SIGN * 4.0 * np.pi * d_est / wavelength)
        objective = np.dot(np.exp(1j * terrain_phase) * weight[None, :], height_search_phasor.T) / np.sum(weight)
        h_est = height_search_space[np.argmax(np.abs(objective), axis=1)]

        for _ in range(ITERATIVE_TIMES):
            displacement_phase = wrap_phase(obs - h_est[:, None] * height_system[None, :])
            d_est = displacement_from_wrapped_phase(displacement_phase, wavelength, DISPLACEMENT_SIGN)
            terrain_phase = wrap_phase(obs - DISPLACEMENT_SIGN * 4.0 * np.pi * d_est / wavelength)
            objective = np.dot(np.exp(1j * terrain_phase) * weight[None, :], height_search_phasor.T) / np.sum(weight)
            h_est = height_search_space[np.argmax(np.abs(objective), axis=1)]

        temporal_model0 = h_est[:, None] * height_system[None, :] + DISPLACEMENT_SIGN * 4.0 * np.pi * d_est / wavelength
        ambiguity = np.rint((temporal_model0 - obs) / (2.0 * np.pi)).astype(np.int64)
        phase_unwrapped = obs + 2.0 * np.pi * ambiguity

        solution = np.dot(phase_unwrapped, fit_operator.T)
        model = np.dot(solution, design.T)

        for iter in range(MAX_AMBIGUITY_REFINE):
            ambiguity_new = np.rint((model - obs) / (2.0 * np.pi)).astype(np.int64)
            if np.array_equal(ambiguity_new, ambiguity): 
                break
            ambiguity = ambiguity_new
            phase_unwrapped = obs + 2.0 * np.pi * ambiguity
            solution = np.dot(phase_unwrapped, fit_operator.T)
            model = np.dot(solution, design.T)

        ambiguity = np.rint((model - obs) / (2.0 * np.pi)).astype(np.int64)
        phase_unwrapped = obs + 2.0 * np.pi * ambiguity
        residual = wrap_phase(obs - model)

        arc_phase_unwrapped[start:end, :] = phase_unwrapped
        arc_ambiguity[start:end, :] = np.clip(ambiguity, -32768, 32767).astype(np.int16)
        arc_delta_h[start:end] = solution[:, 1]
        arc_res_std[start:end] = weighted_std_batch(residual, weight)
        arc_coherence[start:end] = weighted_coherence_batch(residual, weight)

        if batch_id == 0 or (batch_id + 1) % 10 == 0 or end == n_arcs:
            print("  arc temporal unwrap: {}/{} ({:.1f}%)".format(end, n_arcs, 100.0 * end / n_arcs))

    return {
        "arcs": arcs,
        "arc_phase_unwrapped": arc_phase_unwrapped,
        "arc_ambiguity": arc_ambiguity,
        "arc_delta_h": arc_delta_h,
        "arc_res_std": arc_res_std,
        "arc_coherence": arc_coherence,
        "height_system": height_system,
    }


# ============================================================
# 11. Incidence matrix
# ============================================================

def build_incidence_matrix(arcs, n_points, ref_idx):
    n_arcs = arcs.shape[0]
    full_to_reduced = -np.ones(n_points, dtype=int)
    mask = np.arange(n_points) != ref_idx
    full_to_reduced[mask] = np.arange(n_points - 1)

    rows, cols, vals = [], [], []

    for row, (p1, p2) in enumerate(arcs):
        if p1 != ref_idx:
            rows.append(row)
            cols.append(full_to_reduced[p1])
            vals.append(-1.0)

        if p2 != ref_idx:
            rows.append(row)
            cols.append(full_to_reduced[p2])
            vals.append(1.0)

    matrix = coo_matrix((vals, (rows, cols)), shape=(n_arcs, n_points - 1)).tocsr()
    return matrix, mask


# ============================================================
# 12. Network integration
# ============================================================

def integrate_arc_values(arcs, arc_values, n_points, ref_idx, weights):
    matrix, mask = build_incidence_matrix(arcs, n_points, ref_idx)

    weights = np.asarray(weights, dtype=float)
    weights[~np.isfinite(weights) | (weights <= 0)] = 0.0
    if np.max(weights) <= 0: weights[:] = 1.0

    weights /= np.max(weights)
    sqrt_w = np.sqrt(weights)
    weighted_matrix = matrix.multiply(sqrt_w[:, None])
    normal_matrix = weighted_matrix.T.dot(weighted_matrix).tocsc()

    try:
        solver = factorized(normal_matrix)
    except Exception:
        solver = None

    arc_values = np.asarray(arc_values, dtype=float)

    if arc_values.ndim == 1:
        arc_values = arc_values[:, None]
        squeeze = True
    else:
        squeeze = False

    rhs = matrix.T.dot(weights[:, None] * arc_values)
    result_reduced = np.zeros((n_points - 1, arc_values.shape[1]), dtype=float)

    if solver is not None:
        for k in range(arc_values.shape[1]): result_reduced[:, k] = solver(np.asarray(rhs[:, k]).reshape(-1))
    else:
        for k in range(arc_values.shape[1]): result_reduced[:, k] = lsqr(weighted_matrix, arc_values[:, k] * sqrt_w)[0]

    result = np.zeros((n_points, arc_values.shape[1]), dtype=float)
    result[mask, :] = result_reduced

    return result[:, 0] if squeeze else result


# ============================================================
# 13. Spatial network inversion
# ============================================================

def spatial_network_inversion(df, net, wavelength, ref_idx):
    n_points = len(df)
    weights = 1.0 / (net["arc_res_std"] ** 2 + 1e-6)

    ps_phase_unwrapped = integrate_arc_values(net["arcs"], net["arc_phase_unwrapped"], n_points, ref_idx, weights)
    h_est = integrate_arc_values(net["arcs"], net["arc_delta_h"], n_points, ref_idx, weights)
    h_est -= h_est[ref_idx]

    height_phase_est = h_est[:, None] * net["height_system"][None, :]
    deformation_phase = ps_phase_unwrapped - height_phase_est
    deformation_m = deformation_phase * wavelength / (DISPLACEMENT_SIGN * 4.0 * np.pi)

    deformation_m -= deformation_m[:, [0]]
    deformation_m -= deformation_m[ref_idx, :][None, :]

    return ps_phase_unwrapped, h_est, deformation_m


# ============================================================
# 14. Plot deformation comparison
# ============================================================

def plot_deformation_comparison(df, deformation_cols, deformation_m, ref_idx, plot_idx, time_year, output_dir, noise_std):
    if len(deformation_cols) != deformation_m.shape[1]:
        print("Warning: deformation truth columns are missing. Skip deformation comparison plot.")
        return

    true_deformation = df[deformation_cols].values.astype(float)
    true_deformation_relative = true_deformation - true_deformation[ref_idx, :][None, :]
    true_plot = true_deformation_relative[plot_idx, :]
    estimated_plot = deformation_m[plot_idx, :] * 1000.0
    time_days = time_year * 365.25

    error = estimated_plot - true_plot
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    print("\n================ Deformation comparison PS ================")
    print("PS index            :", plot_idx)
    if "point_id" in df.columns: print("Point ID            :", df["point_id"].iloc[plot_idx])
    if "seasonal_amplitude_mm" in df.columns: print("Seasonal amplitude  : {:.3f} mm".format(df["seasonal_amplitude_mm"].iloc[plot_idx]))
    if "seasonal_phase_rad" in df.columns: print("Seasonal phase      : {:.3f} rad".format(df["seasonal_phase_rad"].iloc[plot_idx]))
    if "linear_trend_mm_per_year" in df.columns: print("Linear trend        : {:.3f} mm/year".format(df["linear_trend_mm_per_year"].iloc[plot_idx]))
    print("Deformation RMSE    : {:.3f} mm".format(rmse))
    print("Deformation MAE     : {:.3f} mm".format(mae))

    plt.figure(figsize=(10, 5.5))
    plt.scatter(time_days, true_plot, s=32, marker="o", label="True deformation")
    plt.scatter(time_days, estimated_plot, s=28, marker="^", label="Estimated deformation")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Time / days")
    plt.ylabel("Deformation / mm")
    plt.title("Deformation Comparison of PS Point {}".format(plot_idx))
    plt.legend()
    plt.tight_layout()

    noise_tag = str(noise_std).replace(".", "p")
    figure_path = os.path.join(output_dir, "deformation_comparison_ps_{}_noise_{}rad.png".format(plot_idx, noise_tag))
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Deformation comparison figure:", figure_path)


# ============================================================
# 15. Save result CSV
# ============================================================

def save_results(df, deformation_cols, net, h_est, deformation_m, ref_idx, output_dir, noise_std):
    noise_tag = str(noise_std).replace(".", "p")
    output_data = {}

    keep_cols = [
        "point_id",
        "type",
        "x",
        "y",
        "longitude",
        "latitude",
        "seasonal_amplitude_mm",
        "seasonal_phase_rad",
        "linear_trend_mm_per_year",
        "initial_deformation_true_mm",
        "delta_h_true_m",
        "phase_noise_std_rad",
    ]

    for col in keep_cols:
        if col in df.columns: output_data[col] = df[col].values

    output_data["ref_idx"] = np.full(len(df), ref_idx, dtype=int)
    output_data["h_est_m_relative"] = h_est

    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        h_true_relative = h_true - h_true[ref_idx]
        output_data["delta_h_true_relative_m"] = h_true_relative
        output_data["delta_h_error_m"] = h_est - h_true_relative

    if len(deformation_cols) == deformation_m.shape[1]:
        true_deformation = df[deformation_cols].values.astype(float)
        true_deformation_relative = true_deformation - true_deformation[ref_idx, :][None, :]
    else:
        true_deformation_relative = None

    for k in range(deformation_m.shape[1]):
        output_data["deformation_est_t{:02d}_mm".format(k + 1)] = deformation_m[:, k] * 1000.0

        if true_deformation_relative is not None:
            output_data["deformation_true_relative_t{:02d}_mm".format(k + 1)] = true_deformation_relative[:, k]
            output_data["deformation_error_t{:02d}_mm".format(k + 1)] = deformation_m[:, k] * 1000.0 - true_deformation_relative[:, k]

    result_csv = os.path.join(output_dir, "unwrapped_deformation_noise_{}rad.csv".format(noise_tag))
    pd.DataFrame(output_data).to_csv(result_csv, index=False, encoding="utf-8-sig")

    arc_data = {
        "p1": net["arcs"][:, 0],
        "p2": net["arcs"][:, 1],
        "arc_delta_h_est_m": net["arc_delta_h"],
        "arc_res_std_rad": net["arc_res_std"],
        "arc_coherence": net["arc_coherence"],
    }

    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        arc_h_true = h_true[net["arcs"][:, 1]] - h_true[net["arcs"][:, 0]]
        arc_data["arc_delta_h_true_m"] = arc_h_true
        arc_data["arc_delta_h_error_m"] = net["arc_delta_h"] - arc_h_true

    network_csv = os.path.join(output_dir, "arc_network_noise_{}rad.csv".format(noise_tag))
    pd.DataFrame(arc_data).to_csv(network_csv, index=False, encoding="utf-8-sig")

    return result_csv, network_csv, true_deformation_relative


# ============================================================
# 16. Accuracy report
# ============================================================

def accuracy_report(df, h_est, deformation_m, true_def_rel, ref_idx, net, noise_std):
    summary = {
        "phase_noise_std_rad": noise_std,
        "n_points": len(df),
        "n_arcs": len(net["arcs"]),
        "ref_idx": ref_idx,
    }

    print("\n================ noise std = {:.1f} rad ================".format(noise_std))
    print("PS points: {} | arcs: {} | reference index: {}".format(len(df), len(net["arcs"]), ref_idx))
    print("arc residual std median: {:.4f} rad | coherence median: {:.4f}".format(
        np.median(net["arc_res_std"]),
        np.median(net["arc_coherence"]),
    ))

    boundary_ratio = np.mean(
        (net["arc_delta_h"] <= -2.0 * HEIGHT_STD + HEIGHT_STEP)
        | (net["arc_delta_h"] >= 2.0 * HEIGHT_STD - HEIGHT_STEP)
    )

    print("height-search boundary arcs: {:.2f}%".format(100.0 * boundary_ratio))

    summary["arc_res_std_median_rad"] = np.median(net["arc_res_std"])
    summary["arc_coherence_median"] = np.median(net["arc_coherence"])
    summary["height_search_boundary_ratio"] = boundary_ratio

    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        h_true_rel = h_true - h_true[ref_idx]
        h_err = h_est - h_true_rel

        summary["height_rmse_m"] = np.sqrt(np.mean(h_err ** 2))
        summary["height_mae_m"] = np.mean(np.abs(h_err))

        print("height RMSE: {:.4f} m | MAE: {:.4f} m".format(
            summary["height_rmse_m"],
            summary["height_mae_m"],
        ))

    if true_def_rel is not None:
        est_mm = deformation_m * 1000.0
        err = est_mm - true_def_rel

        summary["deformation_rmse_mm"] = np.sqrt(np.mean(err ** 2))
        summary["deformation_mae_mm"] = np.mean(np.abs(err))
        summary["final_rmse_mm"] = np.sqrt(np.mean(err[:, -1] ** 2))
        summary["final_mae_mm"] = np.mean(np.abs(err[:, -1]))

        print("deformation RMSE: {:.4f} mm | MAE: {:.4f} mm".format(
            summary["deformation_rmse_mm"],
            summary["deformation_mae_mm"],
        ))
        print("final epoch RMSE: {:.4f} mm | MAE: {:.4f} mm".format(
            summary["final_rmse_mm"],
            summary["final_mae_mm"],
        ))

    return summary


# ============================================================
# 17. Main
# ============================================================

def main():
    start_all = time.time()

    input_dir, observed_files, baseline_csv = locate_input_files()
    output_dir = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    if not os.path.isdir(output_dir): os.makedirs(output_dir)

    first_df = pd.read_csv(observed_files[0][1])
    observed_cols = get_time_columns(first_df, "observed_phase_", "rad")
    deformation_cols = get_time_columns(first_df, "deformation_", "mm")

    if len(observed_cols) == 0: raise ValueError("No observed_phase_txx_rad columns were found.")

    if "x" in first_df.columns and "y" in first_df.columns:
        x = first_df["x"].values.astype(float)
        y = first_df["y"].values.astype(float)
    elif "longitude" in first_df.columns and "latitude" in first_df.columns:
        x = first_df["longitude"].values.astype(float)
        y = first_df["latitude"].values.astype(float)
    else:
        raise ValueError("CSV must contain x/y or longitude/latitude.")

    bperp, time_year, wavelength, incidence_angle_deg, slant_range = read_baseline_csv(baseline_csv, len(observed_cols))
    arcs = create_delaunay_network(x, y, MAX_ARC_DIST)
    ref_idx = choose_reference_point(x, y)

    plot_rng = np.random.RandomState(PLOT_RANDOM_SEED)
    plot_candidates = np.arange(len(first_df))
    plot_candidates = plot_candidates[plot_candidates != ref_idx]
    plot_idx = int(plot_rng.choice(plot_candidates))

    print("================ B-spline temporal arc unwrapping ================")
    print("input directory : {}".format(input_dir))
    print("output directory: {}".format(output_dir))
    print("PS points       : {}".format(len(first_df)))
    print("acquisitions    : {}".format(len(observed_cols)))
    print("Delaunay arcs   : {}".format(len(arcs)))
    print("reference index : {}".format(ref_idx))
    print("plot PS index   : {}".format(plot_idx))
    print("wavelength      : {:.6f} m".format(wavelength))
    print("incidence angle : {:.3f} deg".format(incidence_angle_deg))
    print("slant range     : {:.1f} m".format(slant_range))
    print("Bperp range     : {:.3f} ~ {:.3f} m".format(np.min(bperp), np.max(bperp)))
    print("time range      : {:.3f} ~ {:.3f} year".format(np.min(time_year), np.max(time_year)))
    print("time range      : {:.1f} ~ {:.1f} days".format(np.min(time_year) * 365.25, np.max(time_year) * 365.25))
    print("height search   : {:.1f} ~ {:.1f} m, step {:.2f} m".format(-2.0 * HEIGHT_STD, 2.0 * HEIGHT_STD, HEIGHT_STEP))

    summaries = []

    for noise_std, observed_csv in observed_files:
        start_one = time.time()

        df = pd.read_csv(observed_csv)
        current_observed_cols = get_time_columns(df, "observed_phase_", "rad")
        current_deformation_cols = get_time_columns(df, "deformation_", "mm")

        if current_observed_cols != observed_cols:
            raise ValueError("{} has inconsistent acquisition columns.".format(observed_csv))

        phase_wrapped = df[observed_cols].values.astype(float)

        print("\n------------------------------------------------------------")
        print("Start processing: {}".format(os.path.basename(observed_csv)))
        print("Phase noise std : {:.1f} rad".format(noise_std))
        print("------------------------------------------------------------")

        net = unwrap_arcs_bspline(
            phase_wrapped,
            arcs,
            bperp,
            time_year,
            wavelength,
            incidence_angle_deg,
            slant_range,
        )

        ps_phase_unwrapped, h_est, deformation_m = spatial_network_inversion(
            df,
            net,
            wavelength,
            ref_idx,
        )

        plot_deformation_comparison(
            df,
            current_deformation_cols,
            deformation_m,
            ref_idx,
            plot_idx,
            time_year,
            output_dir,
            noise_std,
        )

        result_csv, network_csv, true_def_rel = save_results(
            df,
            current_deformation_cols,
            net,
            h_est,
            deformation_m,
            ref_idx,
            output_dir,
            noise_std,
        )

        summary = accuracy_report(
            df,
            h_est,
            deformation_m,
            true_def_rel,
            ref_idx,
            net,
            noise_std,
        )

        summary["runtime_sec"] = time.time() - start_one
        summaries.append(summary)

        print("PS result CSV : {}".format(result_csv))
        print("arc result CSV: {}".format(network_csv))
        print("runtime        : {:.2f} s".format(summary["runtime_sec"]))

    summary_csv = os.path.join(output_dir, "accuracy_summary.csv")
    pd.DataFrame(summaries).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n================ all finished ================")
    print("summary CSV   : {}".format(summary_csv))
    print("plot PS index : {}".format(plot_idx))
    print("total runtime : {:.2f} s".format(time.time() - start_all))


if __name__ == "__main__":
    main()