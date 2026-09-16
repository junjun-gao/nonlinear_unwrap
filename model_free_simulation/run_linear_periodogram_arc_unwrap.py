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
OUTPUT_FOLDER = "linear_periodogram_results"
MAX_ARC_DIST = None
ARC_BATCH_SIZE = 32
CANDIDATE_CHUNK_SIZE = 8192

EXPECTED_N_ACQUISITIONS = 80
ACQUISITION_INTERVAL_DAYS = 12.0
MAX_CUMULATIVE_DEFORMATION_MM = 200.0

# Point height error: -20 ~ 20 m
# Arc height difference search: -40 ~ 40 m
HEIGHT_STDEV_M = 20.0

# 80 acquisitions, interval = 12 days
# Total time span = 79 * 12 = 948 days = about 2.60 years
# Maximum point deformation = 200 mm -> about 77 mm/year
# Considering deformation difference between two PSs:
# Arc linear velocity search range = -200 ~ 200 mm/year
LINEAR_SEARCH_MAX_MM_PER_YEAR = 200.0
LINEAR_STDEV_M_PER_YEAR = LINEAR_SEARCH_MAX_MM_PER_YEAR / 2.0 / 1000.0

REFINE_POINTS = 15
FINE_N_REFINE = 2

PLOT_RANDOM_SEED = 42
PLOT_POINT_IDX = None

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
        if len(observed_files) == len(PHASE_NOISE_STD_LIST) and len(baseline_files) > 0: return input_dir, observed_files, baseline_files[0]

    raise IOError("Cannot find observed phase CSV files and baseline CSV.")

# ============================================================
# 4. Read Sentinel-1 parameters
# ============================================================

def read_baseline_csv(baseline_csv, n_time):
    df = pd.read_csv(baseline_csv)
    if "bperp_m" not in df.columns: raise ValueError("Baseline CSV does not contain bperp_m.")
    if len(df) != n_time: raise ValueError("Baseline number {} != acquisition number {}.".format(len(df), n_time))

    bperp = df["bperp_m"].values.astype(float)
    time_year = df["time_year"].values.astype(float) if "time_year" in df.columns else np.arange(n_time, dtype=float) * ACQUISITION_INTERVAL_DAYS / 365.25
    wavelength = float(df["wavelength_m"].iloc[0]) if "wavelength_m" in df.columns else 0.0555
    incidence_angle_deg = float(df["incidence_angle_deg"].iloc[0]) if "incidence_angle_deg" in df.columns else 35.0
    slant_range = float(df["slant_range_m"].iloc[0]) if "slant_range_m" in df.columns else 800000.0

    return bperp, time_year, wavelength, incidence_angle_deg, slant_range

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

def choose_plot_point(n_points, ref_idx):
    if PLOT_POINT_IDX is not None:
        if PLOT_POINT_IDX < 0 or PLOT_POINT_IDX >= n_points: raise ValueError("PLOT_POINT_IDX is outside valid point range.")
        if PLOT_POINT_IDX == ref_idx: raise ValueError("PLOT_POINT_IDX cannot be the reference PS.")
        return int(PLOT_POINT_IDX)

    rng = np.random.RandomState(PLOT_RANDOM_SEED)
    candidates = np.arange(n_points)
    candidates = candidates[candidates != ref_idx]
    return int(rng.choice(candidates))

# ============================================================
# 6. Pure linear deformation model
#
# d(t) = v*t
#
# phase =
# height_system * dh
# + linear_system * v
#
# dh : m
# v  : m/year
# ============================================================

def build_linear_model(bperp, time_year, wavelength, incidence_angle_deg, slant_range):
    incidence_angle_rad = np.deg2rad(incidence_angle_deg)
    m2ph = 4.0 * np.pi / wavelength
    height_system = 4.0 * np.pi * bperp / (wavelength * slant_range * np.sin(incidence_angle_rad))

    model_time = np.asarray(time_year, dtype=float)
    model_time = model_time - model_time[0]

    linear_system = m2ph * model_time

    systems = np.column_stack((height_system, linear_system))
    displacement_basis = model_time[:, None]
    stdev = np.asarray([HEIGHT_STDEV_M, LINEAR_STDEV_M_PER_YEAR], dtype=float)

    return {
        "systems": systems,
        "height_system": height_system,
        "linear_system": linear_system,
        "model_time": model_time,
        "displacement_basis": displacement_basis,
        "stdev": stdev,
    }

# ============================================================
# 7. MATLAB Model.stepsize
# ============================================================

def matlab_stepsize(system):
    system_range = float(np.max(system) - np.min(system))
    if system_range <= 0.0: return np.inf
    return 0.4 * np.pi / system_range

# ============================================================
# 8. MATLAB Model.searchspace
# ============================================================

def matlab_searchspace(stepsize, stdev):
    if not np.isfinite(stepsize) or stepsize <= 0.0: return np.asarray([0.0], dtype=float)

    positive = np.arange(0.0, 2.0 * stdev + 1e-12, stepsize)
    if len(positive) == 0: positive = np.asarray([0.0], dtype=float)

    return np.concatenate((-positive[:0:-1], positive))

# ============================================================
# 9. MATLAB refine space
# ============================================================

def matlab_refine_space(stepsize, scale=1.0):
    return stepsize * np.linspace(-0.7, 0.7, REFINE_POINTS) * scale

# ============================================================
# 10. Build candidate grid
# ============================================================

def build_candidate_grid(systems, axes, name):
    meshes = np.meshgrid(*axes, indexing="ij")
    params = np.column_stack([mesh.reshape(-1) for mesh in meshes]).astype(np.float32)
    model_phase = np.dot(params.astype(float), systems.T)
    model_phasor = np.exp(-1j * model_phase).astype(np.complex64)

    return {
        "name": name,
        "params": params,
        "model_phasor": model_phasor,
        "n_candidates": params.shape[0],
    }

# ============================================================
# 11. Prepare pure linear periodogram grids
#
# Search parameters:
#   dh : residual height difference
#   v  : linear velocity difference
#
# Coarse search:
#   dh = -40 ~ 40 m
#   v  = -200 ~ 200 mm/year
#
# Then perform several local refinement searches.
# ============================================================

def prepare_periodogram_grids(model):
    systems = model["systems"]
    stdev = model["stdev"]

    steps = np.asarray([matlab_stepsize(systems[:, i]) for i in range(2)], dtype=float)
    coarse_axes = [matlab_searchspace(steps[i], stdev[i]) for i in range(2)]

    grids = {}
    grids["coarse"] = build_candidate_grid(systems, coarse_axes, "coarse")

    for zoom in range(FINE_N_REFINE + 1):
        scale = 0.1 ** zoom
        refine_axes = [matlab_refine_space(steps[i], scale) for i in range(2)]
        grids["refine_{}".format(zoom)] = build_candidate_grid(systems, refine_axes, "refine_{}".format(zoom))

    return steps, grids

# ============================================================
# 12. Periodogram objective
# ============================================================

def evaluate_periodogram_grid(phase, grid, weight=None):
    phase = np.asarray(phase, dtype=float)
    n_arc = phase.shape[0]

    if weight is None: weight = np.ones(phase.shape[1], dtype=float)

    weight = np.asarray(weight, dtype=float)
    weight_sum = np.sum(weight)
    phase_phasor = (np.exp(1j * phase) * weight[None, :]).astype(np.complex64)

    best_coh = np.full(n_arc, -1.0, dtype=float)
    best_idx = np.zeros(n_arc, dtype=int)
    best_complex = np.zeros(n_arc, dtype=np.complex64)

    for start in range(0, grid["n_candidates"], CANDIDATE_CHUNK_SIZE):
        end = min(start + CANDIDATE_CHUNK_SIZE, grid["n_candidates"])
        score = np.dot(phase_phasor, grid["model_phasor"][start:end, :].T) / weight_sum
        abs_score = np.abs(score)

        local_idx = np.argmax(abs_score, axis=1)
        local_coh = abs_score[np.arange(n_arc), local_idx]
        improve = local_coh > best_coh

        if np.any(improve):
            rows = np.where(improve)[0]
            best_coh[rows] = local_coh[rows]
            best_idx[rows] = start + local_idx[rows]
            best_complex[rows] = score[rows, local_idx[rows]]

    best_params = grid["params"][best_idx, :].astype(float)
    phase0 = np.angle(best_complex).astype(float)

    return best_params, best_coh, phase0

# ============================================================
# 13. Periodogram estimation
# ============================================================

def periodogram_estimation_block(phase, systems, coarse_grid, refine_grids, weight=None):
    params, enscoh, phase0 = evaluate_periodogram_grid(phase, coarse_grid, weight)

    for grid in refine_grids:
        residual_phase = phase - np.dot(params, systems.T)
        dparams, enscoh, phase0 = evaluate_periodogram_grid(residual_phase, grid, weight)
        params += dparams

    modelphase = phase0[:, None] + np.dot(params, systems.T)

    return modelphase, params, enscoh, phase0

# ============================================================
# 14. Pure linear periodogram arc unwrapping
#
# Search:
#
# phi = phi_height(dh) + phi_linear(v)
#
# Only two unknown parameters:
#   dh
#   v
# ============================================================

def unwrap_arcs_linear_periodogram(phase_wrapped, arcs, model, grids):
    systems = model["systems"]
    n_arcs = len(arcs)
    n_time = phase_wrapped.shape[1]

    arc_model_phase = np.zeros((n_arcs, n_time), dtype=np.float32)
    arc_unwrapped_phase = np.zeros((n_arcs, n_time), dtype=np.float32)
    arc_ambiguity = np.zeros((n_arcs, n_time), dtype=np.int16)
    arc_params = np.zeros((n_arcs, 2), dtype=np.float32)
    arc_res_std = np.zeros(n_arcs, dtype=np.float32)
    arc_coherence = np.zeros(n_arcs, dtype=np.float32)

    refine_grids = [grids["refine_{}".format(i)] for i in range(FINE_N_REFINE + 1)]

    for batch_id, start in enumerate(range(0, n_arcs, ARC_BATCH_SIZE)):
        end = min(start + ARC_BATCH_SIZE, n_arcs)
        batch_arcs = arcs[start:end]

        # Arc observed phase = point2 - point1
        obs = wrap_phase(phase_wrapped[batch_arcs[:, 1], :] - phase_wrapped[batch_arcs[:, 0], :])

        # ----------------------------------------------------
        # Pure linear periodogram search:
        #
        # dh + v
        # ----------------------------------------------------
        modelphase, bestfit, enscoh, _ = periodogram_estimation_block(
            obs,
            systems,
            grids["coarse"],
            refine_grids,
        )

        # ----------------------------------------------------
        # Recover integer ambiguity
        # ----------------------------------------------------
        ambiguity = np.rint((modelphase - obs) / (2.0 * np.pi)).astype(np.int64)
        unwrapped_phase = obs + 2.0 * np.pi * ambiguity
        residual = wrap_phase(obs - modelphase)

        arc_model_phase[start:end, :] = modelphase.astype(np.float32)
        arc_unwrapped_phase[start:end, :] = unwrapped_phase.astype(np.float32)
        arc_ambiguity[start:end, :] = np.clip(ambiguity, -32768, 32767).astype(np.int16)
        arc_params[start:end, :] = bestfit.astype(np.float32)
        arc_res_std[start:end] = np.std(residual, axis=1).astype(np.float32)
        arc_coherence[start:end] = enscoh.astype(np.float32)

        if batch_id == 0 or (batch_id + 1) % 10 == 0 or end == n_arcs:
            print("  linear periodogram: {}/{} ({:.1f}%)".format(end, n_arcs, 100.0 * end / n_arcs))

    return {
        "arcs": arcs,
        "arc_model_phase": arc_model_phase,
        "arc_phase_unwrapped": arc_unwrapped_phase,
        "arc_ambiguity": arc_ambiguity,
        "arc_params": arc_params,
        "arc_delta_h": arc_params[:, 0].astype(float),
        "arc_linear_velocity": arc_params[:, 1].astype(float),
        "arc_res_std": arc_res_std.astype(float),
        "arc_coherence": arc_coherence.astype(float),
    }

# ============================================================
# 15. Incidence matrix
# ============================================================

def build_incidence_matrix(arcs, n_points, ref_idx):
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

    matrix = coo_matrix((vals, (rows, cols)), shape=(len(arcs), n_points - 1)).tocsr()

    return matrix, mask

# ============================================================
# 16. Network integration
# ============================================================

def integrate_arc_values(arcs, arc_values, n_points, ref_idx, weights):
    matrix, mask = build_incidence_matrix(arcs, n_points, ref_idx)

    weights = np.asarray(weights, dtype=float).copy()
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
    squeeze = arc_values.ndim == 1
    if squeeze: arc_values = arc_values[:, None]

    rhs = matrix.T.dot(weights[:, None] * arc_values)
    reduced = np.zeros((n_points - 1, arc_values.shape[1]), dtype=float)

    if solver is not None:
        for k in range(arc_values.shape[1]): reduced[:, k] = solver(np.asarray(rhs[:, k]).reshape(-1))
    else:
        for k in range(arc_values.shape[1]): reduced[:, k] = lsqr(weighted_matrix, arc_values[:, k] * sqrt_w)[0]

    result = np.zeros((n_points, arc_values.shape[1]), dtype=float)
    result[mask, :] = reduced

    return result[:, 0] if squeeze else result

# ============================================================
# 17. PS network inversion
# ============================================================

def spatial_network_inversion(net, model, wavelength, n_points, ref_idx):
    weights = 1.0 / (net["arc_res_std"] ** 2 + 1e-6)

    ps_phase_unwrapped = integrate_arc_values(net["arcs"], net["arc_phase_unwrapped"], n_points, ref_idx, weights)
    h_est = integrate_arc_values(net["arcs"], net["arc_delta_h"], n_points, ref_idx, weights)
    point_params = integrate_arc_values(net["arcs"], net["arc_params"], n_points, ref_idx, weights)

    ps_phase_unwrapped -= ps_phase_unwrapped[:, [0]]
    h_est -= h_est[ref_idx]

    height_phase_est = h_est[:, None] * model["height_system"][None, :]
    deformation_phase = ps_phase_unwrapped - height_phase_est
    deformation_m = deformation_phase * wavelength / (4.0 * np.pi)
    deformation_m -= deformation_m[:, [0]]

    return ps_phase_unwrapped, point_params, h_est, deformation_m

# ============================================================
# 18. Plot deformation comparison
# ============================================================

def plot_deformation_comparison(df, deformation_cols, deformation_m, ref_idx, point_idx, time_year, noise_std, output_dir):
    if len(deformation_cols) != deformation_m.shape[1]:
        print("Warning: deformation truth columns are missing. Skip deformation comparison plot.")
        return None

    true_deformation = df[deformation_cols].values.astype(float)
    true_deformation_relative = true_deformation - true_deformation[ref_idx, :][None, :]

    true_plot = true_deformation_relative[point_idx, :]
    estimated_plot = deformation_m[point_idx, :] * 1000.0
    time_days = np.asarray(time_year, dtype=float) * 365.25

    error = estimated_plot - true_plot
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    print("\n================ Deformation comparison PS ================")
    print("PS index              :", point_idx)
    if "point_id" in df.columns: print("Point ID              :", df["point_id"].iloc[point_idx])
    if "linear_trend_mm_per_year" in df.columns: print("Linear trend          : {:.3f} mm/year".format(df["linear_trend_mm_per_year"].iloc[point_idx]))
    print("Deformation RMSE      : {:.3f} mm".format(rmse))
    print("Deformation MAE       : {:.3f} mm".format(mae))

    plt.figure(figsize=(10, 5.5))
    plt.plot(time_days, true_plot, linewidth=2.0, label="True deformation")
    plt.scatter(time_days, estimated_plot, s=28, marker="^", label="Periodogram estimated deformation")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Time / days")
    plt.ylabel("Deformation / mm")
    plt.title("Deformation Comparison of PS Point {}".format(point_idx))
    plt.legend()
    plt.tight_layout()

    tag = str(noise_std).replace(".", "p")
    figure_path = os.path.join(output_dir, "deformation_comparison_ps_{}_noise_{}rad.png".format(point_idx, tag))
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("deformation plot:", figure_path)

    return figure_path

# ============================================================
# 19. Save result CSV
# ============================================================

def save_results(df, deformation_cols, net, point_params, h_est, deformation_m, ref_idx, output_dir, noise_std):
    tag = str(noise_std).replace(".", "p")
    output_data = {}

    keep_cols = [
        "point_id",
        "type",
        "x",
        "y",
        "longitude",
        "latitude",
        "linear_trend_mm_per_year",
        "initial_deformation_true_mm",
        "delta_h_true_m",
        "phase_noise_std_rad",
    ]

    for col in keep_cols:
        if col in df.columns: output_data[col] = df[col].values

    output_data["ref_idx"] = np.full(len(df), ref_idx, dtype=int)
    output_data["h_est_m_relative"] = h_est
    output_data["linear_est_mm_per_year"] = point_params[:, 1] * 1000.0

    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        h_true_rel = h_true - h_true[ref_idx]
        output_data["delta_h_true_relative_m"] = h_true_rel
        output_data["delta_h_error_m"] = h_est - h_true_rel

    if "linear_trend_mm_per_year" in df.columns:
        linear_true = df["linear_trend_mm_per_year"].values.astype(float)
        linear_true_rel = linear_true - linear_true[ref_idx]
        output_data["linear_true_relative_mm_per_year"] = linear_true_rel
        output_data["linear_error_mm_per_year"] = point_params[:, 1] * 1000.0 - linear_true_rel

    true_def = df[deformation_cols].values.astype(float) if len(deformation_cols) == deformation_m.shape[1] else None
    true_def_rel = true_def - true_def[ref_idx, :][None, :] if true_def is not None else None

    for k in range(deformation_m.shape[1]):
        output_data["deformation_est_t{:02d}_mm".format(k + 1)] = deformation_m[:, k] * 1000.0

        if true_def_rel is not None:
            output_data["deformation_true_relative_t{:02d}_mm".format(k + 1)] = true_def_rel[:, k]
            output_data["deformation_error_t{:02d}_mm".format(k + 1)] = deformation_m[:, k] * 1000.0 - true_def_rel[:, k]

    result_csv = os.path.join(output_dir, "unwrapped_deformation_noise_{}rad.csv".format(tag))
    pd.DataFrame(output_data).to_csv(result_csv, index=False, encoding="utf-8-sig")

    arc_params = net["arc_params"]

    arc_data = {
        "p1": net["arcs"][:, 0],
        "p2": net["arcs"][:, 1],
        "arc_delta_h_est_m": arc_params[:, 0],
        "arc_linear_est_m_per_year": arc_params[:, 1],
        "arc_res_std_rad": net["arc_res_std"],
        "arc_coherence": net["arc_coherence"],
        "ambiguity_nonzero_count": np.sum(net["arc_ambiguity"] != 0, axis=1),
    }

    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        arc_h_true = h_true[net["arcs"][:, 1]] - h_true[net["arcs"][:, 0]]
        arc_data["arc_delta_h_true_m"] = arc_h_true
        arc_data["arc_delta_h_error_m"] = arc_params[:, 0] - arc_h_true

    if "linear_trend_mm_per_year" in df.columns:
        linear_true_m = df["linear_trend_mm_per_year"].values.astype(float) / 1000.0
        arc_linear_true = linear_true_m[net["arcs"][:, 1]] - linear_true_m[net["arcs"][:, 0]]
        arc_data["arc_linear_true_m_per_year"] = arc_linear_true
        arc_data["arc_linear_error_m_per_year"] = arc_params[:, 1] - arc_linear_true

    network_csv = os.path.join(output_dir, "arc_network_noise_{}rad.csv".format(tag))
    pd.DataFrame(arc_data).to_csv(network_csv, index=False, encoding="utf-8-sig")

    return result_csv, network_csv, true_def_rel

# ============================================================
# 20. Accuracy report
# ============================================================

def accuracy_report(df, h_est, deformation_m, true_def_rel, ref_idx, net, noise_std, height_stepsize, linear_stepsize):
    summary = {
        "phase_noise_std_rad": noise_std,
        "n_points": len(df),
        "n_arcs": len(net["arcs"]),
        "ref_idx": ref_idx,
    }

    print("\n================ noise std = {:.1f} rad ================".format(noise_std))
    print("PS points: {} | arcs: {} | reference index: {}".format(len(df), len(net["arcs"]), ref_idx))
    print("arc residual std median: {:.4f} rad | coherence median: {:.4f}".format(np.median(net["arc_res_std"]), np.median(net["arc_coherence"])))

    height_limit = 2.0 * HEIGHT_STDEV_M
    linear_limit = LINEAR_SEARCH_MAX_MM_PER_YEAR / 1000.0
    height_boundary_ratio = np.mean(np.abs(net["arc_delta_h"]) >= max(0.0, height_limit - height_stepsize))
    linear_boundary_ratio = np.mean(np.abs(net["arc_linear_velocity"]) >= max(0.0, linear_limit - linear_stepsize))

    print("height-search boundary arcs: {:.2f}%".format(100.0 * height_boundary_ratio))
    print("linear-search boundary arcs: {:.2f}%".format(100.0 * linear_boundary_ratio))

    summary["arc_res_std_median_rad"] = np.median(net["arc_res_std"])
    summary["arc_coherence_median"] = np.median(net["arc_coherence"])
    summary["height_search_boundary_ratio"] = height_boundary_ratio
    summary["linear_search_boundary_ratio"] = linear_boundary_ratio

    if "delta_h_true_m" in df.columns:
        h_true = df["delta_h_true_m"].values.astype(float)
        h_true_rel = h_true - h_true[ref_idx]
        h_err = h_est - h_true_rel

        summary["height_rmse_m"] = np.sqrt(np.mean(h_err ** 2))
        summary["height_mae_m"] = np.mean(np.abs(h_err))

        print("height RMSE: {:.4f} m | MAE: {:.4f} m".format(summary["height_rmse_m"], summary["height_mae_m"]))

    if true_def_rel is not None:
        est_mm = deformation_m * 1000.0
        err = est_mm - true_def_rel

        summary["deformation_rmse_mm"] = np.sqrt(np.mean(err ** 2))
        summary["deformation_mae_mm"] = np.mean(np.abs(err))
        summary["final_rmse_mm"] = np.sqrt(np.mean(err[:, -1] ** 2))
        summary["final_mae_mm"] = np.mean(np.abs(err[:, -1]))

        print("deformation RMSE: {:.4f} mm | MAE: {:.4f} mm".format(summary["deformation_rmse_mm"], summary["deformation_mae_mm"]))
        print("final epoch RMSE: {:.4f} mm | MAE: {:.4f} mm".format(summary["final_rmse_mm"], summary["final_mae_mm"]))

    return summary

# ============================================================
# 21. Main
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
    if len(observed_cols) != EXPECTED_N_ACQUISITIONS: raise ValueError("Expected {} acquisitions, but found {}.".format(EXPECTED_N_ACQUISITIONS, len(observed_cols)))

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

    model = build_linear_model(bperp, time_year, wavelength, incidence_angle_deg, slant_range)
    steps, grids = prepare_periodogram_grids(model)

    plot_point_idx = choose_plot_point(len(first_df), ref_idx)

    total_days = (EXPECTED_N_ACQUISITIONS - 1) * ACQUISITION_INTERVAL_DAYS
    total_years = total_days / 365.25
    max_point_velocity = MAX_CUMULATIVE_DEFORMATION_MM / total_years

    print("================ Pure linear periodogram arc unwrapping ================")
    print("input directory    : {}".format(input_dir))
    print("output directory   : {}".format(output_dir))
    print("PS points          : {}".format(len(first_df)))
    print("acquisitions       : {}".format(len(observed_cols)))
    print("interval           : {:.1f} days".format(ACQUISITION_INTERVAL_DAYS))
    print("Delaunay arcs      : {}".format(len(arcs)))
    print("reference index    : {}".format(ref_idx))
    print("plot point index   : {}".format(plot_point_idx))
    print("wavelength         : {:.6f} m".format(wavelength))
    print("incidence angle    : {:.3f} deg".format(incidence_angle_deg))
    print("slant range        : {:.1f} m".format(slant_range))
    print("time range         : {:.3f} years".format(time_year[-1] - time_year[0]))
    print("time range         : {:.1f} days".format((time_year[-1] - time_year[0]) * 365.25))
    print("deformation model  : d(t) = v*t")
    print("max deformation    : {:.1f} mm".format(MAX_CUMULATIVE_DEFORMATION_MM))
    print("equivalent velocity: {:.3f} mm/year".format(max_point_velocity))
    print("height search      : +/- {:.3f} m | step {:.6f} m".format(2.0 * HEIGHT_STDEV_M, steps[0]))
    print("linear arc search  : +/- {:.3f} mm/year | step {:.3f} mm/year".format(LINEAR_SEARCH_MAX_MM_PER_YEAR, steps[1] * 1000.0))
    print("refine points      : {} | refine passes: {}".format(REFINE_POINTS, FINE_N_REFINE + 1))

    summaries = []

    for noise_std, observed_csv in observed_files:
        start_one = time.time()

        df = pd.read_csv(observed_csv)
        current_observed_cols = get_time_columns(df, "observed_phase_", "rad")
        current_deformation_cols = get_time_columns(df, "deformation_", "mm")

        if current_observed_cols != observed_cols: raise ValueError("{} has inconsistent acquisition columns.".format(observed_csv))

        phase_wrapped = df[observed_cols].values.astype(float)

        print("\n------------------------------------------------------------")
        print("Start processing: {}".format(os.path.basename(observed_csv)))
        print("Phase noise std : {:.1f} rad".format(noise_std))
        print("------------------------------------------------------------")

        net = unwrap_arcs_linear_periodogram(phase_wrapped, arcs, model, grids)

        ps_phase_unwrapped, point_params, h_est, deformation_m = spatial_network_inversion(
            net,
            model,
            wavelength,
            len(df),
            ref_idx,
        )

        plot_deformation_comparison(
            df,
            current_deformation_cols,
            deformation_m,
            ref_idx,
            plot_point_idx,
            time_year,
            noise_std,
            output_dir,
        )

        result_csv, network_csv, true_def_rel = save_results(
            df,
            current_deformation_cols,
            net,
            point_params,
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
            steps[0],
            steps[1],
        )

        summary["runtime_sec"] = time.time() - start_one
        summaries.append(summary)

        print("PS result CSV  : {}".format(result_csv))
        print("arc result CSV : {}".format(network_csv))
        print("runtime        : {:.2f} s".format(summary["runtime_sec"]))

    summary_csv = os.path.join(output_dir, "accuracy_summary.csv")
    pd.DataFrame(summaries).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n================ all finished ================")
    print("summary CSV   : {}".format(summary_csv))
    print("plot PS index : {}".format(plot_point_idx))
    print("total runtime : {:.2f} s".format(time.time() - start_all))

if __name__ == "__main__":
    main()