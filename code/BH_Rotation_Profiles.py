"""
* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ *
| BH_Rotation_Profiles:                                                                                                                       |
| Determining Profiles of Characteristic Angular Velocities from Radio-Image Sequences of Black Holes                                         |
|                                                                                                                                             |
| Version (MFW, 14 Aug 2026) used for final analysis                                                                                          |
| Authors (alphabetically): Rohan G. Amanaganti, Bram van den Berg, and Michael F. Wondrak                                                    |
|                                                                                                                                             |
| Adapted and extended program based on                                                                                                       |
| - Cylinder_Clean.py (N. Conroy, M. Baubock, C. Gammie, https://doi.org/10.5281/zenodo.13717347):                                            |
|   "A cleaned version of the script used in 'Rotation in Event Horizon Telescope Movies', by Conroy et al. 2023"                             |
| - patternspeed.py (R. Dahale and EHT Collaboration, https://github.com/rohandahale/ehteval/commit/f01c5793b1963bfa478f4925528a8fd358d39d17) |
|   "Validation and evaluation of horizon-scale Sagittarius A* video reconstructions"                                                         |
|                                                                                                                                             |
| Determining patten speed profiles (characteristic angular velocities) from videos of black holes in general relativity and beyond.          |
| Input is consecutive covariantly ray-traced snapshots from covariant magnetohydrodynamic simulations and the position and size of the       |
| bright ring including uncertainties.                                                                                                        | 
| Various ray-tracer file formats (as produced by BHOSS, ipole, RAPTOR) as well as Fits can be read in.                                       |
| For information on usage of this script, call this script with the argument --help.                                                         |
* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ *
"""

## Importing
import argparse
from astropy.io import fits
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
import numpy.fft as fft
import os
import scipy.ndimage as ndimage
from scipy.ndimage import label 
from scipy.signal.windows import tukey
from scipy.optimize import least_squares
from natsort import natsorted
import sys
from tqdm import tqdm
import glob
import time
import multiprocessing
from scipy.stats import truncnorm

def save_text_file(pattern_speed_medians, pattern_speed_means, pattern_speed_modes, pattern_speed_std, 
                   pattern_speed_upper_lims, pattern_speed_lower_lims, pattern_speed_max, pattern_speed_min, 
                    orig_ps_list, orig_xicrit_list, orig_xicrit_factor_base_sig_list, 
                    rel_impact_params, end_times, 
                    function_parameter, sfiles_full_path, offset_x0, offset_y0, offset_radius, 
                    num_windows, window_length, specific_output, output_filename_prefix):
    
    with open(f"{specific_output}/{output_filename_prefix}-pattern_speed_summary.txt", 'w') as f:
        f.write(f"Non-Kerr Pattern Speed\n")
        f.write(f"Pattern Speed Script: {__file__}\n")
        f.write(f"Path to first file used: {sfiles_full_path[0]}\n")
        f.write(f"GRRT input file format: {function_parameter}\n")
        f.write(f"Image offset x0 [M]: {offset_x0}, y0 [M]: {offset_y0}. \nReference impact parameter (offset_radius) [M]: {offset_radius}\n")
        f.write(f"Number of windows: {num_windows}\n")
        f.write(f"Length of sub windows [M]: {window_length}\n")
        f.write(f"Specific output directory: {specific_output}\n\n")

        f.write(f"Window index, Impact parameter [M], Impact parameter [b_ref], Pattern speed [deg/M], Pattern speed min [deg/M],"
                +"Pattern speed max [deg/M], Pattern speed median [deg/M], Pattern speed median plus [deg/M], Pattern speed median minus [deg/M],"
                +"Pattern speed mean [deg/M], Pattern speed std [deg/M], Pattern speed mode [deg/M], xi_crit, xi_crit_factor_base_sig\n")

        for idx in range(len(end_times)):
            str_to_write = f'{int(end_times[idx])}, {offset_radius*rel_impact_params[idx]:.6f}, {rel_impact_params[idx]:.6f}, {orig_ps_list[idx]:.6f}, '
            str_to_write += f'{pattern_speed_min[idx]:.6f}, {pattern_speed_max[idx]:.6f}, {pattern_speed_medians[idx]:.6f}, {pattern_speed_upper_lims[idx]:.6f}, {pattern_speed_lower_lims[idx]:.6f}, '
            str_to_write += f'{pattern_speed_means[idx]:.6f}, {pattern_speed_std[idx]:.6f}, {pattern_speed_modes[idx]:.6f}, '
            str_to_write += f'{orig_xicrit_list[idx]:.6f}, {orig_xicrit_factor_base_sig_list[idx]:.6f}\n'
            f.write(str_to_write)


def lower_xicrit_threshold(racf, xi_crit, non_zero_columns, non_zero_rows):
  """
  A ceiling check that ensures xi_crit is low enough that we aren't (too) limited by pixelation effects. Returns new threshold xi_crit and mask with integration bounds Q if necessary. We require at least 5 time slices and angle slices. 

  Args:
      racf: The autocorrelation 
      xi_crit: the calibrated threshold for calculating 2nd Moments
      non_zero_columns: the number of columns (time slices) above the threshold
      non_zero_rows: the number of columns (time slices) above the threshold
  """

  min_required_columns = 5
  min_required_rows = 5

  central_row_index = racf.shape[0] // 2
  central_column_index = racf.shape[1] // 2

  # Attempt to lower xi_crit quickly using max value near center
  xi_crit = np.max(racf[:, central_column_index + min_required_columns // 2])

  # Initial threshold mask
  threshold_mask = (racf >= xi_crit).astype(int) & \
    (np.abs(np.arange(racf.shape[1]) - central_column_index) <= min_required_columns // 2)

  # Label connected regions
  labels, num_features = label(threshold_mask)
  central_peak_label = labels[central_row_index, central_column_index]
  Q = labels == central_peak_label

  # Count non-zero rows and columns
  non_zero_columns = np.count_nonzero(np.sum(Q, axis=0))
  non_zero_rows = np.count_nonzero(np.sum(Q, axis=1))
    
  #print('~~~a) Old Col & Row Nums:', non_zero_columns, non_zero_rows, '. old xi_Crit:', xi_crit)

  # Lower threshold further if necessary. 
  if non_zero_columns < min_required_columns or non_zero_rows < min_required_rows:
    # Step 1: Sort unique values above current xi_crit (descending order)
    xi_crit_values = np.unique(racf[racf < xi_crit])
    xi_crit_values = np.sort(xi_crit_values)[::-1] # descending

    # Step 2: Loop over possible xi_crit values
    for xi_candidate in xi_crit_values:
      threshold_mask = (racf >= xi_candidate).astype(int)

      labels, num_features = label(threshold_mask)
      central_peak_label = labels[central_row_index, central_column_index]

      if central_peak_label == 0:
        continue # center not in any connected region

      Q_candidate = labels == central_peak_label

      # Check connectivity/size
      non_zero_columns_cand = np.count_nonzero(np.sum(Q_candidate, axis=0))
      non_zero_rows_cand = np.count_nonzero(np.sum(Q_candidate, axis=1))

      if (non_zero_columns_cand >= min_required_columns) and (non_zero_rows_cand >= min_required_rows):
        Q = Q_candidate
        xi_crit = xi_candidate
        non_zero_columns = non_zero_columns_cand
        non_zero_rows = non_zero_rows_cand
        break # Found valid region

  #print('~~~b) New Col & Row Nums:', non_zero_columns, non_zero_rows, '. new xi_Crit:', xi_crit)
  return xi_crit, Q


def compute_autocorrelation(qs, curr_log_status, str_output):
    """
    Compute centered autocorrelation function.
    Cyclic in PA (angle wrapping via FFT), non-cyclic in time (no time wrapping).
    Based on Nick Conroy's autocorr_unwrapped_fft method.
    
    Input:  qs shape (Nt, Nphi) -- time x angle
    Output: racf shape (2*Nt-1, Nphi) -- time-lag x angle-shift, peak normalized to 1
            qsn shape (Nt, Nphi) -- mean-subtracted cylinder
    """
    print(f'{str_output}')

    # Mean subtract
    qsn = np.copy(qs)

    if curr_log_status:
        min_float_val = np.finfo(np.float64).tiny
        qsn[qsn < min_float_val] = min_float_val
        qsn = np.log10(qsn) ## note that log can produce nans if values in qs are too small or negative   

    qsn -= qsn.mean(axis=0, keepdims=True)  # subtract temporal mean at each angle
    qsn -= qsn.mean(axis=1, keepdims=True)  # subtract azimuthal mean at each time
    
    Nt, Nphi = qsn.shape
    R = np.zeros((2 * Nt - 1, Nphi), dtype=float)
    
    for j in range(2 * Nt - 1):
        tau_idx = j - (Nt - 1)  # time lag index
        
        # Get overlapping time slices (no time wrapping)
        if tau_idx >= 0:
            f1 = qsn[:Nt - tau_idx, :]   # (noverlap, Nphi)
            f2 = qsn[tau_idx:, :]         # (noverlap, Nphi)
        else:
            f1 = qsn[-tau_idx:, :]        # (noverlap, Nphi)
            f2 = qsn[:Nt + tau_idx, :]    # (noverlap, Nphi)
        
        noverlap = f1.shape[0]
        if noverlap == 0:
            continue
        
        # FFT along angle axis (axis=1) for cyclic PA wrapping
        FB = np.fft.fft(f1, axis=1)
        FC = np.fft.fft(f2, axis=1)
        cross = np.fft.ifft(FB * np.conj(FC), axis=1).real
        
        # Sum over overlapping time points -> (Nphi,)
        numerator = cross.sum(axis=0)
        
        # Map FFT ordering to direct-sum ordering
        numerator = np.roll(numerator[::-1], 1)
        
        # Per-lag normalization (prevents values > 1 at edges)
        sigma1 = np.sqrt(np.mean(f1**2))
        sigma2 = np.sqrt(np.mean(f2**2))
        if sigma1 == 0 or sigma2 == 0:
            continue
        norm = Nphi * noverlap * sigma1 * sigma2
        R[j, :] = numerator / norm
    
    # Center PA shifts so zero-shift is at the middle
    R = np.roll(R, shift=-Nphi // 2, axis=1)
    
    # Normalize peak to 1
    max_val = np.max(R)
    if max_val > 0:
        R /= max_val

    R = R[ (R.shape[0]//4) : (3*R.shape[0]//4) , : ]  ## NEW
    
    return R, qsn


def sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, dtheta = 2., x_shift=0, y_shift=0, r_shift=0, raytracer = 'raytracer'):
    # Dimensions from data
    # sIall is (y, x, time)
    ny, nx, _ = sIall.shape
    nt = end_idx - start_idx
    
    # Base physical coords
    x0 = ring_params['x0']
    y0 = ring_params['y0']
    r_factor_val = ring_params['r_factor']
    r_bright_ring = ring_params['r_ring']
    
    # Apply shifts
    x0_new = x0 + x_shift
    y0_new = y0 + y_shift
    r_new = r_factor_val*(r_bright_ring + r_shift)
    
    # Convert back to pixels
    center_index = nx / 2.0 
    
    xc_new_pix = (x0_new) / dx + center_index
    yc_new_pix = (y0_new) / dx + center_index
    r_new_pix = r_new / dx
    
    # Circle sampling points
    ntheta = int(360./dtheta)
    theta = np.linspace(-np.pi, np.pi, ntheta, endpoint=False)
    pa = theta + 0.5*np.pi 
    
    # Calculate cylinder sampling coordinates
    # icirc (y), jcirc (x)
    icirc = yc_new_pix + r_new_pix*np.sin(pa)
    jcirc = xc_new_pix + r_new_pix*np.cos(pa)

    if raytracer == 'Raptor':
        icirc = M-1 - icirc
    
    # Vectorized Sampling
    # We want to sample at (icirc, jcirc) for every time step t [0...nt-1]
    # Coordinates for map_coordinates: (ndim, npoints)
    # y coordinates: repeat icirc for each time step
    # x coordinates: repeat jcirc for each time step
    # t coordinates: repeat time index for each angle
    
    # Grid construction
    # y (ntheta) -> tile nt times -> (nt*ntheta)
    # x (ntheta) -> tile nt times -> (nt*ntheta)
    # t (nt)     -> repeat ntheta times -> (nt*ntheta)
    # We want result (nt, ntheta).
    # Order: for t in 0..nt: for angle in 0..ntheta
    # So t indices should be 0,0..0 (ntheta times), 1,1..1 (ntheta times)
    
    T_coords = np.repeat(np.arange(nt), ntheta)
    Y_coords = np.tile(icirc, nt)
    X_coords = np.tile(jcirc, nt)
    
    coords = np.vstack([Y_coords, X_coords, T_coords])
    
    # Order=1 (linear interpolation)
    # Mode='nearest' to match clamping behavior (or 'constant', cval=0)
    samples = ndimage.map_coordinates(sIall[:, :, start_idx:end_idx], coords, order=1, mode='nearest')
    
    qs = samples.reshape((nt, ntheta))
             
    return qs

def calculate_pattern_speed(racf, dt, curr_window_idx, rel_impact_param, dtheta=2.0, xi_crit_factor=2.25, curr_orig = False):

    racf_std = np.std(racf)

    xi_crit_factor_base_sig = xi_crit_factor * racf_std
    xi_crit = xi_crit_factor_base_sig

    # Filter connected region
    labels_map, num_features = label((racf > xi_crit).astype(int))
    center_idx = (racf.shape[0]//2, racf.shape[1]//2)

    # Check initial region width
    non_zero_columns = 0
    non_zero_rows = 0
    if labels_map[center_idx] != 0:
         Q = labels_map == labels_map[center_idx]
         non_zero_columns = np.count_nonzero(np.sum(Q, axis=0))
         non_zero_rows = np.count_nonzero(np.sum(Q, axis=1))

    # Fallback 1: if center not in threshold or region too thin (< 5 pixels)
    target_width = 5
    if (labels_map[center_idx] == 0) or (non_zero_columns < target_width) or (non_zero_rows < target_width):
         col_offset = target_width // 2
         row_offset = target_width // 2

         # Determine edge column/row (ensure in bounds)
         edge_col_idx = min(center_idx[1] + col_offset, racf.shape[1] - 1)
         edge_row_idx = min(center_idx[0] + row_offset, racf.shape[0] - 1)
         edge_row = racf[edge_row_idx, :]
         edge_col = racf[:, edge_col_idx]
         unique_edge_vals = np.unique(np.concatenate([edge_row, edge_col]))[::-1]
         unique_edge_vals = unique_edge_vals[unique_edge_vals <= xi_crit]

         # Loop through values on edges from highest to lowest until we get
         # target_width in both dimensions (connected to center)
         for candidate_xi in unique_edge_vals:
                threshold_mask = (racf >= candidate_xi)
                mask_int = threshold_mask.astype(int)

                labels_map, num_features = label(mask_int)

                # Final safety check
                if labels_map[center_idx] == 0:
                    return 0.0, xi_crit, xi_crit_factor_base_sig #np.zeros_like(racf), np.zeros_like(labels_map, dtype=bool)

                Q = labels_map == labels_map[center_idx]

                non_zero_columns = np.count_nonzero(np.sum(Q, axis=0))
                non_zero_rows = np.count_nonzero(np.sum(Q, axis=1))

                if (non_zero_columns >= target_width) and (non_zero_rows >= target_width):
                    xi_crit = candidate_xi
                    break

    else:
         Q = labels_map == labels_map[center_idx]

    # Fallback 2: phi wrapping if xi_crit region spans all values of phi
    non_zero_columns = np.count_nonzero(np.sum(Q, axis=0))
    if non_zero_columns == len(racf[0]):
        # Concatenate racf on the Delta-phi axis to allow the peak to wrap
        racf_cut = np.concatenate([racf, racf, racf], axis=1)

        center_wrap = (center_idx[0], center_idx[1] + len(racf[0]))
        labels_wrap, _ = label((racf_cut > xi_crit).astype(int))

        if labels_wrap[center_wrap] == 0:
            return 0.0, xi_crit, xi_crit_factor_base_sig #np.zeros_like(racf), np.zeros_like(racf, dtype=bool)

        Q_wrap = labels_wrap == labels_wrap[center_wrap]

        # Redefine Q by folding the wrapped mask back onto the original phi axis
        Q = (
            Q_wrap[:, :len(racf[0])]
            | Q_wrap[:, len(racf[0]):2 * len(racf[0])]
            | Q_wrap[:, 2 * len(racf[0]):3 * len(racf[0])]
        )

        # Apply mask
        racf_cut[~Q_wrap] = 0.0
    else:
        racf_cut = np.copy(racf)

        # Apply mask
        racf_cut[~Q] = 0.0


    # Moments
    ts = np.linspace(-len(racf_cut)/2, len(racf_cut)/2, len(racf_cut), endpoint=False)
    phis = np.linspace(-len(racf_cut[0])/2, len(racf_cut[0])/2, len(racf_cut[0]), endpoint=False)

    delta_t = ts[1] - ts[0]
    delta_phi = phis[1] - phis[0]

    # Meshgrid
    T_mesh, Phi_mesh = np.meshgrid(ts, phis, indexing='ij')


    # Weighted sums
    moment_t = np.sum(racf_cut * T_mesh**2)
    moment_t_phi = np.sum(racf_cut * T_mesh * Phi_mesh)

    if moment_t == 0:
        pattern_speed = 0
    else:
        pattern_speed = moment_t_phi / moment_t
        # Units
        pattern_speed = pattern_speed * dtheta / dt

    return pattern_speed, xi_crit, xi_crit_factor_base_sig


def run_mc(start_idx, end_idx, sIall, N, M, ring_params, dx, dt, n_samples, xi_crit_factor_base, curr_window_idx, window_end_time, rel_impact_param, curr_log_status, raytracer):
    # Setup MC
    # Sigma fit from xi_crit RMSE curves

    print("Calculating unperturbed (= original) pattern speed")
    # qs = sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, dtheta, x_shift, y_shift, r_shift, raytracer)
    qs = sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, 2.,     0.,      0.,      0.,      raytracer)
    racf, _ = compute_autocorrelation(qs, curr_log_status, f'{curr_window_idx}, {rel_impact_param:4.2f}: orig')

    orig_ps, orig_xi_crit_used, orig_xi_crit_factor_base_sig = calculate_pattern_speed(racf, dt, curr_window_idx, rel_impact_param, dtheta=2.0, xi_crit_factor = xi_crit_factor_base, curr_orig = True)

    racf_best_std = np.std(racf)

    sigma = 0.7 
    
    # Calculate base absolute threshold (mimicking cylinder.py logic)
    abs_base = xi_crit_factor_base * racf_best_std
    
    # xi_crit bounds (0, 1) in absolute units
    # Truncated normal for xi_crit
    a, b = (0 - abs_base) / sigma, (1 - abs_base) / sigma
    xi_crit_abs_samples = truncnorm.rvs(a, b, loc=abs_base, scale=sigma, size=n_samples)
    
    # Perturb Ring Parameters (x, y, r)
    # rerr is used for std of x, y, r perturbation
    xerr = ring_params.get('x_err', 1.0) # default if missing
    yerr = ring_params.get('y_err', 1.0) # default if missing
    rerr = ring_params.get('r_err', 1.0) # default if missing
    
    x_samples = np.random.normal(0, xerr, n_samples)
    y_samples = np.random.normal(0, yerr, n_samples)
    r_samples = np.random.normal(0, rerr, n_samples)
    
    ps_samples = []
    xi_crit_used_samples = []
    xi_crit_factor_base_sig_samples = []
    
    print("Computing pattern speeds for different realizations")
    for i in range(n_samples):
        # Sample cylinder with perturbed parameters
        # qs = sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, dtheta, x_shift, y_shift, r_shift, raytracer)
        qs = sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, 2., x_samples[i], y_samples[i], r_samples[i], raytracer)
        
        # Compute Autocorr
        racf, _ = compute_autocorrelation(qs, curr_log_status, f'{curr_window_idx}, {rel_impact_param:4.2f}: {i:d}')
        
        # Calculate pattern speed
        # Convert absolute sample to factor for this specific realization
        current_std = np.std(racf)
        factor = xi_crit_abs_samples[i] / current_std
        
        ps, xi_crit_used, xi_crit_factor_base_sig = calculate_pattern_speed(racf, dt, curr_window_idx, rel_impact_param, dtheta=2.0, xi_crit_factor = factor, curr_orig = False)
        ps_samples.append(ps)
        xi_crit_used_samples.append(xi_crit_used)
        xi_crit_factor_base_sig_samples.append(xi_crit_factor_base_sig)
        
    ps_samples = np.array(ps_samples)
    
    # Statistics
    mean_ps = np.mean(ps_samples)
    std_ps = np.std(ps_samples)
    max_ps, min_ps = np.max(ps_samples), np.min(ps_samples)
    percentiles = np.percentile(ps_samples, [15.865, 50, 84.135]) 
    median = percentiles[1]
    
    # Calculate uncertainties relative to the median (1 sigma)
    median_plus = percentiles[2] - median
    median_minus = median - percentiles[0]
    
    # Mode
    counts, bin_edges = np.histogram(ps_samples, bins=50)
    modal_bin_index = np.argmax(counts)
    mode_value = 0.5 * (bin_edges[modal_bin_index] + bin_edges[modal_bin_index + 1])
    
    return {
        'samples': ps_samples,
        'xicrit_samples': xi_crit_used_samples,
        'xicrit_factor_base_sig_samples': xi_crit_factor_base_sig_samples,
        'mean': mean_ps,
        'std': std_ps,
        'median': median,
        'median_plus_sigma': median_plus,
        'median_minus_sigma': median_minus,
        'max': max_ps,
        'min': min_ps,
        'mode': mode_value
    }, window_end_time, rel_impact_param, orig_ps, orig_xi_crit_used, orig_xi_crit_factor_base_sig
  

# gnw colorbar implementation

def plot_I_slice(Iall_input, plot_dir, xc, yc, plot_radius, n=0):
    plt.figure(dpi = 300, figsize=(6, 5))
    plt.imshow(Iall_input[:, :, n], cmap='afmhot')
    plt.colorbar(label='Intensity')
    plt.title(f'Iall, snapshot {n}')
    plt.xlabel('X-axis [px]')
    plt.ylabel('Y-axis [px]')
    plt.tight_layout()
    plt.savefig(plot_dir + f'{output_filename_prefix}-Iall_n_{n}.png')
    plt.savefig(plot_dir + f'{output_filename_prefix}-Iall_n_{n}.pdf')
    plt.close()
    
    Iavg = np.mean(Iall_input, axis=2)
    circle = patches.Circle((xc,  yc), plot_radius, color = 'green', fill = False, label = 'shadow')

    fig, (ax) = plt.subplots(1, 1, dpi = 300, figsize = (6, 5))
    im = ax.imshow(Iavg, cmap='afmhot')
    ax.add_patch(circle)
    ax.scatter(xc, yc, color = 'green', s = 3)
    fig.colorbar(im, ax=ax, label='Intensity')
    ax.set_title(f'Iall, avg')
    ax.set_xlabel('X-axis [px]')
    ax.set_ylabel('Y-axis [px]')
    plt.tight_layout()
    plt.savefig(plot_dir + f'{output_filename_prefix}-Iall_avg.png')
    plt.savefig(plot_dir + f'{output_filename_prefix}-Iall_avg.pdf')
    plt.close()



def colorbar(mappable):
    """ the way matplotlib colorbar should have been implemented """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)



## Definition of different file format readers: 

def Ipole(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]
    
    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles = natsorted(files)
    
    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    ## set some parameters for the run, based on first file in sequence
    hfp = h5py.File(sfiles_full_path[0],'r')
    FOV_M = hfp['header']['camera']['dx'][()] ## image scale in M
    FOV_uas =hfp['header']['camera']['fovx_dsource'][()]
    if  hfp['header']['camera']['fovx_dsource'][()] != hfp['header']['camera']['fovy_dsource'][()]:
        print("Error: need a square image. This image appears to be {0:n} x {1:n} pixels".format(hfp['header']['camera']['fovx_dsource'][()], hfp['header']['camera']['fovy_dsource'][()]))
        sys.exit(0)
    imagep = np.copy(hfp['pol']).transpose((1,0,2))
    N = imagep.shape[0]                       ## x-size of array
    M = imagep.shape[1]                       ## y-size of array
    dx_uas = FOV_uas/N                            ## x-size of pixel
    dy_uas = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    try:
        spin = hfp['fluid_header']['a'][()]
    except KeyError:
        spin = 0.5
    inclination = hfp['header']['camera']['thetacam'][()]

    ## We find dt by reading in header/t [which is in units of M] from the first couple files.
    hf1 = h5py.File(sfiles_full_path[0], 'r')
    hf2 = h5py.File(sfiles_full_path[1], 'r')
    hf3 = h5py.File(sfiles_full_path[2], 'r')
    hf4 = h5py.File(sfiles_full_path[3], 'r')

    dt2 = hf2[('header/t')][()] - hf1[('header/t')][()]
    dt3 = hf3[('header/t')][()] - hf2[('header/t')][()]
    dt4 = hf4[('header/t')][()] - hf3[('header/t')][()]
    dt = (dt2 + dt3 + dt4)/3.

    hf1.close()
    hf2.close()
    hf3.close()
    hf4.close()

    dt = round(dt, 1) ## round to 1 number of significant decimals
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    ## Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N, M, nftot))

    ## Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n = 0
    print(f"Reading files from {list_directory} ...")
    for fname_path in tqdm(sfiles_full_path): ## read in file
        hfp = h5py.File(fname_path,'r')
        imagep = np.copy(hfp['pol']).transpose((1,0,2))
        Iall[:,:,n] = imagep[:,:,0]
        hfp.close()
        n += 1
    
    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall


def Ipole_every_2nd_snapshot(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]
    
    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles_orig = natsorted(files)
    sfiles = sfiles_orig[::2]

    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    ## set some parameters for the run, based on first file in sequence
    hfp = h5py.File(sfiles_full_path[0],'r')
    FOV_M = hfp['header']['camera']['dx'][()] ## image scale in M
    FOV_uas =hfp['header']['camera']['fovx_dsource'][()]
    if  hfp['header']['camera']['fovx_dsource'][()] != hfp['header']['camera']['fovy_dsource'][()]:
        print("Error: need a square image. This image appears to be {0:n} x {1:n} pixels".format(hfp['header']['camera']['fovx_dsource'][()], hfp['header']['camera']['fovy_dsource'][()]))
        sys.exit(0)
    imagep = np.copy(hfp['pol']).transpose((1,0,2))
    N = imagep.shape[0]                       ## x-size of array
    M = imagep.shape[1]                       ## y-size of array
    dx_uas = FOV_uas/N                            ## x-size of pixel
    dy_uas = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    try:
        spin = hfp['fluid_header']['a'][()]
    except KeyError:
        spin = 0.5
    inclination = hfp['header']['camera']['thetacam'][()]

    ## We find dt by reading in header/t [which is in units of M] from the first couple files.
    hf1 = h5py.File(sfiles_full_path[0], 'r')
    hf2 = h5py.File(sfiles_full_path[1], 'r')
    hf3 = h5py.File(sfiles_full_path[2], 'r')
    hf4 = h5py.File(sfiles_full_path[3], 'r')

    dt2 = hf2[('header/t')][()] - hf1[('header/t')][()]
    dt3 = hf3[('header/t')][()] - hf2[('header/t')][()]
    dt4 = hf4[('header/t')][()] - hf3[('header/t')][()]
    dt = (dt2 + dt3 + dt4)/3.

    hf1.close()
    hf2.close()
    hf3.close()
    hf4.close()

    dt = round(dt, 1) ## round to 1 number of significant decimals
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    ## Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N, M, nftot))

    ## Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n = 0
    print(f"Reading files from {list_directory} ...")
    for fname_path in tqdm(sfiles_full_path): ## read in file
        hfp = h5py.File(fname_path,'r')
        imagep = np.copy(hfp['pol']).transpose((1,0,2))
        Iall[:,:,n] = imagep[:,:,0]
        hfp.close()
        n += 1
    
    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall


# Hardcoded: FOV_M, FOV_uas, dt!
def Ipole_Fast_Slow_Comp(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]
    
    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles = natsorted(files)
    
    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    FOV_M = 42.1333
    FOV_uas = 160.

    hfp = h5py.File(sfiles_full_path[0],'r')
    imagep = np.copy(hfp['pol']).transpose((1,0,2))
    N = imagep.shape[0]                       ## x-size of array
    M = imagep.shape[1]                       ## y-size of array
    dx_uas = FOV_uas/N                            ## x-size of pixel
    dy_uas = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    spin = nan
    inclination = nan
    dt = 0.5
    dt = round(dt, 1)
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    ## Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N, M, nftot))

    ## Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n = 0
    print(f"Reading files from {list_directory} ...")
    for fname_path in tqdm(sfiles_full_path): ## read in file
        hfp = h5py.File(fname_path,'r')
        imagep = np.copy(hfp['pol']).transpose((1,0,2))
        Iall[:,:,n] = imagep[:,:,0]
        hfp.close()
        n += 1
    
    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall


# Hardcoded: FOV_M, FOV_uas, dt!
def Ipole_Fast_Slow_Comp_every_20th_snapshot(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]
    
    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles_orig = natsorted(files)
    sfiles = sfiles_orig[::20]
    
    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    FOV_M = 42.1333
    FOV_uas = 160.

    hfp = h5py.File(sfiles_full_path[0],'r')
    imagep = np.copy(hfp['pol']).transpose((1,0,2))
    N = imagep.shape[0]                       ## x-size of array
    M = imagep.shape[1]                       ## y-size of array
    dx_uas = FOV_uas/N                            ## x-size of pixel
    dy_uas = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    spin = nan
    inclination = nan
    dt = 0.5 * 20
    dt = round(dt, 1)
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    ## Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N, M, nftot))

    ## Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n = 0
    print(f"Reading files from {list_directory} ...")
    for fname_path in tqdm(sfiles_full_path): ## read in file
        hfp = h5py.File(fname_path,'r')
        imagep = np.copy(hfp['pol']).transpose((1,0,2))
        Iall[:,:,n] = imagep[:,:,0]
        hfp.close()
        n += 1
    
    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall


# Hardcoded: FOV_M, dt! 
def Raptor(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]

    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles = natsorted(files)

    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    data = np.genfromtxt(sfiles_full_path[0])
    # raptor_input = np.genfromtxt('model.in')  ## Use this only if model.in is present
    # FOV_M = raptor_input[8][2]  ## Use this only if model.in is present
    FOV_M = 40. # for all our Raptor imaging, see model_SgrA.in files
    FOV_uas = 2*(np.abs(data[0][0])) * 1000 * 1000
    N = len(np.unique(data[:,0]))
    M = len(np.unique(data[:,1]))
    dx_uas = FOV_uas/N                            ## x-size of pixel
    dy_uas = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    spin = nan
    inclination = nan

    if args.cadence_man:
        dt = cadence_man
    else:
        dt = 10.

    dt = round(dt, 1) 
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    ## Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N,N, nftot))
    
    ## Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n=0
    print(f"Reading files from {list_directory} ...")
    # Open the file and read the lines
    for fname_path in tqdm(sfiles_full_path):
        with open(fname_path, 'r') as currf:
            first_row = currf.readline().strip('\n')

        num_skiprows = 0
        if 'X' in first_row:
            num_skiprows = 1
            
        usecols_num = 2
        all_data = np.loadtxt(fname_path, unpack = True, usecols = usecols_num, skiprows = num_skiprows)
        Iall[:, :, n] = np.flipud((all_data.reshape(N, N)).T)
        n += 1
    
    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall


def BHOSS_1(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]

    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles = natsorted(files)

    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    datah1 = np.genfromtxt(sfiles_full_path[0], max_rows=1)
    datah2 = np.genfromtxt(sfiles_full_path[0], skip_header=1, max_rows=1)
    datah3 = np.genfromtxt(sfiles_full_path[0], skip_header=2, max_rows=1)
    FOV_M = datah1[0]*2
    FOV_uas = datah2[-1] * FOV_M
    N = int(datah1[2])
    M = int(datah1[2])
    dx_uas = FOV_uas/N                            ## x-size of pixel
    dy_uas = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    spin = datah2[3]
    inclination = datah2[1]
    
    datah2_file2 = np.genfromtxt(sfiles_full_path[1], skip_header=1, max_rows=1)
    datah2_file3 = np.genfromtxt(sfiles_full_path[2], skip_header=1, max_rows=1)
    datah2_file4 = np.genfromtxt(sfiles_full_path[3], skip_header=1, max_rows=1)

    dt2 = datah2_file2[0] - datah2[0]
    dt3 = datah2_file3[0] - datah2_file2[0]
    dt4 = datah2_file4[0] - datah2_file3[0]
    dt = (dt2 + dt3 + dt4)/3.

    dt = round(dt, 1)
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    ## Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N,M, nftot))

    ## Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n=0
    print(f"Reading files from {list_directory} ...")
    # Open the file and read the lines
    for fname_path in tqdm(sfiles_full_path):
        with open(fname_path, 'r') as currf:
            data_lines = currf.readlines()
            lines = data_lines[3:]

        for line in lines:
            # Split the line into columns (x, y, I)
            values = line.split()
            
            # Extract the I value (third column) and convert it to a float
            I_value = float(values[3])
            
            # Calculate the column and row indices in the matrix
            col = int(values[1]) -1
            row =  int(values[0]) -1
            Iall[row, col][n] = I_value 
        n+=1
    
    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall


# Hardcoded: dt!
def BHOSS_2(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]

    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles = natsorted(files)
    
    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    datah1 = np.genfromtxt(sfiles_full_path[0], skip_header=1, max_rows=1)
    datah8 = np.genfromtxt(sfiles_full_path[0], skip_header=8, max_rows=1)
    datah13 = np.genfromtxt(sfiles_full_path[0], skip_header=13, max_rows=1)
    datah9 = np.genfromtxt(sfiles_full_path[0], skip_header=9, max_rows=1)
    datah10 = np.genfromtxt(sfiles_full_path[0], skip_header=10, max_rows=1)
    FOV_M = float(datah8)*2.
    FOV_uas = datah13[0] * FOV_M 
    N = int(datah9)
    M = int(datah10)
    dx_uas = FOV_uas/N                            ## x-size of pixel
    dy_uas = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    spin = datah1[1]
    inclination = datah1[3]
    
    if args.cadence_man:
        dt = cadence_man
    else:
        dt = 10.

    dt = round(dt, 1)
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    ## Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N,M, nftot))

    ## Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n=0
    print(f"Reading files from {list_directory} ...")
    # Open the file and read the lines
    for fname_path in tqdm(sfiles_full_path):
        line_counter = 0 
        with open(fname_path, 'r') as currf:
            data_lines = currf.readlines()
            lines = data_lines[17:]

        for col in range (N):
            for row in range (M):

            # Split the line into columns (x, y, I)
                I_value = float(lines[line_counter].strip())
                Iall[row, col][n] = I_value
                line_counter+=1
        n+=1
    
    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall


# Hardcoded: FOV_M!
def Fits(starting_string):
    slash='/'
    files = [f for f in os.listdir(list_directory) if f.startswith(starting_string)]
    
    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles = natsorted(files)
    
    nftot = len(sfiles)
    if nftot > 1:
        print(f"\n{nftot} files found, \nfrom {sfiles[0]} to {sfiles[-1]}")
        print(f"List of files, in order: \n{sfiles}")
    else:
        print(f"\n{nftot} file found: {sfiles[0]}")

    sfiles_full_path = []
    for file in sfiles:
        sfiles_full_path.append(list_directory + slash + file)
    if nftot > 1:
        print(f"Full paths: from \n{sfiles_full_path[0]} \nto \n{sfiles_full_path[-1]}")
    else:
        print(f"Full path: \n{sfiles_full_path[0]}")

    hdul0 = fits.open(sfiles_full_path[0])
    hdul1 = fits.open(sfiles_full_path[1])
    hdul2 = fits.open(sfiles_full_path[2])
    hdul3 = fits.open(sfiles_full_path[3])

    FOV_M = 60.
    N = hdul0[0].header['NAXIS1']
    M = hdul0[0].header['NAXIS2']                  
    dx_uas = np.abs(hdul0[0].header['CDELT1']) * 3600 * 1e6                      
    dy_uas = np.abs(hdul0[0].header['CDELT2']) * 3600 * 1e6 
    FOV_uas = N * dx_uas                       
    half_FOV = FOV_uas/2.

    print(f"FOV [M]: {FOV_M}")
    print(f"FOV [uas]: {FOV_uas}")
    print(f"Resolution: {N} x {M}")
    print(f"Pixel size [uas x uas]: {dx_uas} x {dy_uas}")

    spin = hdul0[0].header['SPIN']
    inclination = hdul0[0].header['INC']
    
    dt2 = hdul1[0].header['TIME'] - hdul0[0].header['TIME']
    dt3 = hdul2[0].header['TIME'] - hdul1[0].header['TIME']
    dt4 = hdul3[0].header['TIME'] - hdul2[0].header['TIME']
    dt = (dt2 + dt3 + dt4)/3.

    dt = round(dt, 1) 
    print(f"Time cadence dt [M]: {dt}")

    Tduration = (nftot-1)*dt

    Iall = np.zeros((N,M, nftot))
    n=0
    # Open the file and read the lines
    for n, file in enumerate(tqdm(sfiles_full_path)):
        img = fits.getdata(file)  # returns a 2D numpy array (y, x)
        Iall[:, :, n] = img      
    Iall = Iall.transpose((1,0,2))

    return nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall



function_map = {
        "BHOSS_1": BHOSS_1,
        "BHOSS_2": BHOSS_2,
        "Fits": Fits,
        "Ipole": Ipole,
        "Raptor": Raptor,
        "Ipole_every_2nd_snapshot": Ipole_every_2nd_snapshot,
        "Ipole_Fast_Slow_Comp": Ipole_Fast_Slow_Comp,
        "Ipole_Fast_Slow_Comp_every_20th_snapshot": Ipole_Fast_Slow_Comp_every_20th_snapshot
    }


## Reading Files
# try:
if True:

    nan = float('nan')

   ## Defining File Paths and Image Parameters
    
    parser = argparse.ArgumentParser(description="Pattern Speed Super Script")

    # Required positional arguments
    parser.add_argument("function_parameter", help="Name of the function to run (i.e., to choose from BHOSS_1, BHOSS_2, Fits, Ipole, Raptor). Some properties (esp. FOV_M and cadence dt) might be hardcoded.")
    parser.add_argument("list_directory", help="Directory containing files to process.")
    parser.add_argument("starting_string", help="Starting of filename string.")
    parser.add_argument("specific_output", help="Directory for saving script output.")

    # Optional argument:
    parser.add_argument("--radius_list", nargs="+", type=float, default=None,
                        help="Impact parameters/image radii on which to calculate the pattern speed, in units of M. The maximum value needs to be smaller than FOV_uas/2. For internally calculating relative ring radii, the offset_radius is automatically set to the Schwarzschild shadow radius. \nThis option is overwritten if --args.offset_radius is provided. Use: start end (#steps+1) (e.g., --radius_list 1 9 5).")

    parser.add_argument("--offset", nargs="+", type=float, default=None,
                        help="Offset x and y in units of M (e.g., --offset 1.2 -0.7).")
    
    parser.add_argument("--offset_radius", nargs="+", type=float, default=None,
                        help="Reference impact parameter/image radius on which to calculate the pattern speed, in units of M. Use: radius (e.g., --offset_radius 5.223).")

    # parser.add_argument("--offset_err", nargs="+", type=float, default=None,
    #                     help="offset error x and y in units of M (e.g., --offset 1.2 -0.7)")
    
    # parser.add_argument("--offset_radius_err", nargs="+", type=float, default=None,
    #                     help="error in reference impact parameter/image radius on which to calculate the pattern speed, in units of M. Use: radius (e.g., --offset_radius 5.223)")

    parser.add_argument("--output_filename_prefix", nargs="+", type=str, 
                        help="String added in front of output files to associate them with the model investigated and the imaging parameters. Use: string, ending without a '-' (e.g., for the Kerr reference of the Dilaton_2 model: --output_filename_prefix Dilaton_2-Kerr_a06_i150_Rh160).")

    parser.add_argument("--cadence_man", nargs="+", type=float,
                        help="Manually set cadence dt in units of M. Only for BHOSS_2 and Raptor.")

    parser.add_argument("--window_length", nargs="+", type=int, default=[550], 
                        help="Length of subwindow in units of M. Integer required.")

    parser.add_argument("--log_status", nargs="+", type=str, default=['False'],
                        help="Whether to take log of smoothed cylinder before computing autocorrelations. If so, use capitalized 'True'.")

    parser.add_argument("--ordering", nargs="+", type=str, default=['forward'],
                        help="Needs forward in this script. Preparing future extensions, where windows increasing in length from 0 or end of simulation.")

    parser.add_argument("--ring_params_filename", nargs="+", type=str,
                        help="Path to file with ring center and radii for the simulation.")
    


    args = parser.parse_args()

    function_parameter = args.function_parameter
    list_directory = args.list_directory
    starting_string = args.starting_string
    specific_output = args.specific_output

    if args.radius_list:
        start, end, amount = args.radius_list
        radius_list = np.linspace(start,end, num = int(amount))
    else:
        radius_list = []

    if args.offset:
        offset_x0, offset_y0 = args.offset
    else:
        offset_x0 = 0
        offset_y0 = 0
    

    radius_list_addition_relative = []

    if args.offset_radius:
        offset_radius = args.offset_radius[0]
        radius_list_addition_relative = [round(0.5 + i * 0.05, 2) for i in range(41)]
        radius_list = [offset_radius * radius_relative for radius_relative in radius_list_addition_relative]
        radius_list_addition_absolute = [round(1.0 + i * 0.5, 2) for i in range(33)]
        radius_list.extend(radius_list_addition_absolute)
        radius_list = np.insert(radius_list, 0, np.sqrt(27))
        radius_list = np.sort(radius_list)
    else:
        offset_radius = np.sqrt(27)
    radius_list_relative = [radius_absolute/offset_radius for radius_absolute in radius_list]
    
    x_err, y_err, r_err, rc_err = 0, 0, 0, 0

    if args.ring_params_filename:
        with h5py.File(f'{args.ring_params_filename[0]}', 'r') as h5f:
            try:
                rad50 = h5f['rad50'][...]                 ## default radius of the ring for each image in the cube (in units of M)
            except:
                rad50 = h5f['radius50'][...]                 ## default radius of the ring for each image in the cube (in units of M)
            
            centers = h5f['predicted_centers'][...]
        
        r_err = np.std(rad50)
        x_err = np.std(centers[:, 0])
        y_err = np.std(centers[:, 1])
        rc_err = np.sqrt(x_err**2 + y_err**2)

    if args.output_filename_prefix:
        output_filename_prefix = args.output_filename_prefix[0]
    else:
        output_filename_prefix = "-"

    if args.cadence_man:
        cadence_man = args.cadence_man[0]

    if args.window_length:
        window_length = int(args.window_length[0])
    
    log_status = args.log_status[0]
    if log_status == 'True':
        curr_log_status = True
    else:
        curr_log_status = False
    
    window_ordering = args.ordering[0]

    print("Function:", function_parameter)
    print("List directory:", list_directory)
    print("Starting string:", starting_string)
    print("Specific output:", specific_output)
    if args.offset:
        print("Offset x0 [M]: ", offset_x0, "and offset y0 [M]: ", offset_y0)

    if args.ring_params_filename:
        print('r_err [M]: ', r_err)
        print('x_err [M]: ', x_err)
        print('y_err [M]: ', y_err)
        print('rc_err [M]: ', rc_err)

    if radius_list_addition_relative:
        print(f"First group of impact parameters normalized to reference:\n{radius_list_addition_relative}")
    print(f"Total list of absolute impact parameters [M]:\n{radius_list}")

    output_dir = f"{specific_output}/pattern_speed_output/"
    image_dir = f"{specific_output}/simulation_images/"
    smoothed_image_dir = f"{specific_output}/simulation_images_smoothed/"

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(smoothed_image_dir):
            os.makedirs(smoothed_image_dir)
    except:
        print("Directories already exist")

    with open(f"{output_dir}/{output_filename_prefix}-output_dir.txt", 'w') as f:
        f.write(f"output_dir: {output_dir}")
    with open(f"{image_dir}/{output_filename_prefix}-image_dir.txt", 'w') as f:
        f.write(f"image_dir: {image_dir}")
    with open(f"{smoothed_image_dir}/{output_filename_prefix}-smoothed_image_dir.txt", 'w') as f:
        f.write(f"smoothed_image_dir: {smoothed_image_dir}")

    ## Load the files
    if function_parameter in function_map:
        nftot, sfiles_full_path, FOV_M, FOV_uas, N, M, dx_uas, dy_uas, half_FOV, spin, inclination, dt, Tduration, Iall = function_map[function_parameter](starting_string)

        shift_length = 50
        num_windows = 1 + int((Tduration - window_length)/shift_length)
        num_window_snaps = 1 + int(window_length/dt)

        print(f'Number of Windows: {num_windows}')
        print(f'Number Snapshots in a Window: {num_window_snaps}')
        print(f'Window Length [M]: {window_length}')

        if args.offset:
            xc = N/2 + offset_x0 * (N/FOV_M) ## The final factor is a unit conversion to pixel units. ## for Kerr: 2*spin*np.sin(inclination * np.pi/180) *(N/FOV_M)
            yc = M/2 + offset_y0 * (M/FOV_M)

        if function_parameter == 'Raptor':
            yc = M - 1 - yc
        
        plot_I_slice(Iall, image_dir, xc, yc, offset_radius*(M/FOV_M), n=0)

        print(f"Loaded in data!")

    else:
        print(f"Error: Function '{function_parameter}' not recognized. Function parameters are BHOSS_1, BHOSS_2, Fits, Ipole, and Raptor.")
        sys.exit(1)

    ## Smooth
    sig = 20/(dx_uas *(2*np.sqrt(2*np.log(2))))
    print(f'Blurring:\t FWHM [uas]: 20, \t dx_uas [uas]: {dx_uas}, \t sig [px]: {sig}')
    smoothed_Iall = ndimage.gaussian_filter(Iall, sigma=(sig,sig,0))
    del Iall

    plot_I_slice(smoothed_Iall, smoothed_image_dir, xc, yc, offset_radius*(M/FOV_M), n=0)

    cores_per_job = 1
    num_cores = int((multiprocessing.cpu_count())/(cores_per_job))
    print(f'Number of Cores: {num_cores}')
    count = 0


    ## MC params
    n_samples = 1000
    xi_crit_factor_base = 2.25
    dx, dy = FOV_M/N, FOV_M/M # ATTENTION: HERE A DIFFERENT DEFINITION dx AND dy: NO LONGER px size [uas], but px size [M]!
    start_time_process = time.time()

    if window_ordering == 'forward':
        curr_window_ordering = window_ordering

        start_time_list = [float(i*shift_length) for i in range(num_windows)]
        end_time_list = [float(i*shift_length + window_length) for i in range(num_windows)]

        output_filename_prefix_win = output_filename_prefix + f'-{curr_window_ordering}-log_{str(curr_log_status)}' 

        inputs = []
        for curr_window_idx in range(0, num_windows):
            for b_idx in range(len(radius_list_relative)):
                ring_params = {
                                    'x_err' : rc_err,
                                    'y_err' : rc_err,
                                    'r_err' : r_err,
                                    'r_factor': radius_list_relative[b_idx],
                                    'x0': offset_x0,
                                    'y0': offset_y0,
                                    'r_ring': offset_radius
                            }
                inputs.append((int(start_time_list[curr_window_idx]/dt), int(end_time_list[curr_window_idx]/dt), smoothed_Iall, N, M, ring_params, dx, dt, n_samples, xi_crit_factor_base, curr_window_idx, end_time_list[curr_window_idx], radius_list_relative[b_idx], curr_log_status, function_parameter))
        
        print(f'{len(inputs)} input configurations loaded for {curr_window_ordering} and log {curr_log_status}.')

        pool      = multiprocessing.Pool(num_cores)
        async_result    = pool.starmap_async(run_mc, inputs)
        result = async_result.get()
        pool.close()
        pool.join()

        pattern_speed_medians = []
        pattern_speed_means = []
        pattern_speed_modes = []
        pattern_speed_upper_lims = []
        pattern_speed_lower_lims = []
        pattern_speed_std = []
        pattern_speed_max = []
        pattern_speed_min = []

        end_times = []
        rel_impact_params = []

        ps_arr_list = []
        xicrit_arr_list = []
        xicrit_factor_base_sig_arr_list = []

        orig_ps_list = []
        orig_xicrit_list = []
        orig_xicrit_factor_base_sig_list = []

        for el in result:
            mc_res, window_end_time, rel_impact_param, orig_ps, orig_xi_crit_used, orig_xicrit_factor_base_sig = el

            pattern_speed_medians.append(mc_res['median'])
            pattern_speed_means.append(mc_res['mean'])
            pattern_speed_modes.append(mc_res['mode'])
            pattern_speed_upper_lims.append(mc_res['median_plus_sigma'])
            pattern_speed_lower_lims.append(mc_res['median_minus_sigma'])
            pattern_speed_max.append(mc_res['max'])
            pattern_speed_min.append(mc_res['min'])
            pattern_speed_std.append(mc_res['std'])

            ps_arr_list.append(mc_res['samples'])
            xicrit_arr_list.append(mc_res['xicrit_samples'])
            xicrit_factor_base_sig_arr_list.append(mc_res['xicrit_factor_base_sig_samples'])

            end_times.append(window_end_time)
            rel_impact_params.append(rel_impact_param)

            orig_ps_list.append(orig_ps)
            orig_xicrit_list.append(orig_xi_crit_used)
            orig_xicrit_factor_base_sig_list.append(orig_xicrit_factor_base_sig)
                

        del result

        np.save(f'{output_dir}/{output_filename_prefix_win}-ps_samples.npy', np.array(ps_arr_list))
        print('Time taken until saving [...]-ps_samples.npy: ', time.time() - start_time_process)
        np.save(f'{output_dir}/{output_filename_prefix_win}-orig_ps_list.npy', np.array(orig_ps_list))
        np.save(f'{output_dir}/{output_filename_prefix_win}-xicrit_samples.npy', np.array(xicrit_arr_list))
        np.save(f'{output_dir}/{output_filename_prefix_win}-xicrit_factor_base_sig_samples.npy', np.array(xicrit_factor_base_sig_arr_list))

        with h5py.File(f'{output_dir}/{output_filename_prefix_win}-ps_samples-plus.h5', 'w') as hdf5_ps_samples:
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-ps_samples', data=np.array(ps_arr_list))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-orig_ps_list', data=np.array(orig_ps_list))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-orig_xicrit_list', data=np.array(orig_xicrit_list))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-orig_xicrit_factor_base_sig_list', data=np.array(orig_xicrit_factor_base_sig_list))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-radius_list', data=np.array(radius_list))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-offset_radius', data=np.array(offset_radius))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-num_windows', data=np.array(num_windows))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-window_length', data=np.array(window_length))
            hdf5_ps_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-shift_length', data=np.array(shift_length))
            hdf5_ps_samples.close()
        with h5py.File(f'{output_dir}/{output_filename_prefix_win}-xicrit_samples.h5', 'w') as hdf5_xicrit_samples:
            hdf5_xicrit_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-xicrit_samples', data=np.array(xicrit_arr_list))
            hdf5_xicrit_samples.close()
        with h5py.File(f'{output_dir}/{output_filename_prefix_win}-xicrit_factor_base_sig_samples.h5', 'w') as hdf5_xicrit_factor_base_sig_samples:
            hdf5_xicrit_factor_base_sig_samples.create_dataset(f'{output_dir}/{output_filename_prefix_win}-xicrit_factor_base_sig_samples', data=np.array(xicrit_factor_base_sig_arr_list))
            hdf5_xicrit_factor_base_sig_samples.close()

        save_text_file(pattern_speed_medians, pattern_speed_means, pattern_speed_modes, pattern_speed_std,
                       pattern_speed_upper_lims, pattern_speed_lower_lims, pattern_speed_max, pattern_speed_min, 
                        orig_ps_list, orig_xicrit_list, orig_xicrit_factor_base_sig_list, 
                       rel_impact_params, end_times, 
                       function_parameter, sfiles_full_path, offset_x0, offset_y0, offset_radius,
                       num_windows, window_length, specific_output, output_filename_prefix_win)

    end_time_process = time.time()
    print('Total time taken: ', end_time_process - start_time_process)
