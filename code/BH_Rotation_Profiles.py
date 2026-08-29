############################## Chapter 0: Importing ##############################
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
import sys
from tqdm import tqdm
import glob
import time
import multiprocessing
from scipy.stats import truncnorm

def save_text_file(pattern_speed_medians, pattern_speed_means, pattern_speed_modes, pattern_speed_std, 
                   pattern_speed_upper_lims, pattern_speed_lower_lims, pattern_speed_max, pattern_speed_min, 
                    orig_ps_lists, orig_xicrit_lists, orig_xicrit_3sig_lists, 
                    rel_impact_params, end_times, 
                    function_parameter, sfiles, offset_x0, offset_y0, offset_radius, 
                    num_windows, window_length, specific_output, output_filename_prefix):
    
    with open(f"{specific_output}/{output_filename_prefix}pattern_speed_summary.txt", 'w') as f:
        f.write(f"Non-Kerr Pattern Speed\n")
        f.write(f"Path to first file used: {sfiles[0]}\n")
        f.write(f"GRRT input file format: {function_parameter}\n")
        f.write(f"Image offset x0 [M] = {offset_x0}, y0 [M] = {offset_y0}. \nReference impact parameter (offset_radius) [M]: {offset_radius}\n")
        f.write(f"Number of windows: {num_windows}\n")
        f.write(f"Length of sub windows: {window_length}\n")
        f.write(f"Specific output directory: {specific_output}\n\n")

        f.write(f"Window index, Impact parameter [GMc^(-2)], Impact parameter [b_ref], Pattern speed, Pattern speed min,"
                +"Pattern speed max, Pattern speed median,  Pattern speed median plus, Pattern speed median minus,"
                +"Pattern speed mean, Pattern speed std, Pattern speed mode, xi_crit, xi_crit_3sig\n")

        for idx in range(len(end_times)):
            str_to_write = f'{int(end_times[idx])}, {offset_radius*rel_impact_params[idx]:.6f}, {rel_impact_params[idx]:.6f}, {orig_ps_lists[idx]:.6f}, '
            str_to_write += f'{pattern_speed_min[idx]:.6f}, {pattern_speed_max[idx]:.6f}, {pattern_speed_medians[idx]:.6f}, {pattern_speed_upper_lims[idx]:.6f}, {pattern_speed_lower_lims[idx]:.6f}, '
            str_to_write += f'{pattern_speed_means[idx]:.6f}, {pattern_speed_std[idx]:.6f}, {pattern_speed_modes[idx]:.6f}, '
            str_to_write += f'{orig_xicrit_lists[idx]:.6f}, {orig_xicrit_3sig_lists[idx]:.6f}\n'
            f.write(str_to_write)


def plot_autocorrelation(racf, Q, pattern_speed, xi_crit, output_filename_prefix, plot_dir, dt, ntheta, nftot, window_length):
        extent = [-(0.5*nftot + 0.5)*dt,(0.5*nftot - 0.5)*dt,-(0.5*ntheta + 0.5)*dtheta,(0.5*ntheta - 0.5)*dtheta]
        if (nftot % 2) != 0: ## for an odd numer of frames, we shift by half a pixel right so that central peak is at \Delta t = 0. 
                extent = extent + np.array([+dt/2, +dt/2, 0,0]) 

        fig = plt.figure(figsize = (5,5))
        ax = plt.subplot(111)
        im = ax.imshow(racf.T, cmap='afmhot', aspect = 'auto', origin = 'lower',
            extent = extent,
            interpolation='bilinear', vmax = 1., vmin = -1.)
        
        ax.set_xlabel(r'$\Delta t\, [G M c^{-3}]$')
        ax.set_ylabel(r'$\Delta \mathrm{PA}\, [{\rm deg}]$')
        ax.set_title(f'Autocorrelation')
        # ax.set_xlim(-180,180)

        #Plot a line with slope equal to the pattern speed
        x_vals = np.arange(0,5*nftot, 1) * dt
        y_vals = pattern_speed * x_vals
        ax.plot(x_vals, y_vals, 'g--', alpha=0.6)
        plt.contour(Q.transpose(), extent=extent, origin='lower', levels=[0.5], colors='black')
        ax.set_ylim(-len(racf[0,:])*dtheta/2, len(racf[0,:])*dtheta/2.)
        ax.set_xlim(-len(racf[:,0])*dt/2, len(racf[:,0])*dt/2.)

        slope_label = mlines.Line2D([], [], color='green', linestyle='--', label=f'Extracted slope: {pattern_speed:0.4f} ' + r'${\rm deg} / GMc^{-3}$')
        xi_crit_label = mlines.Line2D([], [], color='black', label=r'$\xi_{\rm crit}$: ' + f'{xi_crit:0.4f}')
        plt.legend(handles=[slope_label, xi_crit_label])
        plt.savefig(plot_dir+f'{output_filename_prefix}_Autocorrelation_till_{window_length}M.png', bbox_inches = 'tight')
        plt.close(fig)

def plot_autocorrelation_distribution(racf, xi_crit, output_filename_prefix, plot_dir, window_length):
    fig, (ax) = plt.subplots(1, 1, dpi = 300, figsize = (5, 5))
    ax.hist(racf.flatten(), density = True)
    ax.axvline(xi_crit, linestyle = '--', color = 'black', label = 'xi_crit')
    ax.set_xlabel('autocorrelation')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(0., 10.)
    plt.legend(frameon = False)
    plt.savefig(plot_dir+f'{output_filename_prefix}_Autocorrelation_distribution_till_{window_length}M.png', bbox_inches = 'tight')
    plt.close(fig)


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


# # ###### Autocorrelation with windowing function
def compute_autocorrelation(qs, log_status):
    # Normalize cylinder
    qsn = np.copy(qs)

    if log_status:
        print('log taken')
        min_float_val = np.finfo(np.float64).tiny
        qsn[qsn < min_float_val] = min_float_val
        qsn = np.log10(qsn) ## note that log can produce nans if values in qs are too small or negative   

    # Mean subtract
    # Remove mean from each angle (column)
    qsn = qsn - np.mean(qsn, axis=0, keepdims=True)
    # Remove mean from each time (row)
    qsn = qsn - np.mean(qsn, axis=1, keepdims=True)
    
    qsn = tukey_window(qsn, alpha=0.2)

        
    # FFT
    qk = fft.fft2(qsn)
    Pk = np.absolute(qk)**2
    acf = np.real(fft.ifft2(Pk))
    acf = acf / acf[0,0] # normalize
    
    # Shift
    shifti = int(acf.shape[0]/2.)
    shiftj = int(acf.shape[1]/2.)
    racf = np.roll(acf, (shifti, shiftj), axis=(0, 1))

    return racf, qsn


######### Time Window tapering function ###########
def tukey_window(qsn, alpha=0.2):
  """
  Apply a Tukey (tapered cosine) taper/window to the cylinder plot, to avoid incorrectly wrapping in time when autocorrelating

  Args:
      qsn: The smoothed normalized cylinder data (time x angle)
      alpha: The Tukey parameter, setting what fraction of timeslices to taper. alpha=0.2 is best.
  """
  nftot = qsn.shape[0]
  window = tukey(nftot , alpha) ## define window. 

  qsn_windowed = np.zeros(qsn.shape) ## apply window
  ntheta_window = qsn.shape[1]
  for theta_i in range(ntheta_window):
    qsn_row = qsn[:, theta_i] * window

    qsn_windowed[:, theta_i] = qsn_row

  qsn_windowed = qsn_windowed - np.mean(qsn_windowed) ## ensure mean zero

  return qsn_windowed

def sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, dtheta = 2., x_shift=0, y_shift=0, r_shift=0, raytracer = 'Raptor'):
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

def calculate_pattern_speed(racf, dt, dtheta=2.0, xi_crit_factor = 3.0):

    xi_crit_3sig = xi_crit_factor*np.std(racf)
    xi_crit = xi_crit_3sig
    racf_cut = np.copy(racf) ## define racf_cut, which we'll use for the Omega_p calculation
    ts = np.linspace(-len(racf)/2, len(racf)/2, len(racf), endpoint = False)
    phis = np.linspace(-len(racf[0])/2, len(racf[0])/2, len(racf[0]), endpoint = False)

    central_row_index = racf.shape[0] // 2
    central_column_index = (racf.shape[1]) // 2


### Make sure no noise external to the central peak is included in the calculation. filter external noise using 'labels'

    labels, num_features = label((racf > xi_crit).astype(int)) ## label every feature > cut
    Q = labels == labels[central_row_index, central_column_index] ## create mask for central region
    non_zero_columns = np.count_nonzero( np.sum(Q, axis=0) )
    non_zero_rows = np.count_nonzero(np.sum(Q, axis=1))

    min_required_columns = 5
    min_required_rows = 5

    if (xi_crit > 1) or (non_zero_columns < min_required_columns) or (non_zero_rows < min_required_rows):
        xi_crit, Q = lower_xicrit_threshold(racf, xi_crit, non_zero_columns, non_zero_rows)

    # Moments
    moment_t = 0
    moment_t_phi = 0
    
    # Apply mask
    racf_cut[~Q] = 0.0
    
    # Calculate moments
    # Meshgrid
    T_mesh, Phi_mesh = np.meshgrid(ts, phis, indexing='ij')
    
    moment_t = np.sum(racf_cut * T_mesh**2)
    moment_t_phi = np.sum(racf_cut * T_mesh * Phi_mesh)
    
    # Normalize moments (dividing by moment cancels out the delta_t*delta_phi factors in the ratio)
    if moment_t == 0:
        pattern_speed = 0
    else:
        pattern_speed = moment_t_phi / moment_t
        # Units
        pattern_speed = pattern_speed * dtheta / dt

    return pattern_speed, xi_crit, xi_crit_3sig


def run_mc(start_idx, end_idx, sIall, N, M, ring_params, dx, dt, n_samples, xi_crit_factor_base, window_end_time, rel_impact_param, log_status = True, raytracer='Raptor'):
    # Setup MC
    # Sigma fit from xi_crit RMSE curves

    print("Calculating unperturbed pattern speed")
    qs = sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, dtheta = 2.,
                           x_shift=0., 
                           y_shift=0., 
                           r_shift=0., raytracer=raytracer)
    racf, _ = compute_autocorrelation(qs, log_status)

    orig_ps, orig_xi_crit_used, orig_xi_crit_3sig = calculate_pattern_speed(racf, dt, dtheta=2.0, xi_crit_factor = xi_crit_factor_base)


    print("Sampling the perturbed realizations")
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
    xi_crit_3sig_samples = []
    
    print("Computing pattern speeds for different realizations")
    for i in range(n_samples):
        # Sample cylinder with perturbed parameters
        qs = sample_cylinder(start_idx, end_idx, sIall, ring_params, dx, 
                           x_shift=x_samples[i], 
                           y_shift=y_samples[i], 
                           r_shift=r_samples[i], raytracer=raytracer)
        
        # Compute Autocorr
        racf, _ = compute_autocorrelation(qs, log_status)
        
        # Calculate pattern speed
        # Convert absolute sample to factor for this specific realization
        current_std = np.std(racf)
        factor = xi_crit_abs_samples[i] / current_std
        
        ps, xi_crit_used, xi_crit_3sig = calculate_pattern_speed(racf, dt, dtheta=2.0, xi_crit_factor = factor)
        ps_samples.append(ps)
        xi_crit_used_samples.append(xi_crit_used)
        xi_crit_3sig_samples.append(xi_crit_3sig)
        
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
        'xicrit_3sig_samples': xi_crit_3sig_samples,
        'mean': mean_ps,
        'std': std_ps,
        'median': median,
        'median_plus_sigma': median_plus,
        'median_minus_sigma': median_minus,
        'max': max_ps,
        'min': min_ps,
        'mode': mode_value
    }, window_end_time, rel_impact_param, orig_ps, orig_xi_crit_used, orig_xi_crit_3sig
  
############################## Chapter 1: Defining Functions and Paramters ##############################
######## Chapter 1.1: Defining Functions
### gnw colorbar implementation

def plot_I_slice(Iall, plot_dir, xc, yc, plot_radius, n=0):
    plt.figure(figsize=(6, 5))
    plt.imshow(Iall[:, :, n], cmap='afmhot')
    plt.colorbar(label='Intensity')
    plt.title(f'Iall[:, :, {n}]')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.tight_layout()
    plt.savefig(plot_dir + f'{output_filename_prefix}Iall_n_{n}.png')
    plt.close()
  
    fig, (ax) = plt.subplots(1, 1, figsize = (8, 8), dpi = 300)
    Iavg = np.mean(Iall, axis=2)
    im = ax.imshow(Iavg, cmap='afmhot')
    
    circle = patches.Circle((xc,  yc), plot_radius, color = 'green', fill = False, label = 'shadow')

    ax.add_patch(circle)
    ax.scatter(xc, yc, color = 'green', s = 3)

    fig.colorbar(im, ax=ax, label='Intensity')
    ax.set_title(f'Iall[:, :] avg')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.tight_layout()
    plt.savefig(plot_dir + f'{output_filename_prefix}Iall_avg.png')
    plt.close()



def colorbar(mappable):
    """ the way matplotlib colorbar should have been implemented """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)



## deals with exactly 0 edge cases
def myinterp(dat, x, y):
    i = int(round(x-0.5))
    j = int(round(y-0.5))
    ### bilinear:
    di = x-i
    dj = y-j
    z00 = dat[i,j]
    z01 = dat[i,j+1]
    z10 = dat[i+1,j]
    z11 = dat[i+1,j+1]
    idat = z00*(1.-di)*(1.-dj) + z01*(1.-di)*dj + z10*di*(1.-dj) + z11*di*dj
    return idat

### Definition of different file format readers: ######## Chapter 2.2: Reading Files

def Ipole(starting_string):
    files = [f for f in os.listdir(list_directory) if starting_string in f]
    
    # Check if any files were found before proceeding
    if not files:
        print(f"No files found matching pattern: {starting_string}")
        return

    # Sort the list of full paths
    sfiles = sorted(files)
    
    # Use f-strings for cleaner printing and handle cases with 1 file
    nftot = len(sfiles)
    if nftot > 1:
        print(f"{nftot} files found, from {sfiles[0]} to {sfiles[-1]}")
    else:
        print(f"{nftot} file found: {sfiles[0]}")

    ### set some parameters for the run, based on first file in sequence
    hfp = h5py.File(list_directory + '/' + sfiles[0],'r')
    FOV_M = hfp['header']['camera']['dx'][()] ## image scale in M
    FOV_uas =hfp['header']['camera']['fovx_dsource'][()]
    if  hfp['header']['camera']['fovx_dsource'][()] != hfp['header']['camera']['fovy_dsource'][()]:
        print("Error: need a square image. This image appears to be {0:n} x {1:n} pixels".format(hfp['header']['camera']['fovx_dsource'][()], hfp['header']['camera']['fovy_dsource'][()]))
        sys.exit(0)
    scale = hfp['header']['scale'][()]        ## converts pixel to flux density in Jy
    imagep = np.copy(hfp['pol']).transpose((1,0,2))
    N = imagep.shape[0]                       ## x-size of array
    M = imagep.shape[1]                       ## y-size of array
    dx = FOV_uas/N                            ## x-size of pixel
    dy = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.

    try:
        spin = hfp['fluid_header']['a'][()]
    except KeyError:
        spin = 0.5
    inclination = hfp['header']['camera']['thetacam'][()]

    ### We find dt by reading in header/t [which is in units of M] from the first couple files.
    file1 = sfiles[0]
    file2 = sfiles[1]
    file3 = sfiles[2]
    file4 = sfiles[3]

    hf1 = h5py.File(list_directory + '/' + file1, 'r')
    hf2 = h5py.File(list_directory + '/' + file2, 'r')
    hf3 = h5py.File(list_directory + '/' + file3, 'r')
    hf4 = h5py.File(list_directory + '/' + file4, 'r')

    dt2 = hf2[('header/t')][()] - hf1[('header/t')][()]
    dt3 = hf3[('header/t')][()] - hf2[('header/t')][()]
    dt4 = hf4[('header/t')][()] - hf3[('header/t')][()]
    dt = (dt2 + dt3 + dt4)/3

    hf1.close()
    hf2.close()
    hf3.close()
    hf4.close()

    dt = round(dt, 1)

    Tmax = nftot*dt

    ### Allocate space for 2-D Stokes I image for each file
    Iall = np.zeros((N, M, nftot))

    ### Loop over all the files to define Iall, which contains the Stokes I data from every frame
    n = 0
    print("Reading files from " + str(list_directory) + "...")
    for fname in tqdm(sfiles): ## read in file
        hfp = h5py.File(list_directory + '/' + fname,'r')
        imagep = np.copy(hfp['pol']).transpose((1,0,2))
        Iall[:,:,n] = imagep[:,:,0]
        hfp.close()
        n += 1
    return nftot, sfiles, FOV_M, FOV_uas, N, M, dx, dy, half_FOV, spin, inclination, dt, Tmax, Iall


def Raptor(starting_string): # !dt = 10 hardcoded! / (un-)comment lines whether model.in is present or not
    files = [f for f in os.listdir(list_directory) if starting_string in f]

    nftot = len(files)
    sfiles = sorted(files)[:nftot]   ## sort because OS returns seemingly random order
    print(str(nftot) + " files, from " + sfiles[1] + " to " + sfiles[-1])
    slash='/'
    full_file_names = []
    for i in np.arange(0, len(sfiles), 1):
        file_names = list_directory+slash+sfiles[i]
        full_file_names.append(file_names)
    sfiles = full_file_names
    print("Full paths: from " + sfiles[1] + " to " + sfiles[-1])
    #####

    # raptor_input = np.genfromtxt('model.in')  ##### Use this only if model.in is present
    data = np.genfromtxt(sfiles[0])
    # FOV_M = raptor_input[8][2]   ##### Use this only if model.in is present
    FOV_M = 39.8   ##### Use this as I do not have model.in
    FOV_uas = 2*(data[0][0]) * 1000 * 1000 #from arc seconds to micro arcsecs
    N = int(np.sqrt(data.shape[0]))
    M = int(np.sqrt(data.shape[0]))    
    dx = FOV_uas/N                            ## x-size of pixel
    dy = FOV_uas/N                            ## y-size of pixel
    half_FOV = FOV_uas/2.
    # inclination = raptor_input[5][2]
    inclination = 30.  ##### Use this as I do not have model.in

    ## HARDCODED!
    spin = 15./16
    # spin = 0.5
    dt = 10
    Tmax = nftot*dt
    Iall = np.zeros((N,N, nftot))
    n=0

    # Open the file and read the lines
    for file in tqdm(sfiles):
        with open(file, 'r') as currf:
            first_row = currf.readline().strip('\n')

        num_skiprows = 0
        if 'X' in first_row:
            num_skiprows = 1
            
        usecols_num = 2
        all_data = np.loadtxt(file, unpack = True, usecols = usecols_num, skiprows = num_skiprows)
        Iall[:, :, n] = np.flipud((all_data.reshape(N, N)).T)
        n += 1
    return nftot, sfiles, FOV_M, FOV_uas, N, M, dx, dy, half_FOV, spin, inclination, dt, Tmax, Iall


def BHOSS_1(starting_string):
    files = [f for f in os.listdir(list_directory) if starting_string in f]

    nftot = len(files)
    sfiles = sorted(files)[:nftot]   ## sort because OS returns seemingly random order
    # print(sfiles)
    print(str(nftot) + " files, from " + sfiles[1] + " to " + sfiles[-1])
    slash='/'
    full_file_names = []
    for i in np.arange(0, len(sfiles), 1):
        file_names = list_directory+slash+sfiles[i]
        full_file_names.append(file_names)
    sfiles = full_file_names
    # print(sfiles)
    print("Full paths: from " + sfiles[1] + " to " + sfiles[-1])

    datah1 = np.genfromtxt(sfiles[0], max_rows=1)
    datah2 = np.genfromtxt(sfiles[0], skip_header=1, max_rows=1)
    datah3 = np.genfromtxt(sfiles[0], skip_header=2, max_rows=1)
    #data = np.genfromtxt(file, skip_header=3)
    FOV_M = datah1[0]*2
    FOV_uas = datah2[-1] * FOV_M
    N = int(datah1[2])
    M = int(datah1[2])
    dx = FOV_uas/N                            ## x-size of pixel
    dy = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.
    spin = datah2[3]
    inclination = datah2[1]
    
    datah2_file2 = np.genfromtxt(sfiles[1], skip_header=1, max_rows=1)
    datah2_file3 = np.genfromtxt(sfiles[2], skip_header=1, max_rows=1)
    datah2_file4 = np.genfromtxt(sfiles[3], skip_header=1, max_rows=1)
    dt2 = datah2_file2[0] - datah2[0]
    dt3 = datah2_file3[0] - datah2_file2[0]
    dt4 = datah2_file4[0] - datah2_file3[0]
    dt = (dt2 + dt3 + dt4)/3

    dt = round(dt, 1)
    Tmax = nftot*dt


    Iall = np.zeros((N,M, nftot))
    n=0

    # Open the file and read the lines
    for file in tqdm(sfiles):
        with open(file, 'r') as f:
            data_lines = f.readlines()
            lines = data_lines[3:]

        for i, line in enumerate(lines):
            # Split the line into columns (x, y, I)
            values = line.split()
            
            # Extract the I value (third column) and convert it to a float
            I_value = float(values[3])
            
            # Calculate the column and row indices in the matrix
            col = int(values[1]) -1
            row =  int(values[0]) -1
            Iall[row, col][n] = I_value 
        n+=1
    #Iall = Iall.transpose((1,0,2))
    return nftot, sfiles, FOV_M, FOV_uas, N, M, dx, dy, half_FOV, spin, inclination, dt, Tmax, Iall


def BHOSS_2(starting_string): # !dt = 10 hardcoded!

    files = None
    if starting_string == 'GRRT':
        files = [f for f in os.listdir(list_directory) if starting_string in f]
    else:
    ##### CHANGED FOR JP ONLY, CAN REMOVE LATER ########
        files = [f for f in os.listdir(list_directory) if starting_string not in f]

    nftot = len(files)
    sfiles = sorted(files)[:nftot]   ## sort because OS returns seemingly random order
    print(str(nftot) + " files, from " + sfiles[1] + " to " + sfiles[-1])
    slash='/'
    full_file_names = []
    for i in np.arange(0, len(sfiles), 1):
        file_names = list_directory+slash+sfiles[i]
        full_file_names.append(file_names)
    sfiles = full_file_names
    print("Full paths: from " + sfiles[1] + " to " + sfiles[-1])

    #####
    datah1 = np.genfromtxt(sfiles[0], skip_header=1, max_rows=1)
    datah8 = np.genfromtxt(sfiles[0], skip_header=8, max_rows=1)
    datah13 = np.genfromtxt(sfiles[0], skip_header=13, max_rows=1)
    datah9 = np.genfromtxt(sfiles[0], skip_header=9, max_rows=1)
    #data = np.genfromtxt(file, skip_header=3)
    FOV_M = float(datah8)*2.
    print(FOV_M)
    FOV_uas = datah13[0] * FOV_M 
    N = int(datah9)
    M = int(datah9)
    dx = FOV_uas/N                            ## x-size of pixel
    dy = FOV_uas/M                            ## y-size of pixel
    half_FOV = FOV_uas/2.
    spin = datah1[1]
    inclination = datah1[3]
    
    ## BHOSS_2 does show the timestamp in header!
    # datah1_file2 = np.genfromtxt(sfiles[1], skip_header=1, max_rows=1)
    # datah1_file3 = np.genfromtxt(sfiles[2], skip_header=1, max_rows=1)
    # datah1_file4 = np.genfromtxt(sfiles[3], skip_header=1, max_rows=1)
    # dt2 = datah1_file2[0] - datah1[0]
    # dt3 = datah1_file3[0] - datah1_file2[0]
    # dt4 = datah1_file4[0] - datah1_file3[0]
    # dt = (dt2 + dt3 + dt4)/3
    
    ## HARDCODED!
    dt = 10

    dt = round(dt, 1)
    Tmax = nftot*dt

    Iall = np.zeros((N,M, nftot))
    n=0

    # Open the file and read the lines
    for file in tqdm(sfiles):
        line_counter = 0 
        with open(file, 'r') as f:
            data_lines = f.readlines()
            lines = data_lines[17:]

        #for i, line in enumerate(lines):
        for col in range (N):
            for row in range (M):

            # Split the line into columns (x, y, I)
                I_value = float(lines[line_counter].strip())

                Iall[row, col][n] = I_value
                line_counter+=1
            
        n+=1
    #Iall = Iall.transpose((1,0,2))
    return nftot, sfiles, FOV_M, FOV_uas, N, M, dx, dy, half_FOV, spin, inclination, dt, Tmax, Iall


def Fits(starting_string):
    files = [f for f in os.listdir(list_directory) if starting_string in f]

    nftot = len(files)
    sfiles = sorted(files)[:nftot]   ## sort because OS returns seemingly random order
    print(str(nftot) + " files, from " + sfiles[1] + " to " + sfiles[-1])
    slash='/'
    full_file_names = []
    for i in np.arange(0, len(sfiles), 1):
        file_names = list_directory+slash+sfiles[i]
        full_file_names.append(file_names)
    sfiles = full_file_names
    print("Full paths: from " + sfiles[1] + " to " + sfiles[-1])
    #####
    hdul0 = fits.open(sfiles[0])
    hdul1 = fits.open(sfiles[1])
    hdul2 = fits.open(sfiles[2])
    hdul3 = fits.open(sfiles[3])
    #### CHANGE: FOV_M to 60 instead of 400 which was present before
    FOV_M = 60 ########################################## ENTER FOV_M 
    N = hdul0[0].header['NAXIS1']
    M = hdul0[0].header['NAXIS2']
    dx = abs(hdul0[0].header['CDELT1']) * 206265 * 1e6                      
    dy = dx
    FOV_uas = M * dx                       
    half_FOV = FOV_uas/2.
    spin = hdul0[0].header['SPIN']
    inclination = hdul0[0].header['INC']
    
    
    dt2 = hdul1[0].header['TIME'] - hdul0[0].header['TIME']
    dt3 = hdul2[0].header['TIME'] - hdul1[0].header['TIME']
    dt4 = hdul3[0].header['TIME'] - hdul2[0].header['TIME']
    dt = (dt2 + dt3 + dt4)/3
    dt = round(dt, 1) 
    print("delta T:", dt) ########################Check if this indeed gets the correct dt
    Tmax = nftot*dt

    Iall = np.zeros((N,M, nftot))
    n=0
    # Open the file and read the lines
    for n, file in enumerate(tqdm(sfiles)):
        img = fits.getdata(file)  # returns a 2D numpy array (y, x)
        Iall[:, :, n] = img      
    Iall = Iall.transpose((1,0,2))
    return nftot, sfiles, FOV_M, FOV_uas, N, M, dx, dy, half_FOV, spin, inclination, dt, Tmax, Iall



function_map = {
        "Ipole": Ipole,
        "Raptor": Raptor,
        "BHOSS_1": BHOSS_1,
        "BHOSS_2": BHOSS_2,
        "Fits": Fits
    }


############################## Chapter 2: Reading Files and calculating cylinder plots ##############################
# try:
if True:

    nan = float('nan')

   ######## Chapter 2.1: Defining File Paths and Image Parameters
    
    parser = argparse.ArgumentParser(description="Pattern Speed Super Script")

    # Required positional arguments
    parser.add_argument("function_parameter", help="Name of the function to run (i.e., to choose from BHOSS_1, BHOSS_2, Fits, Ipole, Raptor)")
    parser.add_argument("list_directory", help="Directory containing files to process")
    parser.add_argument("starting_string", help="String to identify files")
    parser.add_argument("specific_output", help="Directory for saving script output")

    # Optional argument: radius list / not used for Non-Kerr purposes
    parser.add_argument("--radius_list", nargs="+", type=float, default=None,
                        help="impact parameters/image radii on which to calculate the pattern speed, in units of M. Overwritten if --args.offset_radius is provided. Use: start end (#steps+1) (e.g., --radius_list 1 9 5)")

    parser.add_argument("--offset", nargs="+", type=float, default=None,
                        help="offset x and y in units of M (e.g., --offset 1.2 -0.7)")
    
    parser.add_argument("--offset_radius", nargs="+", type=float, default=None,
                        help="reference impact parameter/image radius on which to calculate the pattern speed, in units of M. Use: radius (e.g., --offset_radius 5.223)")

    # parser.add_argument("--offset_err", nargs="+", type=float, default=None,
    #                     help="offset error x and y in units of M (e.g., --offset 1.2 -0.7)")
    
    # parser.add_argument("--offset_radius_err", nargs="+", type=float, default=None,
    #                     help="error in reference impact parameter/image radius on which to calculate the pattern speed, in units of M. Use: radius (e.g., --offset_radius 5.223)")

    parser.add_argument("--output_filename_prefix", nargs="+",
                        help="string added in front of output files to associate them with the model investigated and the imaging parameters. Use: string (e.g., for the Kerr reference of the Dilaton_2 model: --output_filename_prefix Dilaton_2-Kerr_a06_i150_Rh160)")

    parser.add_argument("--window_length", nargs="+", type = int, default = 500,
                        help="Length of subwindow in M")

    parser.add_argument("--log_status", nargs="+", type = str, default = 'True',
                        help="Whether to take log of smoothed cylinder before computing autocorrelations")

    parser.add_argument("--ordering", nargs="+", type = str, default = 'forward',
                        help="Windows increasing in length from 0 or end of simulation")

    parser.add_argument("--ring_params_filename", nargs="+", type = str,
                        help = "path to file withring center and raii for the simulation")
    


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
    

    radius_list_addition_relative = 0
    radius_list_relative = 0

    if args.offset_radius:
        offset_radius = float(args.offset_radius[0])
        radius_list_addition_relative = [round(0.5 + i * 0.05, 2) for i in range(41)]
        radius_list = [offset_radius * radius_relative for radius_relative in radius_list_addition_relative]
        radius_list_addition_absolute = [round(1.0 + i * 0.5, 2) for i in range(33)]
        radius_list.extend(radius_list_addition_absolute)
        radius_list = np.insert(radius_list, 0, np.sqrt(27))
        radius_list = np.sort(radius_list)
        radius_list_relative = [radius_absolute/offset_radius for radius_absolute in radius_list]
    else:
        offset_radius = 0
    
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
        print('r_err: ', r_err)
        print('x_err: ', x_err)
        print('y_err: ', y_err)
        print('rc_err: ', rc_err)

    if args.output_filename_prefix:
        output_filename_prefix = args.output_filename_prefix[0]

    else:
        output_filename_prefix = ""

    if args.window_length:
        window_length = args.window_length
    
    log_status = args.log_status[0]
    window_ordering = args.ordering[0]

    print('log status: ', log_status)
    print('window_ordering: ', window_ordering)


    print("Function:", function_parameter)
    print("List directory:", list_directory)
    print("Starting string:", starting_string)
    print("Specific output:", specific_output)
    if args.offset:
        print("Offset x0/M:", offset_x0, "and offset y0/M:", offset_y0)

    if radius_list_addition_relative:
        print("First group of impact parameters normalized to reference:", radius_list_addition_relative)
    print("All impact parameters [GMc^-2]:", radius_list)

    plot_dir = f"{specific_output}/cylinder_plots/"
    output_dir = f"{specific_output}/cylinder_output/"
    image_dir = f"{specific_output}/simulation_images/"
    smoothed_image_dir = f"{specific_output}/simulation_images_smoothed/"

    try:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(smoothed_image_dir):
            os.makedirs(smoothed_image_dir)
    except:
        print("Directories already exist")

    ### Load the files
    if function_parameter in function_map:
        nftot, sfiles, FOV_M, FOV_uas, N, M, dx, dy, half_FOV, spin, inclination, dt, Tmax, Iall = function_map[function_parameter](starting_string)

        shift_length = 50
        window_length = 550
        num_windows = 1 + int((Tmax - window_length)/shift_length)
        num_window_snaps = int(window_length/dt)

        print(f'Number of Windows: {num_windows}')
        print(f'Number of Window Snaps: {num_window_snaps}')
        print(f'Window Length: {window_length}')

        if args.offset:
            xc = N/2 + offset_x0 * (N/FOV_M) ## Note: the final factor is a unit conversion. We want in pixel units ## 2*spin*np.sin(inclination * np.pi/180) *(N/FOV_M)
            yc = M/2 + offset_y0 * (M/FOV_M)

        if function_parameter == 'Raptor':
            yc = M - 1 - yc
        plot_I_slice(Iall, image_dir, xc, yc, offset_radius*(M/FOV_M), n=0)
        plot_I_slice(Iall, smoothed_image_dir, xc, yc, offset_radius*(M/FOV_M), n=0)

        print(f"Loaded in data!")

    else:
        print(f"Error: Function '{function_parameter}' not recognized. Function parameters are Ipole, RAPTOR, BHOSS_1, BHOSS_2, Fits.")
        sys.exit(1)

    ### Smooth
    sig = 20/(dx *(2*np.sqrt(2*np.log(2)))) ## We have 20 uas FWHM resolution. dx = uas/pixels. so 20/dx is FWHM in pixel units.
    smoothed_Iall = ndimage.gaussian_filter(Iall, sigma=(sig,sig,0))
    del Iall

    plot_I_slice(smoothed_Iall, smoothed_image_dir, xc, yc, offset_radius*(M/FOV_M), n=0)

    cores_per_job = 1
    num_cores = int((multiprocessing.cpu_count())/(cores_per_job))
    count = 0


    ###### MC params
    n_samples = 1000
    xi_crit_factor_base = 3.0
    dx, dy = FOV_M/N, FOV_M/M
    start_time = time.time()

    if window_ordering == 'forward' and log_status == 'True':
        for curr_window_ordering, curr_log_status in [('forward', True)]:
            start_time_list = [i*shift_length for i in range(num_windows)]
            end_time_list = [i*shift_length + window_length for i in range(num_windows)]

            output_filename_prefix += f'-{curr_window_ordering}_log_{str(curr_log_status)}-'

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
                    inputs.append((int(start_time_list[curr_window_idx]/dt), int(end_time_list[curr_window_idx]/dt), smoothed_Iall, N, M, ring_params, dx, dt, n_samples, xi_crit_factor_base, int(end_time_list[curr_window_idx]), radius_list_relative[b_idx], log_status, function_parameter))
        
            print(f'{len(inputs) }Inputs loaded for {curr_window_ordering} and log {curr_log_status}')

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
            xicrit_3sig_arr_list = []

            orig_ps_lists = []
            orig_xicrit_lists = []
            orig_xicrit_3sig_lists = []

            for el in result:
                mc_res, window_end_time, rel_impact_param, orig_ps, orig_xi_crit_used, orig_xicrit_3sig = el

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
                xicrit_3sig_arr_list.append(mc_res['xicrit_3sig_samples'])

                end_times.append(window_end_time)
                rel_impact_params.append(rel_impact_param)

                orig_ps_lists.append(orig_ps)
                orig_xicrit_lists.append(orig_xi_crit_used)
                orig_xicrit_3sig_lists.append(orig_xicrit_3sig)
                

            del result

            np.save(f'{output_dir}/{output_filename_prefix}ps_samples.npy', np.array(ps_arr_list))
            np.save(f'{output_dir}/{output_filename_prefix}xicrit_samples.npy', np.array(xicrit_arr_list))
            np.save(f'{output_dir}/{output_filename_prefix}xicrit_3sig_samples.npy', np.array(xicrit_3sig_arr_list))

            save_text_file(pattern_speed_medians, pattern_speed_means, pattern_speed_modes, pattern_speed_std,
                           pattern_speed_upper_lims, pattern_speed_lower_lims, pattern_speed_max, pattern_speed_min, 
                           orig_ps_lists, orig_xicrit_lists, orig_xicrit_3sig_lists, 
                           rel_impact_params, end_times, 
                           function_parameter, sfiles, offset_x0, offset_y0, offset_radius,
                           num_windows, window_length, specific_output, output_filename_prefix)
    end_time = time.time()
    print('Time taken: ', end_time - start_time)
