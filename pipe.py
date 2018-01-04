#!/usr/bin/env python

""" main.py: The main script to trigger the MAPS pipeline for SIP.
    PYTHONPATH=''
    PYTHONPATH=/Users/jamie/Data/ARL/algorithm-reference-library:/Users/jamie/Data/CASACORE/casacore
"""

import os

import time as t

import numpy as np

from arl.visibility.operations import append_visibility

from sip.telescopetools.initinst import init_inst
from sip.accretion.ms import load
from sip.metamorphosis.filter import uv_cut, uv_advice
from sip.metamorphosis.convert import convert_to_stokes
from sip.outflows.images.imaging import wstack
from sip.outflows.rmsynthesis.rmsynth import load_im_data, rmsynth_advice, do_rmsynth, rmcube_save_to_disk
from sip.outflows.rmsynthesis.rmclean import rmclean_prep, do_rmclean
from sip.eventhorizon.plot import uv_cov, uv_dist, plt_rmsf

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


START = t.time()

# Manually specify the data directories.
INPUTS_DIR = '/Users/jamie/Data/SIP-Data/inputs'
RESULTS_DIR = '/Users/jamie/Data/SIP-Data/outputs'
os.makedirs(RESULTS_DIR, exist_ok=True)

MAKE_PLOTS = False


# Automatically adjust the appropriate parameters depending upon the instrument
INST = 'LOFAR'
POLDEF = init_inst(INST)


for channel in range(0, 40):  # loop over chans, only one subband for now
    # Load MSSS snapshots into memory for a single channel:
    vis = load('%s/H033+41/set03/135993/BAND0/L135737_SAP001_BAND0.MS/'
               % (INPUTS_DIR), channel, POLDEF)
    other_vis = load('%s/H033+41/set03/136000/BAND0/L135739_SAP001_BAND0.MS/'
                    % (INPUTS_DIR), channel, POLDEF)

    # Combine MSSS snapshots:
    vis = append_visibility(vis, other_vis)

    # Apply a filter to the uv-data:
    uv_cutoff = 450.0  # approx. MAPS-Mid resolution (for now)
    vis = uv_cut(vis, uv_cutoff)
    npixel_advice, cell_advice = uv_advice(vis, uv_cutoff)

    # Make some basic plots:
    if MAKE_PLOTS:
        uv_cov(vis)
        uv_dist(vis)

    # Calibrate the Ionospheric Faraday Rotation Correction:
    # ------------------------------------------------------
    # ra, dec = get_phase_centre(ms_name)
    # time = get_observing_date(ms_name)
    # corrections = download_tec_data(ra, dec, time, "code")
    # log = apply_correction(corrections)

    # Image I, Q, U, V, per channel
    # (no beam model, clean, blanking beyond pb, is currently implemented):
    vis = convert_to_stokes(vis, POLDEF)  # convert from XX/XY/YX/YY to I/Q/U/V.
    wstack(vis, npixel_advice, cell_advice, channel, RESULTS_DIR)
    # Not all at fixed resolution.

    # Deconvolve (should be done for pol. using LP+MJH method)


# Mosaic. Using Montage and/or a reimplementation.
# ------------------------------------------------------
# Make mosaicing directory structure.
# Create a table of all the files to be mosaiced.
# Investigate the template header for the desired reprojection of the files.
# Add a Frequency header to each output FITS image header.
# Do the reprojection. Can execute using MPI or another framework.
# Make a table of all the reprojected images.
# Add all of the reprojected images into a single mosaic.
# Do it robustly, and apply weighting somehow. (Can also execute using MPI).


# RM Synthesis
# ------------------------------------------------------
# Load in the FITS images:
IMAGECUBE, FREQUENCY, WEIGHTS = load_im_data(RESULTS_DIR)

# Calculate the appropriate parameters for these data:
LAMBDASQ, LAMBDA0, RMSF_EST, SCALE_EST, MAXRM_EST, CELLSIZE, NPIXELS, PHI = \
                                            rmsynth_advice(FREQUENCY, WEIGHTS)

# Form complex (Q+iU) from the polarisation data:
COMPLEX_P = 1j*IMAGECUBE.data[:, 2, :, :]
COMPLEX_P += IMAGECUBE.data[:, 1, :, :]

# Perform the RM Synthesis:
RMSYNTH, RMSF, RA_LEN, DEC_LEN = do_rmsynth(WEIGHTS, PHI, COMPLEX_P, LAMBDASQ, LAMBDA0)

# Plot the RMSF:
if MAKE_PLOTS:
    plt_rmsf(PHI, RMSF)

# Save RM Synthesis parameters to disk:
np.savetxt('%s/phi.txt' % (RESULTS_DIR), PHI)
np.savetxt('%s/rmsf.txt' % (RESULTS_DIR), RMSF)
np.savetxt('%s/weights.txt' % (RESULTS_DIR), WEIGHTS)

# Save the dirty cubes to disk:
rmcube_save_to_disk(RMSYNTH, CELLSIZE, MAXRM_EST, 'abs', RESULTS_DIR, 'dirty')
rmcube_save_to_disk(RMSYNTH, CELLSIZE, MAXRM_EST, 'real', RESULTS_DIR, 'dirty')
rmcube_save_to_disk(RMSYNTH, CELLSIZE, MAXRM_EST, 'imag', RESULTS_DIR, 'dirty')

# Calculate the RM-clean threshold:
RMSF_DOUBLE, CLEAN_THRESHOLD = rmclean_prep(RMSYNTH, MAXRM_EST, NPIXELS, WEIGHTS, LAMBDASQ, LAMBDA0)

# Deconvolve the RM-cube:
RMCLEAN = do_rmclean(RMSYNTH, PHI, RMSF, RMSF_DOUBLE, RMSF_EST, CLEAN_THRESHOLD, RA_LEN, DEC_LEN, CELLSIZE)

# Save the clean cubes to disk:
rmcube_save_to_disk(RMCLEAN, CELLSIZE, MAXRM_EST, 'abs', RESULTS_DIR, 'clean')
rmcube_save_to_disk(RMCLEAN, CELLSIZE, MAXRM_EST, 'real', RESULTS_DIR, 'clean')
rmcube_save_to_disk(RMCLEAN, CELLSIZE, MAXRM_EST, 'imag', RESULTS_DIR, 'clean')


# Source Finding
# ------------------------------------------------------
# Calculate Moments?
# Apply python source-finding implementation?
# Apply SB neutral network algorithm?
# Output source catalogue: ra, dec, RM, PI, etc.

END = t.time()
print(END-START)
