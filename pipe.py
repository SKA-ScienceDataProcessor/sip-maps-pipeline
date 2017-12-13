#!/usr/bin/env python

""" main.py: The main script to trigger the MAPS pipeline for SIP.
    PYTHONPATH=''
    PYTHONPATH=/Users/jamie/Data/ARL/algorithm-reference-library:/Users/jamie/Data/CASACORE/casacore
"""

import os

import time as t

import numpy as np

from arl.visibility.operations import append_visibility

from sip.telescopetools.initinst import initinst
from sip.accretion.ms import load
from sip.metamorphosis.filter import uvcut, uvadvice
from sip.metamorphosis.convert import converttostokes
from sip.outflows.images.imaging import wstack
from sip.outflows.rmsynthesis.rmsynth import loadimdata, rmsynth_advice, dormsynth, rmcube_savetodisk
from sip.outflows.rmsynthesis.rmclean import rmclean_prep, dormclean
from sip.eventhorizon.plot import uvcov, uvdist, pltrmsf

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


START = t.time()

# Manually specify the data directories.
INPUTS_DIR = '/Users/jamie/Data/SIP-Data/inputs'
RESULTS_DIR = '/Users/jamie/Data/SIP-Data/outputs'
os.makedirs(RESULTS_DIR, exist_ok=True)

MAKEPLOTS = False


# Automatically adjust the appropriate parameters depending upon the instrument
INST = 'LOFAR'
POLDEF = initinst(INST)


for channel in range(0, 40):  # loop over chans, only one subband for now
    # Load MSSS snapshots into memory for a single channel:
    vis = load('%s/H033+41/set03/135993/BAND0/L135737_SAP001_BAND0.MS/'
               % (INPUTS_DIR), channel, POLDEF)
    othervis = load('%s/H033+41/set03/136000/BAND0/L135739_SAP001_BAND0.MS/'
                    % (INPUTS_DIR), channel, POLDEF)

    # Combine MSSS snapshots:
    vis = append_visibility(vis, othervis)

    # Apply a filter to the uv-data:
    uvcutoff = 450.0  # approx. MAPS-Mid resolution (for now)
    vis = uvcut(vis, uvcutoff)
    npixeladvice, celladvice = uvadvice(vis, uvcutoff)

    # Make some basic plots:
    if MAKEPLOTS:
        uvcov(vis)
        uvdist(vis)

    # Calibrate the Ionospheric Faraday Rotation Correction:
    # ------------------------------------------------------
    # ra, dec = get_phase_centre(ms_name)
    # time = get_observing_date(ms_name)
    # corrections = download_tec_data(ra, dec, time, "code")
    # log = apply_correction(corrections)

    # Image I, Q, U, V, per channel
    # (no beam model, clean, blanking beyond pb, is currently implemented):
    vis = converttostokes(vis, POLDEF)  # convert from XX/XY/YX/YY to I/Q/U/V.
    wstack(vis, npixeladvice, celladvice, channel, RESULTS_DIR)
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
IMAGECUBE, FREQUENCY, WEIGHTS = loadimdata(RESULTS_DIR)

# Calculate the appropriate parameters for these data:
LAMBDASQ, LAMBDA0, RMSF_EST, SCALE_EST, MAXRM_EST, CELLSIZE, NPIXELS, PHI = \
                                            rmsynth_advice(FREQUENCY, WEIGHTS)

# Form complex (Q+iU) from the polarisation data:
COMPLEX_P = 1j*IMAGECUBE.data[:, 2, :, :]
COMPLEX_P += IMAGECUBE.data[:, 1, :, :]

# Perform the RM Synthesis:
RMSYNTH, RMSF, RA_LEN, DEC_LEN = dormsynth(WEIGHTS, PHI, COMPLEX_P, LAMBDASQ, LAMBDA0)

# Plot the RMSF:
if MAKEPLOTS:
    pltrmsf(PHI, RMSF)

# Save RM Synthesis parameters to disk:
np.savetxt('%s/phi.txt' % (RESULTS_DIR), PHI)
np.savetxt('%s/rmsf.txt' % (RESULTS_DIR), RMSF)
np.savetxt('%s/weights.txt' % (RESULTS_DIR), WEIGHTS)

# Save the dirty cubes to disk:
rmcube_savetodisk(RMSYNTH, CELLSIZE, MAXRM_EST, 'abs', RESULTS_DIR, 'dirty')
rmcube_savetodisk(RMSYNTH, CELLSIZE, MAXRM_EST, 'real', RESULTS_DIR, 'dirty')
rmcube_savetodisk(RMSYNTH, CELLSIZE, MAXRM_EST, 'imag', RESULTS_DIR, 'dirty')

# Calculate the RM-clean threshold:
RMSF_DOUBLE, CLEAN_THRESHOLD = rmclean_prep(RMSYNTH, MAXRM_EST, NPIXELS, WEIGHTS, LAMBDASQ, LAMBDA0)

# Deconvolve the RM-cube:
RMCLEAN = dormclean(RMSYNTH, PHI, RMSF, RMSF_DOUBLE, RMSF_EST, CLEAN_THRESHOLD, RA_LEN, DEC_LEN, CELLSIZE)

# Save the clean cubes to disk:
rmcube_savetodisk(RMCLEAN, CELLSIZE, MAXRM_EST, 'abs', RESULTS_DIR, 'clean')
rmcube_savetodisk(RMCLEAN, CELLSIZE, MAXRM_EST, 'real', RESULTS_DIR, 'clean')
rmcube_savetodisk(RMCLEAN, CELLSIZE, MAXRM_EST, 'imag', RESULTS_DIR, 'clean')


# Source Finding
# ------------------------------------------------------
# Calculate Moments?
# Apply python source-finding implementation?
# Apply SB neutral network algorithm?
# Output source catalogue: ra, dec, RM, PI, etc.

END = t.time()
print(END-START)
