#!/usr/bin/env python

""" pipe.py: The main script to trigger the MAPS pipeline for SIP.
"""

import os

import time as t

import numpy as np

from arl.visibility.operations import append_visibility
from arl.image.operations import export_image_to_fits
from arl.image.deconvolution import restore_cube

from sip.telescopetools.initinst import init_inst
from sip.accretion.ms import load
from sip.metamorphosis.filter import uv_cut, uv_advice
from sip.metamorphosis.convert import convert_to_stokes
from sip.metamorphosis.beam import beam_me_up, apply_station_beams
from sip.metamorphosis.iono import get_ion_rotation_measures, correct_ion_faraday
from sip.outflows.images.imaging import wstack
from sip.outflows.images.deconvolution import deconvolve_cube_complex
from sip.outflows.rmsynthesis.rmsynth import load_im_data, rmsynth_advice, do_rmsynth, rmcube_save_to_disk
from sip.outflows.rmsynthesis.rmclean import rmclean_prep, do_rmclean
from sip.outflows.moments.moments import load_moments_data, calc_moments, moments_save_to_disk
from sip.eventhorizon.plot import uv_cov, uv_dist, plt_rmsf

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


START = t.time()

# Setup Variables
# ------------------------------------------------------
# Manually specify the data directories:
INPUTS_DIR = '/Users/jamie/Data/SIP-Data/inputs'
RESULTS_DIR = '/Users/jamie/Data/SIP-Data/outputs'
MS1 = 'H033+41/set03/135993/BAND0/L135737_SAP001_BAND0.MS/'
MS2 = 'H033+41/set03/136000/BAND0/L135739_SAP001_BAND0.MS/'
os.makedirs(RESULTS_DIR, exist_ok=True)

MAKE_PLOTS = False  # Output diagnostic plots?
APPLY_BEAM = True  # Apply the primary beam?
APPLY_IONO = True  # Correct for ionospheric Faraday rotation?
SAVE_RESIDUAL_IMAGES = False  # Save the residual and clean component images?
UV_CUTOFF = 450.0  # Cut-off for the uv-data. Set to 450 = approx. MAPS-Mid resolution (for now).
FORCE_RESOLUTION = 8.0  # Force the angular resolution to be consistent across the band. In arcmin FWHM.
PIXELS_PER_BEAM = 5.0  # The number of pixels/sampling across the observing beam.
# Automatically adjust some appropriate parameters depending upon the instrument:
INST = 'LOFAR'  # Instrument name (for future use).
POLDEF = init_inst(INST)


for channel in range(0, 40):  # loop over channels, only one subband for now
    # Import Data
    # ------------------------------------------------------
    # Load MSSS snapshots into memory for a single channel:
    vis1 = load('%s/%s' % (INPUTS_DIR, MS1), channel, POLDEF)
    vis2 = load('%s/%s' % (INPUTS_DIR, MS2), channel, POLDEF)

    # Ionospheric Faraday Rotation Correction
    # ------------------------------------------------------
    # Download the TEC data and calculate the ionospheric Faraday rotation:
    if APPLY_IONO:
        if channel == 0:
            ionRM1, times1, time_indices1 = get_ion_rotation_measures(vis1, INPUTS_DIR, MS1)
            ionRM2, times2, time_indices2 = get_ion_rotation_measures(vis2, INPUTS_DIR, MS2)
            # Save the median Faraday rotation to disk, as an instrumental estimate:
            np.savetxt('%s/ionFR.txt' % (RESULTS_DIR), \
                       [np.median(np.concatenate((ionRM1, ionRM2), axis=0))], fmt="%s")
        # Correct the data for the ionospheric rotation measure:
        vis1 = correct_ion_faraday(vis1, ionRM1, times1, time_indices1)
        vis2 = correct_ion_faraday(vis2, ionRM2, times2, time_indices2)

    # Prepare Measurement Set
    # ------------------------------------------------------
    # Combine MSSS snapshots:
    vis = append_visibility(vis1, vis2)

    # Apply a uv-distance cut to the data:
    vis = uv_cut(vis, UV_CUTOFF)
    npixel_advice, cell_advice = uv_advice(vis, UV_CUTOFF, PIXELS_PER_BEAM)

    # Make some basic plots:
    if MAKE_PLOTS:
        uv_cov(vis)
        uv_dist(vis)

    # Primary Beam Correction
    # ------------------------------------------------------
    # Apply the primary beam of the instrument:
    if APPLY_BEAM:
        beams = beam_me_up(INPUTS_DIR, MS1) # use only one MS (each MS should have identical station positions, phase centre, etc.)
        vis = apply_station_beams(vis, beams, channel)

    # Imaging and Deconvolution
    # ------------------------------------------------------
    # Convert from XX/XY/YX/YY to I/Q/U/V:
    vis = convert_to_stokes(vis, POLDEF)

    # Image I, Q, U, V, per channel:
    dirty, psf = wstack(vis, npixel_advice, cell_advice, channel, RESULTS_DIR)

    # Deconvolve (using complex Hogbom clean):
    comp, residual = deconvolve_cube_complex(dirty, psf, niter=10000, threshold=0.001, \
                                             fracthresh=0.001, window_shape='', gain=0.1, \
                                             algorithm='hogbom-complex')

    # Convert resolution (FWHM in arcmin) to a psfwidth (standard deviation in pixels):
    clean_res = (((FORCE_RESOLUTION/2.35482004503)/60.0)*np.pi/180.0)/cell_advice

    # Create the restored image:
    restored = restore_cube(comp, psf, residual, psfwidth=clean_res)

    # Save the images to disk:
    if SAVE_RESIDUAL_IMAGES:
        export_image_to_fits(comp, '%s/imaging_comp_WStack-%s.fits' % (RESULTS_DIR, channel))
        export_image_to_fits(residual, '%s/imaging_residual_WStack-%s.fits' % (RESULTS_DIR, channel))
    export_image_to_fits(restored, '%s/imaging_clean_WStack-%s.fits' % (RESULTS_DIR, channel))


# Rotation Measure (RM) Synthesis
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


# Faraday Moments
# ------------------------------------------------------
# Load in the FITS images:
IMAGECUBE, WEIGHTS = load_moments_data(RESULTS_DIR)

# Calculate the Faraday Moments:
MEAN_Q, MEAN_U, MEAN_P, SIGMA_Q, SIGMA_U, SIGMA_P = calc_moments(IMAGECUBE, WEIGHTS)

# Save the moment images to disk:
moments_save_to_disk(MEAN_Q, 'q', RESULTS_DIR, 'mean')
moments_save_to_disk(MEAN_U, 'u', RESULTS_DIR, 'mean')
moments_save_to_disk(MEAN_P, 'p', RESULTS_DIR, 'mean')
moments_save_to_disk(SIGMA_Q, 'q', RESULTS_DIR, 'sigma')
moments_save_to_disk(SIGMA_U, 'u', RESULTS_DIR, 'sigma')
moments_save_to_disk(SIGMA_P, 'p', RESULTS_DIR, 'sigma')


# Source Finding (Not a formal part of the optimised pipeline)
# ------------------------------------------------------------
# Calculate the background and rms:
os.system('bane --grid=' + str(int(4.0*PIXELS_PER_BEAM)) + ' ' + str(int(4.0*PIXELS_PER_BEAM)) \
          + ' %s/%s_%s.fits' % (RESULTS_DIR, 'mean', 'q'))
os.system('bane --grid=' + str(int(4.0*PIXELS_PER_BEAM)) + ' ' + str(int(4.0*PIXELS_PER_BEAM)) \
          + ' %s/%s_%s.fits' % (RESULTS_DIR, 'mean', 'u'))
os.system('bane --grid=' + str(int(4.0*PIXELS_PER_BEAM)) + ' ' + str(int(4.0*PIXELS_PER_BEAM)) \
          + ' %s/%s_%s.fits' % (RESULTS_DIR, 'mean', 'p'))
os.system('bane --grid=' + str(int(4.0*PIXELS_PER_BEAM)) + ' ' + str(int(4.0*PIXELS_PER_BEAM)) \
          + ' %s/%s_%s.fits' % (RESULTS_DIR, 'sigma', 'q'))
os.system('bane --grid=' + str(int(4.0*PIXELS_PER_BEAM)) + ' ' + str(int(4.0*PIXELS_PER_BEAM)) \
          + ' %s/%s_%s.fits' % (RESULTS_DIR, 'sigma', 'u'))
os.system('bane --grid=' + str(int(4.0*PIXELS_PER_BEAM)) + ' ' + str(int(4.0*PIXELS_PER_BEAM)) \
          + ' %s/%s_%s.fits' % (RESULTS_DIR, 'sigma', 'p'))

# Find the sources using the Faraday Moment images:
print('aegean --negative --autoload --seedclip=3.0 --beam=' + str(FORCE_RESOLUTION/60.0) + ' ' \
      + str(FORCE_RESOLUTION/60.0) + ' 0.0 ' + '%(1)s/%(2)s_%(3)s.fits --out=%(1)s/%(2)s_%(3)s.out' \
      % {'1': RESULTS_DIR, '2': 'mean', '3': 'q'})
os.system('aegean --negative --autoload --seedclip=3.0 --beam=' + str(FORCE_RESOLUTION/60.0) + ' ' \
          + str(FORCE_RESOLUTION/60.0) + ' 0.0 ' + '%(1)s/%(2)s_%(3)s.fits --out=%(1)s/%(2)s_%(3)s.out' \
          % {'1': RESULTS_DIR, '2': 'mean', '3': 'q'})
os.system('aegean --negative --autoload --seedclip=3.0 --beam=' + str(FORCE_RESOLUTION/60.0) + ' ' \
          + str(FORCE_RESOLUTION/60.0) + ' 0.0 ' + '%(1)s/%(2)s_%(3)s.fits --out=%(1)s/%(2)s_%(3)s.out' \
          % {'1': RESULTS_DIR, '2': 'mean', '3': 'u'})
os.system('aegean --autoload --seedclip=5.0 --beam=' + str(FORCE_RESOLUTION/60.0) + ' ' \
          + str(FORCE_RESOLUTION/60.0) + ' 0.0 ' + '%(1)s/%(2)s_%(3)s.fits --out=%(1)s/%(2)s_%(3)s.out' \
          % {'1': RESULTS_DIR, '2': 'mean', '3': 'p'})
os.system('aegean --autoload --seedclip=5.0 --beam=' + str(FORCE_RESOLUTION/60.0) + ' ' \
          + str(FORCE_RESOLUTION/60.0) + ' 0.0 ' + '%(1)s/%(2)s_%(3)s.fits --out=%(1)s/%(2)s_%(3)s.out' \
          % {'1': RESULTS_DIR, '2': 'sigma', '3': 'q'})
os.system('aegean --autoload --seedclip=5.0 --beam=' + str(FORCE_RESOLUTION/60.0) + ' ' \
          + str(FORCE_RESOLUTION/60.0) + ' 0.0 ' + '%(1)s/%(2)s_%(3)s.fits --out=%(1)s/%(2)s_%(3)s.out' \
          % {'1': RESULTS_DIR, '2': 'sigma', '3': 'u'})
os.system('aegean --autoload --seedclip=5.0 --beam=' + str(FORCE_RESOLUTION/60.0) + ' ' \
          + str(FORCE_RESOLUTION/60.0) + ' 0.0 ' + '%(1)s/%(2)s_%(3)s.fits --out=%(1)s/%(2)s_%(3)s.out' \
          % {'1': RESULTS_DIR, '2': 'sigma', '3': 'p'})

# Final step - concatenate sources and output the final source catalogue: ra, dec, RM, PI, etc.


END = t.time()
print(END-START)
