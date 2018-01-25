#!/usr/bin/env python

"""imaging.py: The script for imaging."""

import sys

import numpy as np

from arl.data.polarisation import PolarisationFrame
from arl.imaging import create_image_from_visibility, advise_wide_field
from arl.graphs.graphs import create_invert_graph
from arl.image.operations import export_image_to_fits
from arl.imaging.weighting import weight_visibility

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def wstack(vis, npixel_advice, cell_advice, channel, results_dir):
    """Do w-stacked imaging of visibility data.
        
    Args:
    vis (obj): ARL visibility data.
    npixel_advice (float): number of pixels in output image.
    cell_advice (float): cellsize in output image.
    channel (int): channel number to be imaged (affects output filename).
    results_dir (str): directory to save results.
    
    Returns:
    dirty: dirty image.
    psf: image of psf.
    """
    try:
        vis_slices = len(np.unique(vis.time))
        print("There are %d timeslices" % vis_slices)
        # Obtain advice on w-proj parameters:
        advice = advise_wide_field(vis)
        # Create a model image:
        model = create_image_from_visibility(vis, cellsize=cell_advice,
                                             npixel=npixel_advice,
                                             polarisation_frame=PolarisationFrame('stokesIQUV'))
        # Weight the visibilities:
        vis, _, _ = weight_visibility(vis, model)
        # Create a dirty image:
        dirty, sumwt = create_invert_graph([vis], model, kernel='wstack',
                                           wstep=advice['w_sampling_primary_beam'],
                                           oversampling=2).compute()
        # Create the psf:
        psf, sumwt = create_invert_graph([vis], model, dopsf=True, kernel='wstack',
                                         wstep=advice['w_sampling_primary_beam'],
                                         oversampling=2).compute()
        # Save to disk:
        export_image_to_fits(dirty, '%s/imaging_dirty_WStack-%s.fits'
                             % (results_dir, channel))
        export_image_to_fits(psf, '%s/imaging_psf_WStack-%s.fits'
                             % (results_dir, channel))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return dirty, psf


def wproject(vis, npixel_advice, cell_advice, channel, results_dir):
    """Do w-projected imaging of visibility data.
    
    Args:
    vis (obj): ARL visibility data.
    npixel_advice (float): number of pixels in output image.
    cell_advice (float): cellsize in output image.
    channel (int): channel number to be imaged (affects output filename).
    results_dir (str): directory to save results.
    
    Returns:
    dirty: dirty image.
    psf: image of psf.
    """
    try:
        vis_slices = len(np.unique(vis.time))
        print("There are %d timeslices" % vis_slices)
        # Obtain advice on w-proj parameters:
        advice = advise_wide_field(vis)
        # Create a model image:
        model = create_image_from_visibility(vis, cellsize=cell_advice,
                                             npixel=npixel_advice,
                                             polarisation_frame=PolarisationFrame('stokesIQUV'))
        # Weight the visibilities:
        vis, _, _ = weight_visibility(vis, model)
        # Create a dirty image:
        dirty, sumwt = create_invert_graph([vis], model, kernel='wprojection',
                                           wstep=advice['w_sampling_primary_beam'],
                                           oversampling=2).compute()
        # Create the psf:
        psf, sumwt = create_invert_graph([vis], model, dopsf=True, kernel='wprojection',
                                         wstep=advice['w_sampling_primary_beam'],
                                         oversampling=2).compute()
        # Save to disk:
        export_image_to_fits(dirty, '%s/imaging_dirty_WProj-%s.fits'
                             % (results_dir, channel))
        export_image_to_fits(psf, '%s/imaging_psf_WProj-%s.fits'
                             % (results_dir, channel))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return dirty, psf
