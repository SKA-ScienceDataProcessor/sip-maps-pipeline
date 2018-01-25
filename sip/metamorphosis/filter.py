#!/usr/bin/env python

"""filter.py: The script for filtering a measurement set."""

import numpy as np

from arl.visibility.base import create_visibility_from_rows
from arl.imaging import advise_wide_field

from astropy.constants import c

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def uv_cut(vis, uv_max):
    """Cut the visibility data at uv-distances beyond uvmax.
        
    Args:
    vis (obj): ARL visibility data.
    uv_max (float): maximum uv-coordinate.
    
    Returns:
    vis: New visibility data.
    """
    # Cut off data beyond the maximum uv-distance:
    uv_dist = np.sqrt(vis.data['uvw'][:, 0]**2+vis.data['uvw'][:, 1]**2)
    vis = create_visibility_from_rows(vis, uv_dist < uv_max)
    return vis


def uv_advice(vis, uv_cutoff, pixels_per_beam):
    """Advise on the imaging parameters for fully-sampled images.
        
    Args:
    vis (obj): ARL visibility data.
    uv_cutoff (float): maximum intended uv-coordinate.
    pixels_per_beam (float): number of pixel samples across the beam.
    
    Returns:
    npixel_advice: advised number of pixels.
    cell_advice: advised cellsize.
    """
    # Find the maximum uv-distance:
    uv_dist = np.sqrt(vis.data['uvw'][:, 0]**2+vis.data['uvw'][:, 1]**2)
    uv_max = np.max(uv_dist)
    print("Maximum uv-distance:", uv_max)
    # Calculate the angular resolution:
    print("Observing Frequency, MHz:", vis.frequency[0]/1e6)
    lambda_meas = c.value/vis.frequency[0]
    print("")
    print("Angular resolution, FWHM:", lambda_meas/(uv_cutoff*lambda_meas))
    angres_arcmin = 60.0*(180.0/np.pi)*(1.0/uv_max)
    angres_arcsec = 60.0*60.0*(180.0/np.pi)*(1.0/uv_max)
    print("arcmin", angres_arcmin)
    print("arcsec", angres_arcsec)
    print("")
    # Calculate the cellsize:
    cell_advice = (angres_arcmin/(60.0*pixels_per_beam))*(np.pi/180.0)
    # Determine the npixel size required:
    pixel_options = np.array([512, 1024, 2048, 4096, 8192])
    pb_fov = pixel_options*cell_advice*(180.0/np.pi)
    advice = advise_wide_field(vis)
    npixel_advice = pixel_options[np.argmax(pb_fov >
                                            advice['primary_beam_fov']*(180.0/np.pi)*2.0)]
    print("Recommended npixels/cellsize:", npixel_advice, "/", cell_advice)
    return npixel_advice, cell_advice
