#!/usr/bin/env python

"""filter.py: The script for filtering a measurement set."""

import numpy as np

from arl.visibility.base import create_visibility_from_rows
from arl.imaging import advise_wide_field

from astropy.constants import c

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def uvcut(vis, uvmax):
    """Cut the visibility data at uv-distances beyond uvmax.
        
    Args:
    vis (obj): ARL visibility data.
    uvmax (float): maximum uv-coordinate.
    
    Returns:
    vis: New visibility data.
    """
    # Cut off data beyond the maximum uv-distance:
    uvdist = np.sqrt(vis.data['uvw'][:, 0]**2+vis.data['uvw'][:, 1]**2)
    vis = create_visibility_from_rows(vis, uvdist < uvmax)
    return vis


def uvadvice(vis, uvcutoff):
    """Advise on the imaging parameters for fully-sampled images.
        
    Args:
    vis (obj): ARL visibility data.
    uvcutoff (float): maximum intended uv-coordinate.
    
    Returns:
    npixeladvice: advised number of pixels.
    celladvice: advised cellsize.
    """
    # Find the maximum uv-distance:
    uvdist = np.sqrt(vis.data['uvw'][:, 0]**2+vis.data['uvw'][:, 1]**2)
    uvmax = np.max(uvdist)
    print("Maximum uv-distance:", uvmax)
    # Calculate the angular resolution:
    print("Observing Frequency, MHz:", vis.frequency[0]/1e6)
    lambda_meas = c.value/vis.frequency[0]
    print("")
    print("Angular resolution, FWHM:", lambda_meas/(uvcutoff*lambda_meas))
    angres_arcmin = 60.0*(180.0/np.pi)*(1.0/uvmax)
    angres_arcsec = 60.0*60.0*(180.0/np.pi)*(1.0/uvmax)
    print("arcmin", angres_arcmin)
    print("arcsec", angres_arcsec)
    print("")
    # Calculate the cellsize:
    celladvice = (angres_arcmin/(60.0*5.0))*(np.pi/180.0)
    # Determine the npixel size required:
    pixeloptions = np.array([512, 1024, 2048, 4096, 8192])
    pbfov = pixeloptions*celladvice*(180.0/np.pi)
    advice = advise_wide_field(vis)
    npixeladvice = pixeloptions[np.argmax(pbfov >
                                advice['primary_beam_fov']*(180.0/np.pi)*2.0)]
    print("Recommended npixels/cellsize:", npixeladvice, "/", celladvice)
    return npixeladvice, celladvice
