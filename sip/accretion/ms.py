#!/usr/bin/env python

"""ms.py: A script for loading measurement set data."""

import sys

from arl.visibility.base import create_visibility_from_ms
from arl.data.polarisation import PolarisationFrame

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def load(msfilename='', channumselect=0, poldef='lin'):
    """Load a measurement set and define the polarisation frame.
        
    Args:
    msfilename (str): file name of the measurement set.
    channumselect (int): channel number to load.
    poldef (str): definition of the polarisation frame.
        
    Returns:
    vis: The visibility data.
    """
    try:
        # Load in a single MS:
        vis = create_visibility_from_ms(msfilename, channumselect)[0]
        # Set the polarisation frame:
        if poldef == 'lin':
            vis.polarisation_frame = PolarisationFrame('linear')
        if poldef == 'circ':
            vis.polarisation_frame = PolarisationFrame('circular')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return vis
