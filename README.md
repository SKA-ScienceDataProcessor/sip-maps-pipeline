SDP Integration Prototype (SIP) Pipeline
========================================

This is a project to create a pipeline for SIP to process spectropolarimetric LOFAR data.

Notes
-----

This initial code is in continuous development, and currently provides limited functionality. 

Primary use at this time includes full-Stokes imaging and Faraday Rotation Measure (RM) Synthesis of data from the Multifrequency Snapshot Sky Survey (MSSS) All-Sky Polarisation Survey (MAPS) with LOFAR.

Dask is only used for imaging at this stage. Implementation throughout the code is upcoming.

Additional features that are intended to be implemented are commented at the appropriate places in the code.

Your PYTHONPATH should be set to include both ARL and Casacore (which is an ARL requirement), i.e. PYTHONPATH=/path/algorithm-reference-library:/path/casacore

Dependencies
------------

ARL

Dask
