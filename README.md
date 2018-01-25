SDP Integration Prototype (SIP) Pipeline
========================================

This is a project to create a pipeline for SIP to process spectropolarimetric LOFAR data.

Notes
-----

This initial serial code provides a fully-Pythonic pipeline for processing spectropolarimetric LOFAR data into resulting data products and an initial source catalogue. 

Primary uses includes full-Stokes imaging and Faraday Rotation Measure (RM) Synthesis of data from the Multifrequency Snapshot Sky Survey (MSSS) All-Sky Polarisation Survey (MAPS) with LOFAR. The MSSS total intensity survey description is available (2015A&A...582A.123H).

Features included within the code are ionospheric Faraday rotation calibration, imaging and deconvolution, RM Synthesis and RM-CLEAN, Faraday Moments, and source-finding. The LOFAR station beams are also currently loaded within a dummy function (although full beam correction is not yet implemented). The included features typically use the most up-to-date available algorithms that are required within the "Cosmic Magnetism" science case, for example:
* Generalised Complex CLEAN for improved polarisation image deconvolution (2016MNRAS.462.3483P)
* Faraday RM Synthesis (2005A&A...441.1217B)
* RM-CLEAN (2009IAUS..259..591H) which identifies peaks using complex cross-correlation (performance benchmarked in 2015AJ....149...60S - see algorithm FS-JF)
* Faraday Moments for source-finding (2018MNRAS.474.3280F) in conjunction with Aegean (2012MNRAS.422.1812H)
* A detailed review of many aspects of the science case for these algorithms is available in Akahori et al. 2017, PASJ (https://arxiv.org/abs/1709.02072).

Dask (delayed) is only used for imaging at this stage. Full parallelisation with Dask (distributed) is in progress on P3-AlaSKA, alongside integration with other SIP services.

Your PYTHONPATH should be set to include ARL, Casacore (which is an ARL requirement), RMextract, and pystationresponse, i.e. PYTHONPATH=/path/algorithm-reference-library:/path/casacore

Dependencies
------------

ARL

Dask

RMextract

pystationresponse (from the LOFAR software stack)

Aegean
