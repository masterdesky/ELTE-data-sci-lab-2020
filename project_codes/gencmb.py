import os
import sys
import numpy as np
import healpy as hp
from functools import partial
import astropy.io.fits as fits

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Inuput and output folders
data = './data/'
out = './output/'

# Various size parameters for plotting
axistitlesize = 20
axisticksize = 17
axislabelsize = 26
axislegendsize = 23
axistextsize = 20
axiscbarfontsize = 15

##########################################
##    I. Load arbitrary power spectrum
##########################################
def load_spectrum(fname, lmax=None):
    """
    Loads and arbitrary angular power spectrum from a file.

    Datasets generated with the LAMBDA tool contains only the :math:`D_{l}`
    values, alongside the array of the :math:`l` multipoles and the
    corresponding errors.

    Parameters
    ----------
    fname : str
       Name of input file containing the multipoles and transformed angular
       power spectrum values by columns. The considered case is that the first
       column contains the :math:`l` values, while the second contains the
       :math:`D_{l}` transformed spectrum. The :math:`l` should start with
       the order :math:`l = 2`.
    lmax : int
        Bandlimit of the angular power spectrum. The spectrum and coefficients will
        be read in up to the spherical harmonic order :math:`l_{\mathrm{max}}`.

    Returns
    -------
    ell : numpy.array
        List of multipoles for which the angular power spectrum values
        were evaluated. Contains integers from 2 to :math:`l_{\mathrm{max}}`,
        where :math:`l_{\mathrm{max}}` is included.
    DlTT : numpy.array
        Transformed angular power spectrum bins (:math:`D_{l}`) for every
        multipole value in `ell`. The transformation is
        .. math::
                    D_{l} = \frac{l (l + 1)}{2 \pi} C_{l}.
    """
    assert os.path.exists(data + fname), \
        "There is no file named `{0}` exists in the directory `{1}`".format(fname, data.strip('./'))
    
    d = np.genfromtxt(data + fname)
    ell = d[:lmax-1, 0]
    DlTT = d[:lmax-1, 1]
    ClTT = DlTT * 2 * np.pi / (ell * (ell + 1))
    
    return ell, ClTT, DlTT


def gen_maps(ClTT, N_SIDE=2048, lmax=None,
             pol=False, pixwin=False, fwhm=5.8e-3, sigma=8.7e-6):
    """
    Generate randomized HEALPix arrays from an input angular power spectrum.
    
    Parameters
    ----------
    ClTT : array or tuple of arrays
    
    N_SIDE : int
        The number of pixels per side in a HEALPix projection. This is always
        determined by the input dataset. In the case of the files of the Planck
        telescope, this value is always N_SIDE = 2048.
    lmax : int
    
    alm : bool
        If `True`, returns the :math:`a_{lm}` coefficients corresponding to
        the correct :math:`C_{l}` values.
    
    pol : bool
    
    pixwin : bool
    
    fwhm : float
        
        
        More info on the Planck beam FWHM can be found at
        "https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Beams_LFI"
    
    sigma : float
        
        
        More info on the Planck beam sigma can be found at
        "https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Beams_LFI"
    
    Returns
    -------
    maps : array or tuple of arrays
        The output map (possibly list of maps if polarized input).
        Or, if `alm` is `True`, a tuple of `(map, alm)`, where
        `alm` is possibly a list of :math:`a_{lm}` arrays if polarized input.

    """
    return hp.synfast(ClTT, nside=N_SIDE, lmax=lmax, pol=pol, pixwin=pixwin, fwhm=fwhm, sigma=sigma)