import os
import numpy as np
import healpy as hp
from functools import partial

import astropy.io.fits as fits

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Inuput and output folders
data = '../data/'
out = '../out/'

# Various size parameters for plotting
axistitlesize = 20
axisticksize = 17
axislabelsize = 26
axislegendsize = 23
axistextsize = 20
axiscbarfontsize = 15

##########################################
##    I. Load CMB maps
##########################################

def load_HPX(file, field=1):
    """
    Loads a HEALPix array from a given field of an input FITS file.
    
    Parameters
    ----------
    file : 
        The input `.fits` file.
    field : int
        
    
    Returns
    -------
    hpx : numpy.array of shape (12 * N_SIDE**2, )
        Raw HEALPix dataset, loaded by the `healpy` library from an
        input file (from the `.fits` table in case of Planck's datasets).
        Stored as Kelvin values in case of Planck.
    hpx_muK : numpy.array of length (12 * N_SIDE^2, )
        The same, raw HEALPix dataset with values converted to micro Kelvin.
    header : astropy.io.fits.header.Header
        The header file of the input `.fits` table.
    """
    # Load the given field
    hpx, _ = hp.read_map(file, field=field,
                         dtype=np.float64, h=True, verbose=False, memmap=True)
    
    with fits.open(file, memmap=True) as hdul:
        header = hdul[1].header
    
    # Convert K to muK
    hpx_muK = hpx*1e06
    
    return hpx, hpx_muK, header


##########################################
##    II. Visualize of CMB maps
##########################################

def get_projection(hpx, proj='moll', N_SIDE=2048):
    """
    Projects the input HEALPix dataset on an arbitrary geographical projection,
    which is implemented in the `healpy` package.
    
    Parameters
    ----------
    hpx : numpy.ndarray in the size of (12 * N_SIDE**2, )
        Raw HEALPix dataset, loaded by the `healpy` library from an
        input file (from the `.fits` table in case of Planck's datasets).
        Stored as Kelvin values in case of Planck.
    proj : str
        The projection used to create a 2D matrix from the input HEALPix data. Can be
        either of the following:
            - 'moll' : Mollweide projection
            - 'cart' : Cartesian (Equirectangular) projection
            - 'orth' : Orhographic projection
    N_SIDE: int
        The number of pixels per side in a HEALPix projection. This is always
        determined by the input dataset. In the case of the files of the Planck
        telescope, this value is always N_SIDE = 2048.
    
    Returns
    -------
    hpx_proj : numpy.ndarray in the size of (N, M)
        The projected matrix generate from the input HEALPix dataset. The projected map is
        encompassed inside the borders of the matrix. Values outside the projection were
        assigned with the value `-np.inf`.
    """
    _POSSIBLE_PROJ = ['moll', 'cart', 'orth']
    assert proj in _POSSIBLE_PROJ, (f'Available projections are : \
                                            {_POSSIBLE_PROJ}')

    if proj == 'moll':
        p = hp.projector.MollweideProj
    elif proj == 'cart':
        p = hp.projector.CartesianProj
    elif proj == 'orth':
        p = hp.projector.OrthographicProj
    
    p = p(xsize=N_SIDE, coord='G')
    hpx_proj = p.projmap(hpx, vec2pix_func=partial(hp.vec2pix, N_SIDE))
    
    return hpx_proj


def planck_cmap():
    """
    Generates the Planck CMB colormap from an input file, which stores
    the color values for the complete gradient. The colormap values was
    obtained from the following link:
    - https://github.com/zonca/paperplots/raw/master/data/Planck_Parchment_RGB.txt
    """
    cpath = os.path.join(data, 'Planck_Parchment_RGB.txt')
    colombi1_cmap = ListedColormap(np.loadtxt(cpath)/255.)
    colombi1_cmap.set_bad('black')   # color of missing pixels
    colombi1_cmap.set_under('white') # color of background
    
    return colombi1_cmap


def plot_cmb(proj, cmap=None, c_min=None, c_max=None,
             cbar=False, save=False, save_filename='default_name_map'):
    """
    Plots an input image generated by a `healpy.projector` routine and
    scales the values if needed. The routine uses the classic Planck CMB
    colormap by default to shade pixels on the image.
    
    Parameters
    ----------
    proj : numpy.ndarray of size (N,M)
        The input image is already just a projection of the original
        HEALPix dataset, created by a `healpy.projector` routine. The
        CMB projection is encompassed inside this rectangular matrix.
    cmap : str or a matplotlib.colors.* colormap
        The colormap used to shade the pixels. By default the routine
        uses the classic Planck CMB colormap.
    c_min : float
        The lower limit for plotted values. All values of the input
        matrix below this limit will be scaled up to this value.
    c_max : float
        The upper limit for plotted values. All values of the input
        matrix above this limit will be scaled down to this value.
    cbar : bool
        If `True`, then the routine will plot a colorbar on the right
        side of the image.
    save : bool
        If `True`, then saves the image into the output folder, under
        the name `save_filename`.
    save_filename : str
        The name of the saved image file. Only has effect if `save` is
        set to `True`.
    """
    fig, ax = plt.subplots(figsize=(2*16, 16), dpi=120, facecolor='black',
                           subplot_kw={'facecolor' : 'black'})
    ax.axis('off')
    
    # Convert 'proj' values to display properly
    proj_n = proj.copy()
    proj_n[np.abs(proj_n) == np.inf] = np.nan
    if c_min is not None:
        proj_n[proj_n < c_min] = c_min
    if c_max is not None:
        proj_n[proj_n > c_max] = c_max

    # Set colormap for the image
    colormap = planck_cmap() if cmap is None else cmap

    im = ax.imshow(proj_n, 
                   cmap=colormap, vmin=c_min, vmax=c_max,
                   interpolation='bilinear', origin='lower')
    
    ax.set_xlabel('Angle $[^\circ]$', color='white',
                  fontsize=axislabelsize, fontweight='bold')
    ax.set_ylabel('Angle $[^\circ]$', color='white',
                  fontsize=axislabelsize, fontweight='bold')
    ax.tick_params(axis='both', which='major', colors='white',
                   labelsize=axisticksize)
    
    # Create an axis on the right side of `axes`. The width of `cax` will be 2%
    # of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.1)
        cbar = plt.colorbar(mappable=im, cax=cax)
        cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='white')
        cbar.set_label('Temperature [$\mu$K]', color='white',
                       fontsize=axiscbarfontsize+8, rotation=90, labelpad=12)
    if save:
        os.makedirs(out, exist_ok=True)
        plt.savefig(out + save_filename,
                    format='png', dpi=120,
                    facecolor='black', edgecolor='black',
                    bbox_inches='tight')
    plt.show()


##########################################
##    III. CMB angular power spectrum
##########################################

def cmb_spectrum(hpx, lmax=2500, alm=True):
    """
    Calculates the :math:`a_{lm}` and :math:`C_{l}` parameters using the
    `anafast` subroutine from the Fortran90 standard, up to a given
    :math:`l_{\mathrm{max}}` bandlimit.
    
    Parameters
    ----------
    hpx : numpy.ndarray in the size of (12 * N_SIDE**2, )
        The input raw HEALPix dataset, loaded by the `healpy` library from an
        input file (from the `.fits` table in case of Planck's datasets).
    lmax : int
        Bandlimit of the angular power spectrum. The spectrum and coefficients will
        be calculated up to the spherical harmonic order :math:`l_{\mathrm{max}}`.
    
    Returns
    -------
    ell : numpy.array
        List of multipoles for which the angular power spectrum values
        were evaluated. Contains integers from 2 to :math:`l_{\mathrm{max}}`,
        where :math:`l_{\mathrm{max}}` is included.
    Cl : numpy.array
        Angular power spectrum bins (:math:`C_{l}`) for every multipole value
        in `ell`.
    alm : 
        Spherical harmonics coefficients in the expansion of the
        :math:`\Delta T (\theta, \varphi)` function.
    """
    ell = np.arange(lmax + 1)
    if alm:
        Cl, alm = hp.anafast(hpx, lmax=lmax, alm=True)
        Dl = ell * (ell + 1) / (2 * np.pi) * Cl
        return ell[2:], Cl[2:], Dl[2:], alm[2:]
    else:
        Cl = hp.anafast(hpx, lmax=lmax, alm=False)
        Dl = ell * (ell + 1) / (2 * np.pi) * Cl
        return ell[2:], Cl[2:], Dl[2:]


def plot_spectrum(ell, Dl, DlTT,
                  save=False, save_filename='default_name_spectrum'):
    """
    Plots the angular power spectrum of the CMB.
    
    Paramters
    ---------
    ell : numpy.array
        List of multipoles for which the angular power spectrum values
        were evaluated. Contains integers from 2 to :math:`l_{\mathrm{max}}`,
        where :math:`l_{\mathrm{max}}` is included.
        
    Dl : numpy.array
        
    DlTT : numpy.array
        
    """
    fig, axes = plt.subplots(figsize=(15, 9),
                         facecolor='black', subplot_kw={'facecolor' : 'black'})
    axes.plot(ell, Dl, label='Planck 2018 DR3',
              color=cm.magma(0.92), lw=1)
    axes.plot(ell, DlTT, label='$\Lambda$-CDM simulation',
              color=cm.magma(0.65), lw=3, ls='--')

    axes.set_xlabel('$\ell$', fontsize=axislabelsize, fontweight='bold', color='white')
    axes.set_ylabel('$D_{\ell}$ [$\,\mu$K$^2\,$]', fontsize=axislabelsize, fontweight='bold', color='white')
    axes.tick_params(axis='both', which='major', labelsize=axisticksize, colors='white')

    axes.legend(loc='best', fontsize=axislegendsize,
                labelcolor='white', facecolor='black', framealpha=0.8, shadow=True)

    if save:
        if not os.path.exists(out):
            os.makedirs(out)
        plt.savefig(out + save_filename,
                    format='png', dpi=200,
                    facecolor='black', edgecolor='black',
                    bbox_inches='tight')

    plt.show()