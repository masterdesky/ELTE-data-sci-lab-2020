import os
import sys
import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from constants import *

data = '../data/'
out = '../output/'

axistitlesize = 20
axisticksize = 15
axislabelsize = 23
axislegendsize = 23
axistextsize = 20
axiscbarfontsize = 15

def haversine(X, Y):
    """
    Calculates the Haversine formula for every gridpoint on a given domain.
    
    Parameters
    ----------
    X : numpy.ndarray of shape(N_x, N_y)
        X coordinates of the domain.
    Y : numpy.ndarray of shape(N_x, N_y)
        Y coordinates of the domain.
    
    Returns
    -------
    R : numpy.ndarray of shape(N_x, N_y)
        Distance matrix of the grid of the input domain.
    """
    R = 2 * np.arcsin(np.sqrt(np.sin(X/2)**2 + np.cos(X) * np.sin(Y/2)**2))
    
    return R

def make_coordinates(N_x, N_y,
                     X_width, Y_width,
                     absolute=False):
    """
    Make an "absolute" or "relative", 2D equirectangular coordinate system.
    
    Parameters
    ----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
        
    Returns
    -------
    R : numpy.ndarray of shape (N_x, N_y)
        The distance matrix over the generated domain.
    """
    
    if absolute:
        x = np.linspace(-np.deg2rad(X_width)/2, np.deg2rad(X_width)/2, N_x)
        y = np.linspace(-np.deg2rad(Y_width)/2, np.deg2rad(Y_width)/2, N_y)
    else:
        x_short = True if N_x < N_x else False
        prop = N_y/N_x if x_short else N_x/N_y
        base = np.linspace(-0.5, 0.5, (N_x if x_short else N_y))
        long = np.linspace(-0.5*prop, 0.5*prop, (N_y if x_short else N_x))
        x = base if x_short else long
        y = long if x_short else base
    X, Y = np.meshgrid(x, y)
    # Calculating the distance matrix of a grid on a spherical surface using
    # the Haversine formula
    R = haversine(X, Y)
    
    # The haversine formula above gives us distances on the surface of the sphere,
    # which are dependent of the radius of the sphere in question.
    # To make them independent of this quantity (which is completely incomprehensible
    # in case of the CMB), we need to convert these values to angles. Since we're
    # working with arcminutes everywhere in this project, I'll convert them
    # to arcmins.
    # Sometimes only the proportions needed, but sometimes the arcmins itself.
    if absolute:
        R = np.rad2deg(R) * 60
    
    return R
  ###############################

def make_CMB_I_map(ell, DlTT,
                   N_x, N_y,
                   X_width, Y_width, pix_size,
                   random_seed=None):
    """
    Makes a realization of a simulated CMB sky map given an input :math:`D_{\ell}` as a function
    of :math:`\ell`. This routine creates a 2D :math:`\ell` and :math:`C_{\ell}` spectrum and
    generates a Gaussian, random realization of the CMB in Fourier space using these. At last the
    map is converted into Image space, which will result us a randomly generated intensity map of
    the CMB temperature anisotropy. 
    
    Parameters
    ----------
    ell : numpy.array or array-like
        List of multipoles for which the angular power spectrum values
        were evaluated. Contains integers from 2 to :math:`l_{\mathrm{max}}`,
        where :math:`l_{\mathrm{max}}` is included.
    DlTT : numpy.array or array-like
        Transformed angular power spectrum bins (:math:`D_{l}`) for every
        multipole value in `ell`. The transformation is
        .. math::
                    D_{l} = \frac{l (l + 1)}{2 \pi} C_{l}.
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
    pix_size : float
        Size of a pixel in arcminutes.
    random_seed : float
        Sets the random seed for `numpy`'s Mersenne Twister pseudo-random number generator.
        
    Returns
    -------
    CMB_I : numpy.ndarray of shape (N_x, N_y)
        The generated intensity map of the CMB temperature anisotropy in Image space.
    ell2d : numpy.ndarray of shape (N_x, N_y)
        2D spectrum of the :math:`\ell` values.
    ClTT2d : numpy.ndarray of shape (N_x, N_y)
        2D realization of the :math:`C_{\ell}` power spectrum in Image space.
    FT_2d : numpy.ndarray of shape (N_x, N_y)
        Randomly generated Gaussian map in Fourier space.
    """
    # Convert Dl to Cl
    ClTT = DlTT * 2 * np.pi / (ell * (ell + 1))
    # Set the monopole and the dipole of the Cl spectrum to zero
    ClTT[0] = 0
    ClTT[1] = 0

    # Calculate distances to the center of the image on the map
    R = make_coordinates(N_x, N_y,
                         X_width, Y_width,
                         absolute=False)

    # Now make a 2D CMB power spectrum
    pix_to_rad = np.deg2rad(pix_size/60)       # Going from `pix_size` in arcmins to radians
    ell_scale_factor = 2 * np.pi / pix_to_rad  # Now relating the angular size in radians to multipoles
    ell2d = R * ell_scale_factor               # Making a fourier space analogue to the real space `R` vector

    # Making an expanded Cl spectrum (of zeros) that goes all the way to the size of the 2D `ell` vector
    # if the latter is shorter, than the `ell2d` vector
    _THRES = int(ell2d.max()) + 1
    if _THRES > ClTT.size:
        ClTT_expanded = np.zeros(int(ell2d.max()) + 1) 
        ClTT_expanded[0:(ClTT.size)] = ClTT        # Fill in the Cls until the max of the `ClTT` vector
    else:
        ClTT_expanded = ClTT

    # The 2D Cl spectrum is defined on the multiple vector set by the pixel scale
    ClTT2d = ClTT_expanded[ell2d.astype(int)] 
    
    # Now make a realization of the CMB with the given power spectrum in real space
    ## Generate a Gaussian random CMB map in Fourier space
    np.random.seed(random_seed)
    random_array_for_T = np.random.normal(0, 1, (N_y, N_x))
    FT_random_array_for_T = np.fft.fft2(random_array_for_T)   # Take FFT since we are in Fourier space
    FT_2d = np.sqrt(ClTT2d) * FT_random_array_for_T           # We take the sqrt since the power spectrum is T^2
    
    ## Converting the random map to real space
    # Move back from ell space to real space
    CMB_I = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
    # Move back to pixel space for the map
    CMB_I /= (pix_size /60 * np.pi/180)
    # We only want to plot the real component
    CMB_I = np.real(CMB_I)

    return(ell2d, ClTT2d, FT_2d, CMB_I)
  ###############################

def planck_cmap():
    """
    Generates the Planck CMB colormap from an input file, which stores the color values
    for the complete gradient. The colormap values was obtained from the following link:
    - https://github.com/zonca/paperplots/raw/master/data/Planck_Parchment_RGB.txt
    """
    cmap = ListedColormap(np.loadtxt(data + 'Planck_Parchment_RGB.txt')/255.)
    cmap.set_bad('black')   # color of missing pixels
    cmap.set_under('white') # color of background
    
    return cmap

def plot_CMB_map(CMB_I, X_width, Y_width,
                 c_min, c_max, cmap=None,
                 save=False, save_filename='default_name_cmb',
                 no_axis=False, no_grid=True, no_title=True,
                 no_cbar=False):
    """
    Plots the generated rectangular intensity map of the CMB temperature anisotropy.
    
    Parameters
    ----------
    CMB_I : numpy.ndarray of shape (N_x, N_y)
        The generated intensity map of the CMB temperature anisotropy in real space.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
    cmap : str or a matplotlib.colors.* colormap
        The colormap used to shade the pixels. By default the routine uses the classic Planck CMB
        colormap.
    c_min : float
        The lower limit for plotted values. All values of the input matrix below this limit will be
        scaled up to this value.
    c_max : float
        The upper limit for plotted values. All values of the input matrix below this limit will be
        scaled up to this value.
    save : bool
        If `True`, then saves the image into the output folder, under the name `save_filename`.
    no_axis : bool
        If `True`, then the axis labels and ticks will be hidden.
    no_grid : bool
        If `True`, then the gridlines will be hidden.
    no_cbar : bool
        If `True`, then the colorbar will be hidden.
    """
    # Setup figure size
    f_size = (12 * X_width/Y_width) if X_width > Y_width else (12 * Y_width/X_width)
    fig, axes = plt.subplots(figsize=(f_size, f_size),
                             facecolor='black', subplot_kw={'facecolor' : 'black'})
    axes.set_aspect('equal')
    if no_axis : axes.axis('off')
    if no_grid : axes.grid(False)
    
    # Convert 'CMB_I' to display properly
    _map = CMB_I.copy()
    _map[CMB_I < c_min] = c_min
    _map[CMB_I > c_max] = c_max
    
    # Set colormap for the image
    colormap = planck_cmap() if cmap is None else cmap

    im = axes.imshow(_map,
                     cmap=colormap, vmin=c_min, vmax=c_max,
                     interpolation='bilinear', origin='lower')
    im.set_extent([-X_width/2,X_width/2, -Y_width/2,Y_width/2])
    
    if not no_title : axes.set_title('map mean : {0:.3f} | map rms : {1:.3f}'.format(np.mean(CMB_I), np.std(CMB_I)),
                                     color='white', fontsize=axistitlesize, fontweight='bold', pad=10)
    axes.set_xlabel('Angle $[^\circ]$', color='white', fontsize=axislabelsize, fontweight='bold')
    axes.set_ylabel('Angle $[^\circ]$', color='white', fontsize=axislabelsize, fontweight='bold')
    axes.tick_params(axis='both', which='major', colors='white', labelsize=axisticksize, labelrotation=42, pad=10)
    
    if not no_cbar:
        # Create an axis on the right side of `axes`. The width of `cax` will be 2%
        # of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='2%', pad=0.1)
        cbar = plt.colorbar(mappable=im, cax=cax)#, shrink=0.735, pad=0.02)
        cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='white')
        cbar.set_label('Temperature [$\mu$K]', color='white',
                       fontsize=axiscbarfontsize+8, rotation=90, labelpad=12)

    if save:
        if not os.path.exists(out):
            os.makedirs(out)
        f = save_filename.split('.')
        fn = f[0]
        ff = 'png' if len(f) == 1 else f[1]
        plt.savefig(out + fn + '.' + ff,
                    format=ff, dpi=200,
                    facecolor='black', edgecolor='black',
                    bbox_inches='tight')
    plt.show()
  ###############################

def plot_steps_2D_ps(ClTT2d,
                     X_width, Y_width,
                     no_axis=False, no_grid=True):
    """
    Plots the logarithm of the 2D :math:`C_{l}` power spectrum in Image space.
    
    Parameters
    ----------
    ClTT2d : numpy.ndarray of shape (N_x, N_y)
        2D realization of the :math:`C_{\ell}` power spectrum in Image space.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
    no_axis : bool
        If `True`, then the axis labels and ticks will be hidden.
    no_grid : bool
        If `True`, then the gridlines will be hidden.
    """
    f_size = (12 * X_width/Y_width) if X_width > Y_width else (12 * Y_width/X_width)
    fig, axes = plt.subplots(figsize=(f_size, f_size))
    fig.subplots_adjust(hspace=0.20)

    axes.set_aspect('equal')
    if no_axis : axes.axis('off')
    if no_grid : axes.grid(False)

    ### PLOT 1. -- Logarithm of the 2D Cl spectrum
    ## Set 0 values to the minimum of the non-zero values to avoid `ZeroDivision error` in `np.log()`
    ClTT2d[ClTT2d == 0] = np.min(ClTT2d[ClTT2d != 0]) 
    im = axes.imshow(np.log(ClTT2d), vmin=None, vmax=None,
                     interpolation='bilinear', origin='lower', cmap=cm.RdBu_r)
    im.set_extent([-X_width/2,X_width/2, -Y_width/2,Y_width/2])
    
    axes.set_title('Log. of the 2D $C_{\ell}$ spectrum in Image space',
                   fontsize=axistitlesize, fontweight='bold')
    axes.set_xlabel('Angle $[^\circ]$', fontsize=axislabelsize, fontweight='bold')
    axes.set_ylabel('Angle $[^\circ]$', fontsize=axislabelsize, fontweight='bold')
    axes.tick_params(axis='both', which='major', labelsize=axisticksize, rotation=42, pad=10)
    
    ## Create an axis on the right side of `axes`. The width of `cax` will be 5%
    ## of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = plt.colorbar(mappable=im, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.ax.yaxis.get_offset_text().set(size=axiscbarfontsize)
    cbar.set_label('Log-temperature [$\mu$K]', fontsize=axiscbarfontsize+8,
                   rotation=90, labelpad=19)

def plot_steps_2D_gauss(ell2d, FT_2d, CMB_2D,
                        X_width, Y_width,
                        no_axis=False, no_grid=True):
    """
    Plots the the real part of the generated 2D Gauissian noise in
    Fourier space.
    
    Parameters
    ----------
    ell2d : numpy.ndarray of shape (N_x, N_y)
        2D spectrum of the :math:`\ell` values.
    FT_2d : numpy.ndarray of shape (N_x, N_y)
        The generated 2D Gaussian map in Fourier space.
    CMB_2D : numpy.ndarray of shape (N_x, N_y)
        The randomly generated CMB map in 2D Fourier space, calculated using the
        originally created random 2D Gaussian map.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
    no_axis : bool
        If `True`, then the axis labels and ticks will be hidden.
    no_grid : bool
        If `True`, then the gridlines will be hidden.
    """
    fig, axes = plt.subplots(figsize=(12,12))
    axes.set_aspect('equal')
    if no_axis : axes.axis('off')
    if no_grid : axes.grid(False)
    
    ### PLOT 2. -- CMB in Fourier space
    im = axes.imshow(CMB_2D, vmin=0, vmax=np.max(CMB_2D),
                     interpolation='bilinear', origin='lower', cmap=cm.RdBu_r)
    ext = ell2d.max()   # Upper border of the extent of the whole map
    im.set_extent([-ext, ext, -ext, ext])
    
    lim = int(ext / 3)  # Limit to be plotted
    axes.set_xlim(-lim, lim)
    axes.set_ylim(-lim, lim)
    
    axes.set_title('Real part of the 2D CMB map in Fourier space',
                   fontsize=axistitlesize, fontweight='bold')
    axes.set_xlabel('Multipoles $(\ell)$', fontsize=axislabelsize, fontweight='bold')
    axes.set_ylabel('Multipoles $(\ell)$', fontsize=axislabelsize, fontweight='bold')
    axes.tick_params(axis='both', which='major', labelsize=axisticksize, rotation=42, pad=10)
    
    ## Set ticks and ticklabels
    ticks = np.linspace(-lim, lim, 11)
    ticklabels = np.array(['{0:.0f}'.format(t) for t in ticks])
    axes.set_xticks(ticks)
    axes.set_xticklabels(ticklabels)
    axes.set_yticks(ticks)
    axes.set_yticklabels(ticklabels)
    
    ## Create an axis on the right side of `axes`. The width of `cax` will be 5%
    ## of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = plt.colorbar(mappable=im, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.ax.yaxis.get_offset_text().set(size=axiscbarfontsize)
    cbar.set_label('Frequency', fontsize=axiscbarfontsize+8,
                   rotation=90, labelpad=19)
    
    plt.show()
  ###############################

def poisson_source_component(N_x, N_y,
                             pix_size,
                             number_of_sources, amplitude_of_sources):
    """
    Makes a realization of the naive foreground point source map with Poisson
    distribution.
    
    Parameters:
    -----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    pix_size : float
        Size of a pixel in arcminutes.
    number_of_sources : int
        Number of Poisson distributed point sources on the source map.
    amplitude_of_sources : float
        Amplitude of point sources, which serves as the `lambda` parameter
        for the Poisson-distribution used to choose random points from.

    Returns:
    --------
    PSMap : numpy.ndarray of shape (N_x, N_y)
        The Poisson distributed point sources marked on the map in the form of a 2D matrix.
    """
    PSmap = np.zeros((N_x, N_y))
    # We throw random numbers repeatedly with amplitudes given by a Poisson distribution around the mean amplitude
    for i in range(number_of_sources):
        pix_x = int(N_x*np.random.rand())
        pix_y = int(N_y*np.random.rand()) 
        PSmap[pix_x, pix_y] += np.random.poisson(lam=amplitude_of_sources)

    return(PSmap.T)
  ############################### 

def exponential_source_component(N_x, N_y,
                                 pix_size,
                                 number_of_sources_EX, amplitude_of_sources_EX):
    """
    Makes a realization of the naive foreground point source map with exponential
    distribution.
    
    Parameters:
    -----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    pix_size : float
        Size of a pixel in arcminutes.
    number_of_sources_EX : int
        Number of exponentially distributed point sources on the source map.
    amplitude_of_sources_EX : float
        Amplitude of point sources, which serves as the scale parameter
        for the exponential distribution

    Returns:
    --------
    PSMap : numpy.ndarray of shape (N_x, N_y)
        The exponentially distributed point sources marked on the map in the form of a 2D matrix.
    """
    PSmap = np.zeros((N_x, N_y))
    # We throw random numbers repeatedly with amplitudes given by an exponential
    # distribution around the mean amplitude
    for i in range(number_of_sources_EX):
        pix_x = int(N_x*np.random.rand()) 
        pix_y = int(N_y*np.random.rand()) 
        PSmap[pix_x,pix_y] += np.random.exponential(scale=amplitude_of_sources_EX)

    return(PSmap.T)
  ###############################

def beta_function(N_x, N_y,
                  X_width, Y_width, pix_size,
                  SZ_beta, SZ_theta_core):
    """
    Makes a 2D beta function map to mock the intensity spread of Sunyaev–Zeldovich
    sources. 
    
    Parameters:
    -----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
    pix_size : float
        Size of a pixel in arcminutes.
    SZ_beta : float
        desc
    SZ_theta_core : float
        desc

    Returns:
    --------
    beta : numpy.ndarray of shape (N_x, N_y)
    """
    # Calculate distances to the center of the image on the map
    R = make_coordinates(N_x, N_y,
                         X_width, Y_width,
                         absolute=True)
    
    beta = (1 + (R/SZ_theta_core)**2)**((1 - 3*SZ_beta)/2)

    return(beta)

def SZ_source_component(N_x, N_y,
                        X_width, Y_width, pix_size,
                        number_of_SZ_clusters, mean_amplitude_of_SZ_clusters,
                        SZ_beta, SZ_theta_core):
    """
    Makes a realization of a naive Sunyaev–Zeldovich effect map.

    Parameters:
    -----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
    pix_size : float
        Size of a pixel in arcminutes.
    number_of_SZ_clusters : int
        Number of the Sunyaev–Zeldovich sources on the map.
    mean_amplitude_of_SZ_clusters : float
        Mean amplitude/size of the Sunyaev–Zeldovich sources on the map.
    SZ_beta : float
        desc
    SZ_theta_core : float
        desc

    Returns:
    --------
    SZmap : numpy.ndarray of shape (N_x, N_y)
        The intensity map of the generated SZ sources with beta
        profiles.
    SZcat : numpy.ndarray of shape (3, number_of_SZ_clusters)
        Catalogue of SZ sources, containing (X, Y, amplitude) in each entry
    """

    # Placeholder for the SZ map
    SZmap = np.zeros([N_x,N_y])
    # Catalogue of SZ sources, X, Y, amplitude
    SZcat = np.zeros([3, number_of_SZ_clusters])
    # Make a distribution of point sources with varying amplitude
    for i in range(number_of_SZ_clusters):
        pix_x = int(N_x*np.random.rand())
        pix_y = int(N_y*np.random.rand())
        pix_amplitude = np.random.exponential(mean_amplitude_of_SZ_clusters)*(-1)
        SZcat[0,i] = pix_x
        SZcat[1,i] = pix_y
        SZcat[2,i] = pix_amplitude
        SZmap[pix_x,pix_y] += pix_amplitude

    # Make a beta function
    beta = beta_function(N_x, N_y, X_width, Y_width, pix_size, SZ_beta, SZ_theta_core)

    # Convolve the beta function with the point source amplitude to get the SZ map
    FT_beta = np.fft.fft2(np.fft.fftshift(beta))
    FT_SZmap = np.fft.fft2(np.fft.fftshift(SZmap))
    SZmap = np.fft.fftshift(np.real(np.fft.ifft2(FT_beta.T*FT_SZmap)))

    return(SZmap.T, SZcat, beta.T)
  ############################### 

def make_2d_gaussian_beam(N_x, N_y,
                          beam_size_fwhp):
    """
    Creates a 2D Gaussian function.
    
    Paramters
    ---------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    beam_size_fwhp : float
        Mean FWHM of the simulated beam.
    
    Returns
    -------
    gaussian : numpy.ndarray of shape (N_x, N_y)
        The 2D Gaussian function over the input domain.
    """
    # Calculate distances to the center of the image on the map
    R = make_coordinates(N_x, N_y,
                         X_width, Y_width,
                         absolute=True)

    ## Make a 2D Gaussian 
    # Planck's beam sigma values are approximately similar to this in magnitude
    beam_sigma = beam_size_fwhp / np.sqrt(8 * np.log(2))
    gaussian = np.exp(-0.5 * (R/beam_sigma)**2)
    gaussian = gaussian / np.sum(gaussian)

    return(gaussian)

def convolve_map_with_gaussian_beam(Map,
                                    N_x, N_y,
                                    beam_size_fwhp):
    """
    Convolves a map with a Gaussian beam pattern.
    
    Paramters
    ---------
    Map : numpy.ndarray of shape (N_x, N_y)
        The input map to be convolved with the generated Gaussian.
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    beam_size_fwhp : float
        Mean FWHM of the simulated beam.
    
    Returns
    -------
    convolved_map : numpy.ndarray of shape (N_x, N_y)
        The beam convolved with the input map.
    """ 
    # Make a 2D Gaussian 
    gaussian = make_2d_gaussian_beam(N_x, N_y,
                                     beam_size_fwhp)
  
    ## Do the convolution
    # 1. First add the shift so that it is central
    FT_gaussian = np.fft.fft2(np.fft.fftshift(gaussian))
    # 2. Shift the map too
    FT_map = np.fft.fft2(np.fft.fftshift(Map))
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(FT_gaussian*FT_map))) 
    
    return(convolved_map)
  ###############################  

def gen_white_noise(N_x, N_y,
                    pix_size,
                    white_noise_level):
    """
    Makes a white noise map.
    
    Parameters
    ----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    pix_size : float
        Size of a pixel in arcminutes.
    white_noise_level : float
    
    Returns
    -------
    white_noise : numpy.ndarray of shape (N_x, N_y)
        The white noise map.
    """
    white_noise = np.random.normal(0,1,(N_x,N_y)) * white_noise_level/pix_size
    
    return(white_noise)

def gen_atmospheric_noise(N_x, N_y,
                          X_width, Y_width, pix_size,
                          atmospheric_noise_level):
    """
    Makes an atmospheric noise map.
    
    Parameters
    ----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    X_width : float
        Size of the map along the X-axis in degrees.
    Y_width : float
        Size of the map along the Y-axis in degrees.
    pix_size : float
        Size of a pixel in arcminutes.
    atmospheric_noise_level : float
    
    Returns
    -------
    atmospheric_noise : numpy.ndarray of shape (N_x, N_y)
        The atmospheric noise map.
    """
    # Calculate distances to the center of the image on the map
    R = make_coordinates(N_x, N_y,
                         X_width, Y_width,
                         absolute=True)
    # Convert distances in arcmin to degrees
    R /= 60
    mag_k = 2 * np.pi/(R + 0.01)  # 0.01 is a regularization factor
    atmospheric_noise = np.fft.fft2(np.random.normal(0,1,(N_x,N_y)))
    atmospheric_noise  = np.fft.ifft2(atmospheric_noise.T * np.fft.fftshift(mag_k**(5/3)))
    atmospheric_noise = atmospheric_noise * atmospheric_noise_level/pix_size
    
    return(atmospheric_noise)

def gen_one_over_f_noise(N_x, N_y,
                         pix_size,
                         one_over_f_noise_level):
    """
    Generates 1/f noise in the X direction.
    
    Parameters
    ----------
    N_x : int
        Number of pixels in the linear dimension along the X-axis.
    N_y : int
        Number of pixels in the linear dimension along the Y-axis.
    pix_size : float
        Size of a pixel in arcminutes.
    one_over_f_noise_level : float
    
    Returns
    -------
    one_over_f_noise : numpy.ndarray of shape (N_x, N_y)
        The 1/f noise map along the X direction.
    """
    x = np.linspace(-N_x/2, N_x/2, N_x)
    y = np.linspace(-N_y/2, N_y/2, N_y)
    X, _ = np.meshgrid(x, y)
    X *= pix_size / 60                       # Convert to [degrees]
    kx = 2 * np.pi/(X+0.01)                  # 0.01 is a regularization factor
    one_over_f_noise = np.fft.fft2(np.random.normal(0,1,(N_x,N_y)))
    one_over_f_noise = np.fft.ifft2(one_over_f_noise.T * np.fft.fftshift(kx)) * one_over_f_noise_level/pix_size
    
    return(one_over_f_noise)
    
def make_noise_map(N_x, N_y,
                   X_width, Y_width, pix_size,
                   white_noise_level=10,
                   atmospheric_noise_level=0.1, one_over_f_noise_level=0.2):
    """
    Makes a realization of instrument noise, atmosphere and :math:`1/f`
    noise level set at 1 degrees.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Make a white noise map
    white_noise = gen_white_noise(N_x, N_y,
                                 pix_size,
                                 white_noise_level)
 
    # Make an atmosperhic noise map
    atmospheric_noise = 0
    if (atmospheric_noise_level != 0):
        atmospheric_noise = gen_atmospheric_noise(N_x, N_y,
                                                  X_width, Y_width, pix_size,
                                                  atmospheric_noise_level)

    # Make a 1/f map, along a single direction to illustrate striping 
    one_over_f_noise = 0
    if (one_over_f_noise_level != 0): 
        one_over_f_noise = gen_one_over_f_noise(N_x, N_y,
                                                pix_size,
                                                one_over_f_noise_level)

    noise_map = np.real(white_noise.T + atmospheric_noise + one_over_f_noise)
    return(noise_map)