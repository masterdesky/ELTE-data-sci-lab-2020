import sys
import numpy as np
import astropy.io.fits as fits

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.io.fits as fits

axistitlesize = 20
axisticksize = 17
axislabelsize = 26
axislegendsize = 23
axistextsize = 20
axiscbarfontsize = 15


def make_CMB_I_map(ell, DlTT,
                   N_x=2**10, N_y=2**10//2,
                   X_width=360, Y_width=180, pix_size=0.5,
                   random_seed=None):
    """
    Makes a realization of a simulated CMB sky map given an input :math:`D_{\ell}` as a function
    of :math:`\ell`. This routine creates a 2D :math:`\ell` and :math:`C_{\ell}` spectrum and
    generates a Gaussian, random realization of the CMB in Fourier space using these. At last the
    map is converted into Image space, which will result us a randomly generated intensity map of
    the CMB temperature anisotropy. 
    
    Parameters
    ----------
    ell : list or array-like
        List of multipoles for which the angular power spectrum values
        were evaluated. Contains integers from 2 to :math:`l_{\mathrm{max}}`,
        where :math:`l_{\mathrm{max}}` is included.
    DlTT : list or array-like
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
    CMB_I : numpy.ndarray
        The generated intensity map of the CMB temperature anisotropy in Image space.
    ell2d : numpy.ndarray
        2D spectrum of the :math:`\ell` values.
    ClTT2d : numpy.ndarray
        2D realization of the :math:`C_{\ell}` power spectrum in Image space.
    FT_2d : numpy.ndarray
        Randomly generated Gaussian map in Fourier space.
    """
    # Convert Dl to Cl
    ClTT = DlTT * 2 * np.pi / (ell * (ell + 1))
    # Set the monopole and the dipole of the Cl spectrum to zero
    ClTT[0] = 0
    ClTT[1] = 0

    # Make a 2D equirectangular coordinate system
    x = np.linspace(-np.deg2rad(X_width)/2, np.deg2rad(X_width)/2, N_x)
    y = np.linspace(-np.deg2rad(Y_width)/2, np.deg2rad(Y_width)/2, N_y)
    X, Y = np.meshgrid(x, y)
    # Radial component `R`
    R = 2 * np.arcsin(np.sqrt(np.sin(X/2)**2 + np.cos(X) * np.sin(Y/2)**2))
    R *= (np.sqrt(pix_size)/2) / np.max(R)

    # Now make a 2D CMB power spectrum
    pix_to_rad = (pix_size/60 * np.pi/180)     # Going from `pix_size` in arcmins to degrees and then degrees to radians
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

    return(CMB_I, ell2d, ClTT2d, FT_2d)
  ###############################

def planck_cmap():
    """
    Generates the Planck CMB colormap from an input file, which stores the color values
    for the complete gradient. The colormap values was obtained from the following link:
    - https://github.com/zonca/paperplots/raw/master/data/Planck_Parchment_RGB.txt
    """
    cmap = ListedColormap(np.loadtxt('Planck_Parchment_RGB.txt')/255.)
    cmap.set_bad('black')   # color of missing pixels
    cmap.set_under('white') # color of background
    
    return cmap

def plot_CMB_map(CMB_I, X_width, Y_width,
                 cmap=None, c_min=-400, c_max=400,
                 save=False, no_axis=False, no_grid=True):
    """
    Plots the generated rectangular intensity map of the CMB temperature anisotropy.
    
    Parameters
    ----------
    CMB_I : numpy.ndarray
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
    """
    fig, axes = plt.subplots(figsize=(16,16),
                             facecolor='black', subplot_kw={'facecolor' : 'black'})
    axes.set_aspect('equal')
    if no_axis : axes.axis('off')
    if no_grid : axes.grid(False)
    
    # Convert 'CMB_I' to display properly
    CMB_I[CMB_I < c_min] = c_min
    CMB_I[CMB_I > c_max] = c_max
    
    # Set colormap for the image
    colormap = planck_cmap() if cmap is None else cmap

    im = axes.imshow(CMB_I,
                     cmap=colormap, vmin=c_min, vmax=c_max,
                     interpolation='bilinear', origin='lower')
    im.set_extent([-X_width/2,X_width/2, -Y_width/2,Y_width/2])
    
    axes.set_title('map mean : {0:.3f} | map rms : {1:.3f}'.format(np.mean(CMB_I), np.std(CMB_I)),
                   color='white', fontsize=axistitlesize, fontweight='bold', pad=10)
    axes.set_xlabel('Angle $[^\circ]$', color='white', fontsize=axislabelsize, fontweight='bold')
    axes.set_ylabel('Angle $[^\circ]$', color='white', fontsize=axislabelsize, fontweight='bold')
    axes.tick_params(axis='both', which='major', colors='white', labelsize=axisticksize, labelrotation=42, pad=10)
    
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
        plt.savefig(out + 'CMB_I_map_gen.png',
                    format='png', dpi=200,
                    facecolor='black', edgecolor='black',
                    bbox_inches='tight')
    plt.show()
  ###############################

def plot_CMB_steps(ell2d, ClTT2d, CMB_2D,
                   X_width, Y_width,
                   no_axis=False, no_grid=True):
    """
    It plots... something...    
    
    Parameters
    ----------
    ell2d : numpy.ndarray
        2D spectrum of the :math:`\ell` values.
    ClTT2d : numpy.ndarray
        2D realization of the :math:`C_{\ell}` power spectrum in Image space.
    CMB_2D : numpy.ndarray
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
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*13, nrows*13))
    fig.subplots_adjust(hspace=0.20)
    
    titles = ['Log. of the 2D $C_{\ell}$ spectrum in Image space',
              'Real part of the 2D $C_{\ell}$ spectrum in Fourier space']
    labels = ['Angle $[^\circ]$',
              'Multipoles $(\ell)$']
    
    ### PLOT 1. -- Logarithm of the 2D Cl spectrum
    ax = axes[0]
    ax.set_aspect('equal')
    if no_axis : ax.axis('off')
    if no_grid : ax.grid(False)

    ## Set 0 values to the minimum of the non-zero values to avoid `ZeroDivision error` in `np.log()`
    ClTT2d[ClTT2d == 0] = np.min(ClTT2d[ClTT2d != 0]) 
    im_1 = ax.imshow(np.log(ClTT2d), vmin=None, vmax=None,
                     interpolation='bilinear', origin='lower', cmap=cmap.RdBu_r)
    im_1.set_extent([-X_width/2,X_width/2, -Y_width/2,Y_width/2])
    ## Create an axis on the right side of `axes`. The width of `cax` will be 5%
    ## of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = plt.colorbar(mappable=im_1, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.ax.yaxis.get_offset_text().set(size=axiscbarfontsize)
    cbar.set_label('Log-temperature [$\mu$K]', fontsize=axiscbarfontsize+8, rotation=90, labelpad=19)
    
    ### PLOT 2. -- 2D CMB map in Fourier space
    ax = axes[1]
    ax.set_aspect('equal')
    if no_axis : ax.axis('off')
    if no_grid : ax.grid(False)
    
    im_2 = ax.imshow(CMB_2D, vmin=0, vmax=np.max(np.conj(FT_2d) * FT_2d * ell2d * (ell2d + 1) / (2 * np.pi)).real,
                     interpolation='bilinear', origin='lower', cmap=cmap.RdBu_r)
    x_min, x_max = int(ell2d.min()), int(ell2d.max())
    im_2.set_extent([x_min, x_max, x_min, x_max])
    
    ## Set ticks and ticklabels
    ticks = np.linspace(x_min, x_max, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    
    ## Create an axis on the right side of `axes`. The width of `cax` will be 5%
    ## of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = plt.colorbar(mappable=im_2, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.ax.yaxis.get_offset_text().set(size=axiscbarfontsize)
    cbar.set_label('Frequency', fontsize=axiscbarfontsize+8, rotation=90, labelpad=19)
    
    for i in range(nrows):
        ax = axes[i]
        
        ax.set_title(titles[i], fontsize=axistitlesize, fontweight='bold')
        ax.set_xlabel(labels[i], fontsize=axislabelsize, fontweight='bold')
        ax.set_ylabel(labels[i], fontsize=axislabelsize, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=axisticksize, rotation=42, pad=10)
    
    plt.show()
  ###############################

def poisson_source_component(N, pix_size, number_of_sources, amplitude_of_sources):
    """
    Makes a realization of a naive Poisson distributed point source map.
    
    Parameters:
    -----------
    N : int
        Number of pixels in the linear dimension.
    pix_size : float
        Size of a pixel in arcminutes.
    number_of_sources : int
        Number of Poisson distributed point sources on the source map.
    amplitude_of_sources : float
        Amplitude of point sources, which serves as the `lambda` parameter
        for the Poisson-distribution used to choose random points from.

    Returns:
    --------
    PSMap : array of shape (N, N)
        The Poisson distributed point sources marked on the map in the form of a 2D matrix.
    """
    PSMap = np.zeros([int(N),int(N)])
    # We throw random numbers repeatedly with amplitudes given by a Poisson distribution around the mean amplitude
    for i in range(number_of_sources):
        pix_x = int(N*np.random.rand())
        pix_y = int(N*np.random.rand()) 
        PSMap[pix_x, pix_y] += np.random.poisson(lam=amplitude_of_sources)

    return(PSMap)   
  ############################### 

def exponential_source_component(N, pix_size, number_of_sources_EX, amplitude_of_sources_EX):
    """
    Makes a realization of a naive exponentially-distributed point source map
    
    Parameters:
    -----------
    N : int
        Number of pixels in the linear dimension.
    pix_size : float
        Size of a pixel in arcminutes.
    number_of_sources_EX : int
        Number of exponentially distributed point sources on the source map.
    amplitude_of_sources_EX : float
        Amplitude of point sources, which serves as the scale parameter
        for the exponential distribution

    Returns:
    --------
    PSMap : array of shape (N, N)
        The exponentially distributed point sources marked on the map in the form of a 2D matrix.
    """
    PSMap = np.zeros([int(N), int(N)])
    # We throw random numbers repeatedly with amplitudes given by an exponential distribution around the mean amplitude
    for i in range(number_of_sources_EX):
        pix_x = int(N*np.random.rand()) 
        pix_y = int(N*np.random.rand()) 
        PSMap[pix_x,pix_y] += np.random.exponential(scale=amplitude_of_sources_EX)

    return(PSMap)  
  ############################### 

def SZ_source_component(N, pix_size, number_of_SZ_clusters, mean_amplitude_of_SZ_clusters, SZ_beta, SZ_theta_core, do_plots):
    """
    Makes a realization of a naive SZ effect map.

    Parameters:
    -----------
    N : int
        Number of pixels in the linear dimension.
    pix_size : float
        Size of a pixel in arcminutes.
    number_of_SZ_clusters : int
        desc
    mean_amplitude_of_SZ_clusters : float
        desc
    SZ_beta : float
        desc
    SZ_theta_core : float
        desc
    do_plots : bool
        desc

    Returns:
    --------
    SZMap : 
        desc
    SZcat : 
        desc
    """

    # Placeholder for the SZ map
    SZmap = np.zeros([N,N])
    # Catalogue of SZ sources, X, Y, amplitude
    SZcat = np.zeros([3, number_of_SZ_clusters])
    # make a distribution of point sources with varying amplitude
    for i in range(number_of_SZ_clusters):
        pix_x = int(N*np.random.rand())
        pix_y = int(N*np.random.rand())
        pix_amplitude = np.random.exponential(mean_amplitude_of_SZ_clusters)*(-1)
        SZcat[0,i] = pix_x
        SZcat[1,i] = pix_y
        SZcat[2,i] = pix_amplitude
        SZmap[pix_x,pix_y] += pix_amplitude

    if do_plots:
        hist, bins = np.histogram(SZMap,bins = 50,range=[SZmap.min(),-10])
        width = 1.0 * np.diff(bins).min()
        centers = (bins[1:] + bins[:-1]) / 2
        fig, axes = plt.subplots(figsize=(12,12))
        axes.set_yscale('log')
        axes.bar(centers, hist, width=width,
                 ec='black', lw=0.5)
        axes.set_xlabel('Source amplitude [$\mu$K]', fontsize=axislabelsize, fontweight='bold')
        axes.set_ylabel('Number of pixels', fontsize=axislabelsize, fontweight='bold')
        axes.tick_params(axis='both', which='major', labelsize=axisticksize)
        plt.show()

    # make a beta function
    beta = beta_function(N, pix_size, SZ_beta, SZ_theta_core)

    # convolve the beta function with the point source amplitude to get the SZ map
    # NOTE: you should go back to the Intro workshop for more practice with convolutions!
    FT_beta = np.fft.fft2(np.fft.fftshift(beta))
    FT_SZmap = np.fft.fft2(np.fft.fftshift(SZmap))
    SZmap = np.fft.fftshift(np.real(np.fft.ifft2(FT_beta*FT_SZmap)))

    # return the SZ map
    return(SZmap, SZcat)
  ############################### 

def beta_function(N, pix_size, SZ_beta, SZ_theta_core):
    """
    Makes a beta function.
    
    Parameters:
    -----------
    N : int
        Number of pixels in the linear dimension.
    pix_size : float
        Size of a pixel in arcminutes.
    SZ_beta : float
        desc
    SZ_theta_core : float
        desc

    Returns:
    --------
    beta : array of shape ()
    """
    ones = np.ones(N)
    inds  = (np.arange(N) + 0.5 - N/2) * pix_size
    X = np.outer(ones, inds)
    Y = np.transpose(X)
    # Compute the same real-space R function as before for the PS
    R = np.sqrt(X**2 + Y**2)
    
    beta = (1 + (R/SZ_theta_core)**2)**((1-3*SZ_beta)/2)

    # return the beta function map
    return(beta)
  ############################### 

def convolve_map_with_gaussian_beam(N,pix_size,beam_size_fwhp,Map):
    "convolves a map with a gaussian beam pattern.  NOTE: pix_size and beam_size_fwhp need to be in the same units" 
    # make a 2d gaussian 
    gaussian = make_2d_gaussian_beam(N,pix_size,beam_size_fwhp)
  
    # do the convolution
    FT_gaussian = np.fft.fft2(np.fft.fftshift(gaussian))
    FT_Map = np.fft.fft2(np.fft.fftshift(Map))
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(FT_gaussian*FT_Map)))
    
    # return the convolved map
    return(convolved_map)
  ###############################   

def make_2d_gaussian_beam(N,pix_size,beam_size_fwhp):
     # make a 2d coordinate system
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
  
    # make a 2d gaussian 
    beam_sigma = beam_size_fwhp / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.)
    gaussian = gaussian / np.sum(gaussian)
 
    # return the gaussian
    return(gaussian)
  ###############################  

def make_noise_map(N,pix_size,white_noise_level,atmospheric_noise_level,one_over_f_noise_level):
    "makes a realization of instrument noise, atmosphere and 1/f noise level set at 1 degrees"
    ## make a white noise map
    N=int(N)
    white_noise = np.random.normal(0,1,(N,N)) * white_noise_level/pix_size
 
    ## make an atmosperhic noise map
    atmospheric_noise = 0.
    if (atmospheric_noise_level != 0):
        ones = np.ones(N)
        inds  = (np.arange(N)+.5 - N/2.) 
        X = np.outer(ones,inds)
        Y = np.transpose(X)
        R = np.sqrt(X**2. + Y**2.) * pix_size /60. ## angles relative to 1 degrees  
        mag_k = 2 * np.pi/(R+.01)  ## 0.01 is a regularizaiton factor
        atmospheric_noise = np.fft.fft2(np.random.normal(0,1,(N,N)))
        atmospheric_noise  = np.fft.ifft2(atmospheric_noise * np.fft.fftshift(mag_k**(5/3.)))* atmospheric_noise_level/pix_size

    ## make a 1/f map, along a single direction to illustrate striping 
    oneoverf_noise = 0.
    if (one_over_f_noise_level != 0): 
        ones = np.ones(N)
        inds  = (np.arange(N)+.5 - N/2.) 
        X = np.outer(ones,inds) * pix_size /60. ## angles relative to 1 degrees 
        kx = 2 * np.pi/(X+.01) ## 0.01 is a regularizaiton factor
        oneoverf_noise = np.fft.fft2(np.random.normal(0,1,(N,N)))
        oneoverf_noise = np.fft.ifft2(oneoverf_noise * np.fft.fftshift(kx))* one_over_f_noise_level/pix_size

    ## return the noise map
    noise_map = np.real(white_noise + atmospheric_noise + oneoverf_noise)
    return(noise_map)
  ###############################
def Filter_Map(Map,N,N_mask):
    N=int(N)
    ## set up a x, y, and r coordinates for mask generation
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) 
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)  ## angles relative to 1 degrees  
    
    ## make a mask
    mask  = np.ones([N,N])
    mask[np.where(np.abs(X) < N_mask)]  = 0

    return apply_filter(Map,mask)


def apply_filter(Map,filter2d):
    ## apply the filter in fourier space
    FMap = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Map)))
    FMap_filtered = FMap * filter2d
    Map_filtered = np.real(np.fft.fftshift(np.fft.fft2(FMap_filtered)))
    
    ## return the output
    return(Map_filtered)



def cosine_window(N):
    "makes a cosine window for apodizing to avoid edges effects in the 2d FFT" 
    # make a 2d coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.)/N *np.pi ## eg runs from -pi/2 to pi/2
    X = np.outer(ones,inds)
    Y = np.transpose(X)
  
    # make a window map
    window_map = np.cos(X) * np.cos(Y)
   
    # return the window map
    return(window_map)
  ###############################


def average_N_spectra(spectra,N_spectra,N_ells):
    avgSpectra = np.zeros(N_ells)
    rmsSpectra = np.zeros(N_ells)
    
    # calcuate the average spectrum
    i = 0
    while (i < N_spectra):
        avgSpectra = avgSpectra + spectra[i,:]
        i = i + 1
    avgSpectra = avgSpectra/(1. * N_spectra)
    
    #calculate the rms of the spectrum
    i =0
    while (i < N_spectra):
        rmsSpectra = rmsSpectra +  (spectra[i,:] - avgSpectra)**2
        i = i + 1
    rmsSpectra = np.sqrt(rmsSpectra/(1. * N_spectra))
    
    return(avgSpectra,rmsSpectra)

def calculate_2d_spectrum(Map,delta_ell,ell_max,pix_size,N,Map2=None):
    "calculates the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"
    import matplotlib.pyplot as plt
    # make a 2d ell coordinate system
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    
    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array = np.zeros(N_bins)
    
    # get the 2d fourier transform of the map
    FMap = np.fft.ifft2(np.fft.fftshift(Map))
    if Map2 is None: FMap2 = FMap.copy()
    else: FMap2 = np.fft.ifft2(np.fft.fftshift(Map2))
    
    #print FMap
    PSMap = np.fft.fftshift(np.real(np.conj(FMap) * FMap2))
    #print PSMap
    # fill out the spectra
    i = 0
    while (i < N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array[i] = np.mean(PSMap[inds_in_bin])
        i = i + 1


    CL_array_new = CL_array[~np.isnan(CL_array)]
    ell_array_new = ell_array[~np.isnan(CL_array)]
    # return the power spectrum and ell bins
    return(ell_array_new,CL_array_new*np.sqrt(pix_size /60.* np.pi/180.)*2.)
