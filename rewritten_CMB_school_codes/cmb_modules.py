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


def make_CMB_T_map(ell, DlTT,
                   N=2**10, pix_size=0.5):
    """
    Makes a realization of a simulated CMB sky map given an input `DlTT` as a function of `ell`.
    
    Parameters:
    -----------
    ell : list or array-like
        desc
    DlTT : list or array-like
        desc
    N : int
        Number of pixel in the linear dimension
    pix_size : float
        Size of a pixel in arcminutes
        
    Returns:
    --------
    CMB_T : 
        desc
    CLTT2d : 
        desc
    FT_2d : 
        desc
    ell2d : 
        desc
    """
    # Convert Dl to Cl
    ClTT = DlTT * 2 * np.pi / (ell * (ell + 1))
    # Set the monopole and the dipole of the Cl spectrum to zero
    ClTT[0] = 0
    ClTT[1] = 0

    # Make a 2D real space coordinate system
    onesvec = np.ones(N)
    inds  = (np.arange(N) + 0.5 - N/2) /(N - 1) # create an array of size `N` between -0.5 and +0.5
    # Compute the outer product matrix:
    #      X[i, j] = onesvec[i] * inds[j] for i,j in range(N)
    # which is just `N` rows copies of `inds` - for the x dimension
    X = np.outer(onesvec, inds) 
    # Compute the transpose for the y dimension
    Y = np.transpose(X)
    # Radial component `R`
    R = np.sqrt(X**2 + Y**2)
    
    # Now make a 2D CMB power spectrum
    pix_to_rad = (pix_size/60 * np.pi/180)     # Going from `pix_size` in arcmins to degrees and then degrees to radians
    ell_scale_factor = 2 * np.pi / pix_to_rad  # Now relating the angular size in radians to multipoles
    ell2d = R * ell_scale_factor               # Making a fourier space analogue to the real space `R` vector
    ClTT_expanded = np.zeros(int(ell2d.max()) + 1) 
    # Making an expanded Cl spectrum (of zeros) that goes all the way to the size of the 2D `ell` vector
    ClTT_expanded[0:(ClTT.size)] = ClTT        # Fill in the Cls until the max of the `ClTT` vector

    # The 2D Cl spectrum is defined on the multiple vector set by the pixel scale
    ClTT2d = ClTT_expanded[ell2d.astype(int)] 
    
    # Now make a realization of the CMB with the given power spectrum in real space
    random_array_for_T = np.random.normal(0, 1, (N,N))
    FT_random_array_for_T = np.fft.fft2(random_array_for_T)   # Take FFT since we are in Fourier space
    FT_2d = np.sqrt(ClTT2d) * FT_random_array_for_T           # We take the sqrt since the power spectrum is T^2
    
    # Move back from ell space to real space
    CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
    # Move back to pixel space for the map
    CMB_T /= (pix_size /60 * np.pi/180)
    # We only want to plot the real component
    CMB_T = np.real(CMB_T)

    return(CMB_T, ClTT2d, FT_2d, ell2d)
  ###############################

def plot_CMB_map(cmb_map, X_width, Y_width,
                 c_min=-400, c_max=400):
    """
    
    """
    fig, axes = plt.subplots(figsize=(12,12))
    
    im = axes.imshow(cmb_map, vmin=c_min, vmax=c_max,
                     interpolation='bilinear', origin='lower', cmap=cmap.RdBu_r)
    im.set_extent([0,X_width, 0,Y_width])
    
    axes.set_title('map mean : {0} map rms : {1}'.format(np.mean(cmb_map), np.std(cmb_map)),
                   fontsize=axistitlesize, fontweight='bold')
    axes.set_xlabel('Angle $[^\circ]$', fontsize=axislabelsize, fontweight='bold')
    axes.set_ylabel('Angle $[^\circ]$', fontsize=axislabelsize, fontweight='bold')
    axes.tick_params(axis='both', which='major', labelsize=axisticksize)
    
    # Create an axis on the right side of `axes`. The width of `cax` will be 5%
    # of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(mappable=im, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.set_label('Temperature [$\mu$K]', fontsize=axiscbarfontsize+8, rotation=90, labelpad=22)
    
    plt.show()
  ###############################

def plot_CMB_steps(ClTT2d, CMB_2D, ell2d, X_width, Y_width):
    """
    Visualizes 
    
    
    """
    nrows = 1
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*12, nrows*12))
    fig.subplots_adjust(wspace=0.3)
    
    titles = ['2D Cl spectrum in Image space', 'Real part of the 2D Cl spectrum in Fourier space']
    labels = ['Angle $[^\circ]$', 'Frequency $[1/^\circ]$']
    
    ### PLOT 1.
    ax = axes[0]
    # Set 0 values to the minimum of the non-zero values to avoid `ZeroDivision error` in `np.log()`
    ClTT2d[ClTT2d == 0] = np.min(ClTT2d[ClTT2d != 0])
    im_1 = ax.imshow(np.log(ClTT2d), vmin=None, vmax=None,
                     interpolation='bilinear', origin='lower', cmap=cmap.RdBu_r)
    im_1.set_extent([0, X_width, 0, Y_width])
    # Create an axis on the right side of `axes`. The width of `cax` will be 5%
    # of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(mappable=im_1, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.set_label('Log-temperature [$\mu$K]', fontsize=axiscbarfontsize+8, rotation=90, labelpad=19)
    
    ### PLOT 2.
    ax = axes[1]
    im_2 = ax.imshow(CMB_2D, vmin=0, vmax=np.max(np.conj(FT_2d) * FT_2d * ell2d * (ell2d + 1) / (2 * np.pi)).real,
                     interpolation='bilinear', origin='lower', cmap=cmap.RdBu_r)
    im_2.set_extent([ell2d.min(), ell2d.max(), ell2d.min(), ell2d.max()])
    ax.set_xlim(14000, 16500)
    ax.set_ylim(14000, 16500)
    ax.tick_params(axis='x', which='major', labelsize=axisticksize, rotation=42)
    # Create an axis on the right side of `axes`. The width of `cax` will be 5%
    # of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(mappable=im_2, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.ax.yaxis.get_offset_text().set(size=axiscbarfontsize)
    
    for i in range(ncols):
        ax = axes[i]
        
        ax.set_title(titles[i], fontsize=axistitlesize, fontweight='bold')
        ax.set_xlabel(labels[i], fontsize=axislabelsize, fontweight='bold')
        ax.set_ylabel(labels[i], fontsize=axislabelsize, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=axisticksize)
    
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
    
#    print FMap
    PSMap = np.fft.fftshift(np.real(np.conj(FMap) * FMap2))
 #   print PSMap
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
