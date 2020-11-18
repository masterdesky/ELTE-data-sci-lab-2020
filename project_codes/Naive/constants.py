# Variables to set up the size of the map
## Number of pixels in the longer linear dimension.
## Since we are using lots of FFTs this should be a factor of 2^N
N = 2**10
c_min = -400     # Minimum for color bar [micro K]
c_max = 400      # Maximum for color bar [micro K]
X_width = 8      # Horizontal map width  [degrees]
Y_width = 8      # Vertical map width    [degrees]

N_x = N if X_width < Y_width else int(N * (X_width/Y_width)) # Pixel number along X-axis
N_y = N if X_width > Y_width else int(N * (Y_width/X_width)) # Pixel number along Y-axis
pix_size  = (X_width/N_x) * 60                            # Size of a pixel  [arcminutes]

# Paramaters to set up point sources
number_of_sources = 5000              # Numer of point sources with Poisson distribution
amplitude_of_sources = 200            # Lambda param. for point sources with Poisson distribution       [micro K]
number_of_sources_EX = 50             # Number of point sources with exponential distribution
amplitude_of_sources_EX = 1000        # Lambda param. for point sources with exponential distribution   [micro K]

# Parameters to set up Sunyaev-Zeldovich sources
number_of_SZ_clusters = 500           # Number of SZ sources
mean_amplitude_of_SZ_clusters = 50    # 
SZ_beta = 0.86                        # 
SZ_theta_core = 1.0                   # 

beam_size_fwhp = 1.25                 # FWHM of the simulated beam

# Parameters to set up noise level
white_noise_level = 10                # White noise [micro K/arcmin]
atmospheric_noise_level = 0.1         # Multiply by zero to turn this off
one_over_f_noise_level = 0.2          # Multiply by zero to turn this off