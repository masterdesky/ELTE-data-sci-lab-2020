# Variables to set up the size and format of the maps
## Number of pixels in the longer linear dimension.
## Since we are using lots of FFTs this should be a factor of 2^N
N = 2**10
c_min = -750     # Minimum for color bar
c_max = 750      # Maximum for color bar
X_width = 8      # Horizontal map width in degrees
Y_width = 8      # Vertical map width in degrees

N_x = N if X_width > Y_width else N // (Y_width//X_width) # Pixel number along X-axis
N_y = N if X_width < Y_width else N // (X_width//Y_width) # Pixel number along Y-axis
pix_size  = (X_width/N_x) * 60                            # Size of a pixel in arcminutes

# Paramaters to set up the SZ point sources
number_of_sources = 5000
amplitude_of_sources = 200
number_of_sources_EX = 50
amplitude_of_sources_EX = 1000
number_of_SZ_clusters = 500
mean_amplitude_of_SZ_clusters = 50
SZ_beta = 0.86
SZ_theta_core = 1.0

beam_size_fwhp = 1.25

N_iterations = 16

white_noise_level = 10
atmospheric_noise_level = 0.5 * 0
one_over_f_noise_level = 0

#### parameters for setting up the spectrum
delta_ell = 50
ell_max = 5000
