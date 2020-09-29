# Parameters to set up the size of the maps
N = 2**10                # Number of pixels in a linear dimension.
                         # Since we are using lots of FFTs this should be a factor of 2^N
pix_size  = 0.5          # Size of a pixel in arcminutes

# Parameters to set up the map plots
c_min = -400             # Minimum for color bar
c_max = 400              # Maximum for color bar
X_width = N*pix_size/60  # Horizontal map width in degrees
Y_width = N*pix_size/60  # Vertical map width in degrees

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
