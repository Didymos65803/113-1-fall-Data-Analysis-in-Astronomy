# %% [markdown]
# # Data analysis in Astronomy 
# 
# ### Homework 3 due 11/18 11:59 pm
# 
# ### Name: Cheng-An, Hsieh

# %%
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as ss

# %% [markdown]
# ### 1. A few weeks ago, you have learned how to do simple cross-correlation measurements. 
# ### Now extend your code to calculate mean($\delta y$) (mean(y_x)-mean(y_0)) when x = 1, 2, 3, 4. Estimate the uncertainty with bootstrap. 
# ### To do: 
# ### a. Estimate the uncertainty with the error of the mean and propagate the uncertainty. Compare the estimated uncertainty based on the error propagation and bootstrapping. (5 points) 
# ### b. Produce a plot with xaxis showing x=[1,2,3,4], and with yaxis showing the corresponding delta y values. (Please include the uncertainty) (5 points)
# 
# you can find the data at https://www.dropbox.com/s/ak159coklf1eplf/simple_correlation_10000_x_y_01234.npy?dl=0
# ### c. Do the linear regression with a model, delta y= a x, and get the best fit a parameter value. Plot your best-fit model in the figure. (15 points) 

# %% [markdown]
# ### -----------------------------------------------------------------------------------------------------------------

# %%
### a. Estimate the uncertainty with the error of the mean and propagate the uncertainty. 
# Compare the estimated uncertainty based on the error propagation and bootstrapping. (5 points) 

def bootstrap(x, y, n  = 1000):
    
    errors = []
    n_bootstrap = n
    if len(x) <= len(y):
        n_point = len(x)
    else:
        n_point = len(y)
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_point, n_point)
        sample_x = x[indices]
        sample_y = y[indices]
        mean_diff = np.mean(sample_y) - np.mean(sample_x)
        errors.append(mean_diff)
    bootstrap_error = np.std(errors)
    return bootstrap_error

# Load the data
data = np.array(np.load('simple_correlation_10000_x_y_01234.npy'))

x = data[0]
y = data[1] 

# Calculate mean(Δy) for x = 1, 2, 3, 4 and estimate uncertainty with bootstrap
mean_dy_values = []
bootstrap_errors = []

for i in range(1, 5):
    y_x = y[x == i]
    y_0 = y[x == 0]

    if len(y_x) > 0 and len(y_0) > 0:
        mean_y_x = np.mean(y_x)
        mean_y_0 = np.mean(y_0)
        
        mean_dy = mean_y_x - mean_y_0
        mean_dy_values.append(mean_dy)
        
        # Bootstrap uncertainty for Δy
        bootstrap_error_dy = bootstrap(y_x, y_0)
        bootstrap_errors.append(bootstrap_error_dy)

        # Print mean(Δy) and bootstrap uncertainty
        print(f"x = {i}")
        print(f'Δy = ', mean_dy)
        print(f'Bootstrap uncertainty = ', bootstrap_error_dy)



# %%


np.array(mean_dy_values)
np.array(bootstrap_errors)
x_values = [1, 2, 3, 4]
np.array(x_values)

print(type(mean_dy_values), type(bootstrap_errors))


plt.figure(figsize=(10, 6))

plt.errorbar(
    x_values,
    mean_dy_values,
    yerr=bootstrap_errors,
    fmt='o',
    color='blue',
    ecolor='red',
    capsize=5,
    label='Mean Δy'
)

plt.xlabel('x')
plt.ylabel('Δy')
plt.title('Mean(Δy) vs. x')
plt.grid(True)
plt.legend()
plt.show()


# %%
### c. Do the linear regression with a model, delta y= a x, and get the best fit a parameter value.
#  Plot your best-fit model in the figure. (15 points) 

# Linear regression
slope, intercept, r_value, p_value, std_err = ss.linregress(x_values, mean_dy_values)

plt.figure(figsize=(10, 6))

plt.errorbar(
    x_values,
    mean_dy_values,
    yerr=bootstrap_errors,
    fmt='o',
    color='blue',
    ecolor='red',
    capsize=5,
    label='Mean Δy'
)

plt.plot(
    x,
    slope * x + intercept,
    color='green',
    label='Linear fit'
)

plt.text(0, 4, f'a = {slope:.2f}, r = {r_value:.2f}', fontsize=12, color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear regression')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ### 2. Finish fitting the black body spectrum
# 
# 
# ### To do:
# 
# ### Finish writing the blackbody model and get the best-fit temperature and uncertainty of the spectrum. (10 points) 

# %%
# https://www.dropbox.com/s/c8qomgjpbvlj87o/hubble_original_data.fits?dl=0

# %%
black_body = pf.getdata('spec-3744-55209-0386.fits',1)
# https://www.dropbox.com/s/n2xrychdkulsqjp/spec-3744-55209-0386.fits?dl=0

# %%
search = np.where(black_body['ivar']>0)

# %%
flux = black_body['flux'][search[0]].astype('float64')
wavelength = 10**black_body['loglam'][search[0]].astype('float64')
ivar = black_body['ivar'][search[0]].astype('float64')

# %%
plt.figure(figsize=(12,8))
plt.plot(wavelength,flux)
plt.plot(wavelength,1./np.sqrt(ivar))
plt.xlabel('Wavelength [$\\rm \\AA$]',fontsize=20)
plt.ylabel('Flux',fontsize=20)

# %%
def blackbody_SED(x, T, A):
    h, c, k = 6.626e-34, 3.0e8, 1.38e-23  # Planck constant, speed of light, Boltzmann constant
    temperature = T
    a = A
    f_lambda =   a * (2.0 * h * c**2) / (x**5 * (np.exp(h * c / (x * k * temperature)) - 1.0))
    # Write the content 
    # parameter [amplitude and temperature]
    # Let's just get the temperature right
    return f_lambda

# %%
def residual_function(p,data):
    x,y,y_ivar = data
    return np.sqrt(y_ivar)*(y-blackbody_SED(p,x))

wavelength_SI = wavelength * 1e-10
print(wavelength)


# %%
from scipy.optimize import curve_fit

paramsinitial =  [5000, 1]

# Perform the fit
popt, conv= curve_fit(blackbody_SED, wavelength_SI, flux, p0=paramsinitial, maxfev=20000, sigma=1./np.sqrt(ivar), absolute_sigma=True)

print(f"temperature:{popt[0]}K, amplitude:{popt[1]}")

plt.figure(figsize=(12,8))
plt.plot(wavelength,flux)
plt.plot(wavelength,blackbody_SED(wavelength_SI,popt[0],popt[1]))
plt.xlabel('Wavelength [$\\rm \\AA$]',fontsize=20)
plt.ylabel('Flux',fontsize=20)

# %% [markdown]
# ### -----------------------------------------------------------------

# %% [markdown]
# ### 3. Measuring the rest equavalent width
# 
# ### The rest equivalent width (the area of flux absorbed by the gas) of an absorption line contains physical information about the abundance of the gas. Astronomers frequently need to measure the rest equivalent width of absorption lines. 
# 
# ### We now have 6 absorption line spectra. The two absorption lines are MgII 2796 and 2803. Their exact wavelengths are 2796.35 and 2803.53 Angstrom. 
# 
# ### The goal is to obtain the rest equivalent widths of the two absorption lines for the 6 spectra. 
# 
# ### To do:
# 
# ### 1. Write a function consisting of two gaussians with central wavelengths at 2796.35 and 2803.53 Angstrom and obtain the best fit parameters of the gaussians. Based on the best fit parameters, you can get the rest equivalent widths of the absorption lines and the uncertainty of the rest equivalent widths. (You need to input the error array as well when doing the fitting.) (15 points) 
# 
# ### 2. Plot the best fit models with the absorption line spectra together to demonstrate that what you get is truly the best fit models. (Similar to Figure 11 in https://ui.adsabs.harvard.edu/abs/2017ApJ...850..156L/abstract) (15 points) 
# 
# You can find the data at https://www.dropbox.com/s/bp63yptckslvzd5/Simple_absorption_line_spectra.fits?dl=0
# 
# The data structure can be found below. 

# %%
data = pf.open('Simple_absorption_line_spectra.fits')

# %%
spectra = data[0].data
error = data[1].data
wavelength = data[2].data

# %%
plt.figure(figsize=(15,10))
for i in range(0,len(spectra[:,0])):
    plt.subplot(2,3,i+1)
    plt.plot(wavelength,spectra[i,:])

# %%
def double_gaussian_continuum(wavelength, continuum, A1, mu1, sigma1, A2, mu2, sigma2):
    
    gaussian1 = A1 * np.exp(-0.5 * ((wavelength - mu1) / sigma1)**2)
    gaussian2 = A2 * np.exp(-0.5 * ((wavelength - mu2) / sigma2)**2)
    return continuum + gaussian1 + gaussian2

def fit_spectrum(wavelength, flux, flux_error):
   
    # Initial guesses
    continuum_init = 1.0
    A1_init = -(1 - min(flux))
    mu1_init = 2796.35
    sigma1_init = 1.0
    A2_init = -(1 - min(flux))
    mu2_init = 2803.53
    sigma2_init = 1.0

    initial_guess = [continuum_init, A1_init, mu1_init, sigma1_init, A2_init, mu2_init, sigma2_init]

    # Define bounds to ensure realistic fitting
    bounds_lower = [0.9, -np.inf, 2790, 0.5, -np.inf, 2800, 0.5]
    bounds_upper = [1.1, 0, 2800, 3.0, 0, 2810, 3.0]

    try:
        popt, pcov = curve_fit(
            double_gaussian_continuum,
            wavelength,
            flux,
            p0=initial_guess,
            sigma=flux_error,
            absolute_sigma=True,
            bounds=(bounds_lower, bounds_upper)
        )

        perr = np.sqrt(np.diag(pcov))

    except RuntimeError:
        print("Error - curve_fit failed")
        popt = [np.nan] * 7
        perr = [np.nan] * 7

    # Calculate fitted flux
    fitted_flux = double_gaussian_continuum(wavelength, *popt)

    # Extract parameters
    continuum, A1, mu1, sigma1, A2, mu2, sigma2 = popt
    continuum_err, A1_err, mu1_err, sigma1_err, A2_err, mu2_err, sigma2_err = perr

    # Calculate Equivalent Widths (EW)
    # EW = abs(A) * sigma * sqrt(2 * pi)
    ew1 = np.abs(A1) * sigma1 * np.sqrt(2 * np.pi)
    ew2 = np.abs(A2) * sigma2 * np.sqrt(2 * np.pi)

    # Propagate uncertainties
    if not np.isnan(A1_err) and not np.isnan(sigma1_err):
        ew1_err = np.sqrt( (sigma1 * np.sqrt(2 * np.pi) * A1_err)**2 +
                           (A1 * np.sqrt(2 * np.pi) * sigma1_err)**2 )
    else:
        ew1_err = np.nan
        print("nan")

    if not np.isnan(A2_err) and not np.isnan(sigma2_err):
        ew2_err = np.sqrt( (sigma2 * np.sqrt(2 * np.pi) * A2_err)**2 +
                           (A2 * np.sqrt(2 * np.pi) * sigma2_err)**2 )
    else:
        ew2_err = np.nan
        print("nan")

    ew_rest = {
        'MgII_2796_EW': (ew1, ew1_err),
        'MgII_2803_EW': (ew2, ew2_err)
    }

    return popt, perr, fitted_flux, ew_rest


# %%
# Convert lists to NumPy arrays
spectra = np.array(spectra)  # Shape: (6, 46)
error = np.array(error)      # Shape: (6, 46)

# Initialize lists to store results
all_fit_params = []
all_fit_errors = []
all_fitted_flux = []
all_ew_rest = []

# Loop through each spectrum
for i in range(spectra.shape[0]):
    print(f"Spectrum {i+1}/{spectra.shape[0]}")

    # Extract the current spectrum and its error
    current_flux = spectra[i]
    current_error = error[i]

    # Fit the absorption lines
    popt, perr, fitted_flux, ew_rest = fit_spectrum(wavelength, current_flux, current_error)

    # Store the results
    all_fit_params.append(popt)
    all_fit_errors.append(perr)
    all_fitted_flux.append(fitted_flux)
    all_ew_rest.append(ew_rest)

### 2. Plot the best fit models with the absorption line spectra together to demonstrate that what you get is truly the best fit models. 
# (Similar to Figure 11 in https://ui.adsabs.harvard.edu/abs/2017ApJ...850..156L/abstract) (15 points) 

    # Plot the observed spectrum and the best-fit model
    plt.figure(figsize=(10, 5))
    plt.errorbar(wavelength, current_flux, yerr=current_error, fmt='k.', markersize=4, label='Observed Spectrum')
    plt.plot(wavelength, fitted_flux, 'r-', label='Best-Fit Model')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Flux')
    plt.title(f'Spectrum {i+1}: MgII Absorption Lines Fit')
    plt.legend()

    # Highlight the fitted regions
    plt.axvline(2796.35, color='blue', linestyle='--', alpha=0.7)
    plt.axvline(2803.53, color='green', linestyle='--', alpha=0.7)
    plt.text(2796.35 + 0.3, 1.01, 'MgII 2796', color='blue')
    plt.text(2803.53 + 0.3, 1.01, 'MgII 2803', color='green')

    plt.tight_layout()
    plt.show()

    # Print the fit parameters and EWs
    continuum, A1, mu1, sigma1, A2, mu2, sigma2 = popt
    continuum_err, A1_err, mu1_err, sigma1_err, A2_err, mu2_err, sigma2_err = perr

    print("Best-Fit Parameters:")
    print(f"  MgII 2796 Amplitude (A1): {A1:.4f} ± {A1_err:.4f}")
    print(f"  MgII 2796 Mean (mu1): {mu1:.4f} Å ± {mu1_err:.4f} Å")
    print(f"  MgII 2796 Std (sigma1): {sigma1:.4f} Å ± {sigma1_err:.4f} Å")
    print(f"  MgII 2803 Amplitude (A2): {A2:.4f} ± {A2_err:.4f}")
    print(f"  MgII 2803 Mean (mu2): {mu2:.4f} Å ± {mu2_err:.4f} Å")
    print(f"  MgII 2803 Stddev (sigma2): {sigma2:.4f} Å ± {sigma2_err:.4f} Å")

    print("\nRest Equivalent Widths:")
    for key, (ew, ew_err) in ew_rest.items():
        if not np.isnan(ew_err):
            print(f"  {key}: {ew:.4f} Å ± {ew_err:.4f} Å\n")
        else:
            print(f"  {key}: {ew:.4f} Å")




# %% [markdown]
# ### -----------------------------------------------------------------------------------------------------------------

# %% [markdown]
# ### 4. Decomposing QSO spectra with NMF
# ### In 11/5, we use PCA to decompose the SDSS QSO spectra. Now, we use another technique, called none-negative matrix factoriation (NMF), to do so. [Use the median-filter smooth spectra]
# 
# ### TO DO:
# 
# ### 1. Do the the NMF and explore the outputs! (10 points)
# 
# Follow the code above!
# 
# 1. Data structure matrix [i_object, i_feature]
# 
# 
# 2. nmf = NMF(n_components=6)
#    
#    nmf.fit(matrix)
#    
#    
# 3. plot wavelength, nmf.components_[i] i=0,1,2,3,4,5,6 (please plot each component separately)
# 
# ### 2. Please observe the NMF "eigenspectra" and describe what spectral features are captured by each component. (10 points)
# 
# ### 3. Please use the NMF eigenspectra to reconstruct the observed quasar spectra and plot the first 10 observed quasar spectra (i_object<10) and the reconstructed spectra. (15 points)
# 
# 

# %%
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

data = pf.open('clean_spectra.fits')
wavelength = 10**data[2].data

import scipy.ndimage as sn
smooth_spectra = np.zeros((len(data[0].data),len(data[0].data[0])))
for i_object in range(0,len(data[0].data)):
    smooth_spectra[i_object] = sn.median_filter(data[0].data[i_object],10)




# %%
Data_matrix = np.maximum(smooth_spectra, 0)  # Ensure non-negative values
nmf = NMF(n_components=6)
nmf.fit(Data_matrix)

plt.figure(figsize=(15,8))
plt.plot(wavelength, nmf.components_[0], color='black')
plt.xlabel('Wavelength [$\\rm \\AA$]', fontsize=20)
plt.ylabel('Flux []', fontsize=20)
plt.show()

print(len(nmf.components_))
for i in range(len(nmf.components_)):
   plt.plot(wavelength,nmf.components_[i],alpha=0.5)
   plt.xlabel('Wavelength [$\\rm \\AA$]',fontsize=20)
   plt.ylabel('Flux []',fontsize=20)
   plt.show()

# %% [markdown]
# 
# ### 2. Please observe the NMF "eigenspectra" and describe what spectral features are captured by each component. (10 points)
# 
# - the first eigenspectra show the significant two peaks at ~1900A and ~2800A.
# - the second eigenspectra show a small valley at 2800A.
# - the third eigenspectra show a small valley at 1900A and large valley at 2800A, also have a increase trend.
# - the fourth eigenspectra show only significant peak at ~2800A, it means some spectra only have one peak.
# - the fifth eigenspectra show also two peak, but the other part have different trend with mean spectra.
# - the sixth spectra show two significant valleys.
# 
# 
# 
# 

# %%

### 3. Please use the NMF eigenspectra to reconstruct the observed quasar spectra and plot the first 10 observed quasar spectra (i_object<10) and the reconstructed spectra. (15 points)

# Reconstruct the spectra using the NMF components
W = nmf.transform(Data_matrix)
H = nmf.components_
reconstructed_spectra = np.dot(W, H)

# Plot the first 10 observed and reconstructed spectra
plt.figure(figsize=(10, 30))
for i in range(10):
    plt.subplot(10, 1, i + 1)
    plt.plot(wavelength, Data_matrix[i], label='Observed Spectrum')
    plt.plot(wavelength, reconstructed_spectra[i], label='Reconstructed Spectrum', linestyle='--')
    plt.xlabel('Wavelength [$\\rm \\AA$]')
    plt.ylabel('Flux')
    plt.legend()
    plt.title(f'Spectrum {i + 1}')
plt.tight_layout()
plt.show()



