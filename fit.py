"""

    This script fits a number of individual conformer
    spectra to an experimental spectrum.

    Carlos Outeiral
    April 2019

"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# Hyperparameters
W_COEF  = 1.0E08   # Coefficient of weight penalty
RS_COEF = 1.0E02   # Coefficient of rescaling penalty
R_EQ    = 0.05     # Equilibrium frequency rescaling
S_COEF  = 5.0E08   # Coefficient of shift penalty
S_UPPER = 200.0    # Upper bound of shift penalty
S_LOWER = 0.0      # Lower bound of shift penalty

# Random seed (for reproducibility)
np.random.seed(42)


def read_files(symbol):
    """Reads the data from the CSV files. The files
    must be named as:

        - {symbol}_%d.csv for the individual runs
        - {symbol}_spectrum.csv for the experiment

    Each of these files must have one column, named 'WN'
    with the wavenumber, and a column 'I' with the intensity.

    This subroutine returns a pandas DataFrame with the
    experimental data, and a list of DataFrames with the
    individual runs."""

    exp = pd.read_csv(f'{symbol}_spectrum.csv')
    calc = [pd.read_csv(x) for x in os.listdir()
            if x.startswith(symbol) and x.endswith('.csv')
            and x != f'{symbol}_spectrum.csv']

    return exp, calc

def interpolate(exp, calc):
    """Interpolates the spectral data using cubic splines.
    This is necessary since the wavenumbers of different
    spectra need not coincide. Returns a function for the
    experimental data and a list of functions for the
    individual runs."""

    n_runs = len(calc)

    # Normalise spectra
    exp['I'] /= np.sum(exp['I'])
    for i in range(n_runs):
        calc[i]['I'] /= np.sum(calc[i]['I'])
    
    exp_wn  = np.array(exp['WN'], dtype=np.float64)
    exp_in  = np.array(exp['I'], dtype=np.float64)
    calc_wn = [np.array(x['WN'], dtype=np.float64) for x in calc]
    calc_in = [np.array(x['I'], dtype=np.float64) for x in calc]
    
    # Fit using cubic splines
    exp_fit  = interp1d(exp_wn, exp_in, kind='cubic', 
                        bounds_error=False, fill_value=0.0)
    calc_fit = [interp1d(calc_wn[t], calc_in[t], kind='cubic', 
                         bounds_error=False, fill_value=0.0) 
                for t in range(n_runs)]

    return exp_fit, calc_fit

def calculate_bounds(exp):
    """Computes the minimum and maximum wavenumber of the
    experimental spectrum."""
    return exp.iloc[0]['WN'], exp.iloc[-1]['WN']

def define_training(exp_fit, lower_bound, upper_bound):
    """Defines the reference experimental spectrum."""

    x_train = np.linspace(lower_bound, upper_bound, 5000)
    y_train = exp_fit(x_train)

    return x_train, y_train / np.max(y_train)

def define_spectrum(calc_fit, x_train):
    """Defines a function that returns a spectrum based on a
    vector of parameters. The first element of the vector is a
    general frequence shift; the (2n+1)th element is a frequency
    rescaling, and the (2n+2)th element is a weight."""

    n_runs = len(calc_fit)

    def spectrum(vec):
        y_pred = np.zeros(5000)
        for i in range(n_runs):
            y_pred += vec[2*i+1]**2 * calc_fit[i](vec[2*i+2] * x_train + vec[0])
        return y_pred / np.max(y_pred)

    return spectrum

def define_loss(y_train, spectrum, n_runs):
    """Defines a loss function to be used for optimisation.
    This function will include constraints to ensure physical
    interpretability, for example that the square of the weights
    adds up to 1.0."""

    def loss_function(vec):

        y_attempt = spectrum(vec)
        loss = np.sum([(y_attempt[i]-y_train[i])**2 
                       for i in range(5000)])

        # Weight penalty (\sum w_i^2 - 1.0)**2
        weight_penalty = 0.0
        for i in range(n_runs):
            weight_penalty += vec[2*i+1]**2
        weight_penalty = W_COEF * (weight_penalty-1.0)**2
        loss += weight_penalty

        # Rescaling penalty \sum (|r_i-1.0|-0.05)^2
        rescaling_penalty = 0.0
        for i in range(n_runs):
            rescaling_penalty += (abs(vec[2*(i+1)]-1)-R_EQ)**2
        rescaling_penalty = RS_COEF * rescaling_penalty
        loss += rescaling_penalty

        # Shift penalty
        shift_penalty = 0.0
        if vec[0] < S_LOWER or vec[0] > S_UPPER:
            shift_penalty += S_COEF 
        loss += shift_penalty

        return loss

    return loss_function

def optimise(loss, n_runs):
    """Finds a minimum of the loss function using the
    BFGS method. The run is initialised with several
    random starting parameters to attempt to find the
    best result."""

    opt_runs = []

    for _ in range(5):

        # Weights (must add up to 1)
        w = np.random.random(n_runs)
        w /= np.linalg.norm(w)

        # Rescaling factor (preferably 0.95<r<1.05)
        r = np.random.random(n_runs)
        r = 0.95 + r/10

        # Shift factor
        s = 50.0 + 25.0 * np.random.randn()

        in_arg = [s]
        for weight, rescaling in zip(w, r):
            in_arg.append(weight)
            in_arg.append(rescaling)
        in_arg = np.array(in_arg)
        
        opt = minimize(loss, in_arg, method='BFGS')
        opt_runs.append(opt)
        
    opt_runs = sorted(opt_runs, key=lambda x: x['fun'])
    return opt_runs[0]['x']

def optimisation(symbol):

    exp, calc = read_files(symbol)
    n_runs = len(calc)

    exp_fit, calc_fit = interpolate(exp, calc)
    lower_bound, upper_bound = calculate_bounds(exp)

    x_train, y_train  = define_training(exp_fit, lower_bound, upper_bound)
    spectrum = define_spectrum(calc_fit, x_train)
    loss = define_loss(y_train, spectrum, n_runs)

    opt_vec = optimise(loss, n_runs)

    return opt_vec, x_train, spectrum(opt_vec), y_train


if __name__ == '__main__':

    opt_vec, x_train, y_pred, y_train = optimisation('A')
    plt.plot(x_train, y_train, 'k-', label='Experimental')
    plt.plot(x_train, y_pred,  'r-', label='Predicted')
    plt.xlabel('Raman shift [$cm^{-1}$]')
    plt.ylabel('Intensity [a. u.]')
    plt.xlim(600, 1600)

    plt.show()
