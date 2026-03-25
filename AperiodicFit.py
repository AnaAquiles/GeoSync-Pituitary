
import numpy as np
from scipy.optimize import curve_fit


"""

Aperiodic spectral analysis of calcium imaging fluorescence traces.

Fits a Lorentzian (1/f-like) model to the power spectrum of each cell to
extract the aperiodic exponent, and summarises cross-cell spectral
variability in log-power space.

    Assumed upstream dependency:
    datosNorm_exponential : np.ndarray, shape (n_cells, n_samples)
        Background-subtracted, exponentially normalised fluorescence traces
        from preprocessing.py.
"""


def lorentzian(f: np.ndarray, bias: float, alpha: float, k: float) -> np.ndarray:
    """Lorentzian (aperiodic) spectral model in log-power space.

    Models the 1/f-like aperiodic component of the power spectrum as:

        log P(f) = bias - log(k + f^alpha)

    Parameters
    ----------
    f : np.ndarray
        Frequency values (Hz).
    bias : float
        Intercept (log-power offset).
    alpha : float
        Aperiodic exponent; steeper slopes correspond to larger alpha.
    k : float
        Knee parameter; controls the low-frequency roll-off.

    Returns
    -------
    np.ndarray
        Predicted log-power at each frequency.
    """
    return bias - np.log(k + f ** alpha)


def fit_aperiodic_exponent(freqs: np.ndarray, power: np.ndarray) -> float:
    """Fit the Lorentzian model and return the aperiodic exponent.

    Uses non-linear least squares (``scipy.optimize.curve_fit``) with default
    initialisations.  The absolute value of the fitted exponent is returned so
    that steeper spectra always yield larger positive values.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis (Hz), shape (n_freqs,).
    power : np.ndarray
        Power spectrum corresponding to ``freqs``, shape (n_freqs,).

    Returns
    -------
    float
        Absolute aperiodic exponent alpha.
    """
    popt, _ = curve_fit(lorentzian, freqs, power)
    _bias, alpha, _k = popt
    return float(np.abs(alpha))


def log_power_variance(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-cell variance and mean in log10-power space.

    A small regularisation constant (1e-10) is added before taking the
    logarithm to avoid -inf values at bins with zero power.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_cells, n_freqs)
        Power spectra (raw or normalised), one row per cell.

    Returns
    -------
    variance : np.ndarray, shape (n_freqs,)
        Variance of log10-power across cells at each frequency bin.
    mean : np.ndarray, shape (n_freqs,)
        Mean log10-power across cells at each frequency bin.
    """
    log_spectra = np.log10(spectra + 1e-10)
    return np.var(log_spectra, axis=0), np.mean(log_spectra, axis=0)


if __name__ == "__main__":
    N_CELLS = datosNorm_exponential.shape[0]
    N_SAMPLES = 600   # samples passed to the FFT
    FS = 2            # sampling frequency (Hz)
    FREQ_THRESHOLD = 0.0005   # discard bins at or below this frequency (Hz)

    # --- Frequency axis (shared across all cells) ---
    freqs_trial = np.fft.rfftfreq(N_SAMPLES, d=1.0 / FS)
    mask = freqs_trial > FREQ_THRESHOLD
    freqs_all = freqs_trial[mask]

    # --- Power spectra (one per cell) ---
    power_all = np.zeros((N_CELLS, mask.sum()))

    for i in range(N_CELLS):
        power_full = np.abs(np.fft.rfft(datosNorm_exponential[i, :N_SAMPLES])) ** 2
        power_all[i] = power_full[mask]

    # --- Aperiodic exponent (Lorentzian fit, one per cell) ---
    aperiodic_exp = np.array([
        fit_aperiodic_exponent(freqs_all, power_all[i])
        for i in range(N_CELLS)
    ])

    # --- Log-power variance and mean across cells ---
    log_variance, log_mean = log_power_variance(power_all)
