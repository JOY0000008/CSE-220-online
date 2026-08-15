"""
Shared core for all three Online-3 (CFT) practice templates.

1D Continuous Fourier Transform via numerical integration -- no np.fft,
no complex-exponential shortcut (real/imaginary split by hand, same
constraint your real papers impose).

    X(f) = integral x(t) * exp(-j*2*pi*f*t) dt
         = integral x(t)*cos(2*pi*f*t) dt  -  j * integral x(t)*sin(2*pi*f*t) dt
"""
import numpy as np


def cft(t, x, f):
    """
    Arguments: t (1D array of time samples), x (1D array, x(t) at those
    times, real or complex), f (1D array of frequencies to evaluate at).
    Returns: complex array X(f), same length as f.
    """
    ft = np.outer(f, t)  # dim(f,t)
    # X(f)= integration x(t) * exp^(-j*2*pi*f*t)
    real_part = np.trapezoid(x[None, :]*np.cos(2*np.pi*ft), t, axis=-1)
    imag_part = -np.trapezoid(x[None, :]*np.sin(2*np.pi*ft), t, axis=-1)
    return real_part + 1j * imag_part


def mse(a, b):
    return np.mean((a - b) ** 2)


def significant_mask(X, threshold_frac=0.1):
    """
    IMPORTANT GOTCHA: phase (np.angle) is numerically meaningless wherever
    |X(f)| is near zero -- floating-point noise in a near-zero complex
    number produces huge, meaningless phase swings. Any magnitude/phase
    MSE comparison should restrict to frequencies where the signal
    actually has energy, or the MSE will look terrible even when your
    implementation is completely correct. This function returns a
    boolean mask for "significant" frequencies -- use it before computing
    MSE, on BOTH signals being compared.
    """
    return np.abs(X) > threshold_frac * np.abs(X).max()
