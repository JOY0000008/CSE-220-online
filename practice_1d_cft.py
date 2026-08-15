"""
PRACTICE TEMPLATE — CFT property verification, YOU build everything.
"""
import numpy as np


def cft(t, x, f):
    ft = np.outer(f, t)
    real_part = np.trapezoid(x[None, :] * np.cos(2 * np.pi * ft), t, axis=-1)
    imag_part = -np.trapezoid(x[None, :] * np.sin(2 * np.pi * ft), t, axis=-1)
    return real_part + 1j * imag_part


def mse(a, b):
    return np.mean((np.abs(a) - np.abs(b)) ** 2)


def significant_mask(X, threshold_frac=0.1):
    return np.abs(X) > threshold_frac * np.abs(X).max()


def x_t(t):
    return np.exp(-t**2)


# =====================================================================
# Q12 — Time-shift property
# =====================================================================
def verify_time_shift(t, x, f, X_base, t0):
    # IMPORTANT NOTE: Time shifting by t0 means replacing 't' with 't - t0'.
    # WAY 1: Put x(t-t0) directly into the CFT equation
    Y_direct = cft(t, x_t(t - t0), f)

    # IMPORTANT NOTE: A delay in time domain multiplies the frequency domain by e^{-j*2*pi*f*t0}
    # WAY 2: Use the property formula (e^{-j*2*pi*f*t0} * X)
    Y_formula = X_base * np.exp(-1j * 2 * np.pi * f * t0)

    return Y_direct, Y_formula


# =====================================================================
# Q13 — Differentiation property
# =====================================================================
def verify_differentiation(t, x, f, X_base):
    # IMPORTANT NOTE: The analytic derivative of x(t) = exp(-t^2) is d/dt = -2t * exp(-t^2)
    # You must calculate this derivative by hand first, then pass it to the CFT.
    # WAY 1: Put the analytic derivative of x(t) directly into the CFT equation
    derivative_x = -2 * t * x_t(t)
    Y_direct = cft(t, derivative_x, f)

    # IMPORTANT NOTE: Differentiating in time multiplies the frequency domain by j*2*pi*f
    # WAY 2: Use the property formula (j*2*pi*f * X)
    Y_formula = 1j * 2 * np.pi * f * X_base

    return Y_direct, Y_formula


# =====================================================================
# Q14 — Frequency-shift / modulation
# =====================================================================
def verify_modulation(t, x, f, f0):
    # IMPORTANT NOTE: Multiplying by a complex exponential in time shifts the signal in frequency.
    # WAY 1: Put [x(t) * e^{j*2*pi*f0*t}] directly into the CFT equation
    modulated_x = x * np.exp(1j * 2 * np.pi * f0 * t)
    Y_direct = cft(t, modulated_x, f)

    # IMPORTANT NOTE: To shift in frequency, we just evaluate the CFT at (f - f0).
    # Since we can't easily shift a discrete array without interpolation, we call cft()
    # again using the shifted frequency array.
    # WAY 2: Use the property formula X(f - f0) by calling cft with shifted frequencies
    Y_formula = cft(t, x, f - f0)

    return Y_direct, Y_formula


# =====================================================================
# Q16 — Time-scaling
# =====================================================================
def verify_time_scaling(t, x, f, a):
    # IMPORTANT NOTE: Scaling time by 'a' compresses or expands the signal.
    # WAY 1: Put x(a*t) directly into the CFT equation
    scaled_x = x_t(a * t)
    Y_direct = cft(t, scaled_x, f)

    # IMPORTANT NOTE: Time scaling shrinks the amplitude by 1/|a| and scales the frequency by f/a.
    # WAY 2: Use the property formula (1/|a|) * X(f/a) by calling cft with scaled frequencies
    Y_formula = (1 / np.abs(a)) * cft(t, x, f / a)

    return Y_direct, Y_formula


# =====================================================================
# Self-check driver — main does the testing
# =====================================================================
def main():
    t = np.linspace(-5, 5, 2000)
    x = x_t(t)
    f = np.linspace(-10, 10, 1000)

    # Baseline X for the formula side
    X_base = cft(t, x, f)
    mask = significant_mask(X_base, 0.1)

    print("=== Q12: time-shift ===")
    Y_direct, Y_formula = verify_time_shift(t, x, f, X_base, t0=1.0)
    print("mag MSE (Direct vs Formula):", mse(Y_direct[mask], Y_formula[mask]))

    print("\n=== Q13: differentiation ===")
    Y_direct, Y_formula = verify_differentiation(t, x, f, X_base)
    print("mag MSE (Direct vs Formula):", mse(Y_direct[mask], Y_formula[mask]))

    print("\n=== Q14: modulation ===")
    Y_direct, Y_formula = verify_modulation(t, x, f, f0=3.0)
    print("mag MSE (Direct vs Formula):", mse(Y_direct[mask], Y_formula[mask]))

    print("\n=== Q16: time-scaling ===")
    Y_direct, Y_formula = verify_time_scaling(t, x, f, a=2.0)
    # Applying mask might drop scaled frequencies awkwardly, so testing whole array
    print("mag MSE (Direct vs Formula):", mse(Y_direct, Y_formula))


if __name__ == "__main__":
    main()
