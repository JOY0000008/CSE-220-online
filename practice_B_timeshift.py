"""
PRACTICE TEMPLATE — Online 3, Section B pattern
"Verify the time-shift property of the CFT."

    y(t) = x(t - t0)   =>   Y(f) = X(f) * exp(-j*2*pi*f*t0)
    =>  |Y(f)| = |X(f)|
    =>  angle(Y(f)) = angle(X(f)) - 2*pi*f*t0

Given: x(t) = exp(-t^2)  (Gaussian, a=1), t0 = 1.
Fill in the TODOs, then run this file to check your work.

VERIFIED before being handed to you: this ran to floating-point-zero
precision (Gaussians have a smooth, non-sparse spectrum, so there's no
near-zero-magnitude phase instability to work around here).
"""
from cft_core import cft, mse, significant_mask
import numpy as np


# ---------------------------------------------------------------
# YOUR WORK GOES HERE
# ---------------------------------------------------------------

# Arguments: t (1D array), a (float, width parameter).
# Returns: 1D array, the Gaussian x(t) = exp(-a*t^2).
def gaussian(t, a=1.0):
    # TODO: implement the Gaussian
    return np.exp(-a*t**2)


# Arguments: t (1D array), a (float), t0 (float, the shift amount).
# Returns: 1D array, y(t) = x(t - t0) where x is the same Gaussian.
# Hint: this is direct formula substitution -- replace every t in the
# Gaussian formula with (t - t0). Since we have a closed form (not an
# array we're indexing into), this correctly represents "shift by t0"
# without ever touching array indices.
def shifted_gaussian(t, a=1.0, t0=1.0):
    # TODO: implement the shifted Gaussian
    new_t = t-t0
    return np.exp(-a*new_t**2)


def main():
    t = np.linspace(-5, 5, 2000)
    a = 1.0
    t0 = 1.0

    x = gaussian(t, a)
    y = shifted_gaussian(t, a, t0)

    f = np.linspace(-10, 10, 1000)
    X = cft(t, x, f)
    Y = cft(t, y, f)

    theoretical_Y = X * np.exp(-1j * 2 * np.pi * f * t0)

    mask = significant_mask(X, threshold_frac=0.1)
    print(
        f"Comparing at {mask.sum()} significant frequencies (out of {len(f)})")

    mag_mse = mse(np.abs(Y)[mask], np.abs(theoretical_Y)[mask])
    phase_mse = mse(np.angle(Y)[mask], np.angle(theoretical_Y)[mask])
    print(f"Magnitude MSE: {mag_mse:.6e}")
    print(f"Phase MSE:     {phase_mse:.6e}")
    print("PASS" if mag_mse < 1e-2 and phase_mse <
          1e-2 else "FAIL — check gaussian / shifted_gaussian")


if __name__ == "__main__":
    main()
