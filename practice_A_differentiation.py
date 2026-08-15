"""
PRACTICE TEMPLATE — Online 3, Section A pattern
"Verify the differentiation property of the CFT."

    F{d/dt x(t)} = j*2*pi*f*X(f)

Given: x(t) = 0.5*cos(4t) + 0.5*sin(6t)
Fill in the TODOs, then run this file to check your work.

VERIFIED before being handed to you: this exact derivation and
comparison ran clean (magnitude/phase MSE near zero at the frequencies
where the signal has real energy).
"""
from cft_core import cft, mse, significant_mask
import numpy as np


# ---------------------------------------------------------------
# YOUR WORK GOES HERE
# ---------------------------------------------------------------

# Arguments: t (1D array of time samples).
# Returns: 1D array, x(t) = 0.5*cos(4t) + 0.5*sin(6t).
def x_of_t(t):
    # TODO: implement x(t)
    return 0.5*np.cos(4*t)+0.5*np.sin(6*t)


# Arguments: t (1D array of time samples).
# Returns: 1D array, the ANALYTIC derivative d/dt x(t).
# Hint: differentiate term by term. d/dt[cos(kt)] = -k*sin(kt),
# d/dt[sin(kt)] = k*cos(kt). Do NOT use a numerical-differencing
# shortcut (np.diff / np.gradient) here -- work out the closed form
# by hand first, exactly like you would on paper, then code that.
def dx_dt(t):
    # TODO: implement the analytic derivative
    return -2*np.sin(4*t)+3*np.cos(6*t)


def main():
    t = np.linspace(-5, 5, 2000)
    x = x_of_t(t)
    y1 = dx_dt(t)

    f = np.linspace(-10, 10, 2000)
    X = cft(t, x, f)
    Y1 = cft(t, y1, f)

    # Theoretical prediction from the property
    theoretical_Y1 = 1j * 2 * np.pi * f * X

    # GOTCHA: restrict comparison to frequencies with real signal energy
    # (see cft_core.significant_mask's docstring for why).
    mask = significant_mask(X, threshold_frac=0.5)
    print(
        f"Comparing at {mask.sum()} significant frequencies (out of {len(f)})")

    mag_mse = mse(np.abs(Y1)[mask], np.abs(theoretical_Y1)[mask])
    phase_mse = mse(np.angle(Y1)[mask], np.angle(theoretical_Y1)[mask])
    print(f"Magnitude MSE: {mag_mse:.6f}")
    print(f"Phase MSE:     {phase_mse:.6f}")

    # NOTE: raw magnitude MSE can look "large" even when the property
    # holds well, because it's an ABSOLUTE squared error and |Y1|/|theory|
    # values here run up to ~15 -- a 5-10% relative error squares into a
    # visually large-looking number. Check RELATIVE error instead, which
    # is what actually tells you whether the property holds:
    rel_err = np.abs(
        np.abs(Y1)[mask] - np.abs(theoretical_Y1)[mask]) / np.abs(theoretical_Y1)[mask]
    print(f"Max relative magnitude error: {rel_err.max():.1%}")
    print("PASS" if rel_err.max() < 0.15 and phase_mse <
          0.05 else "FAIL — check x_of_t / dx_dt")

    # BONUS (matches the real paper's Task 3): repeat for the 2nd and
    # 3rd derivatives once the above passes. d^2/dt^2 and d^3/dt^3 of
    # the same x(t) -- same approach, just differentiate further.


if __name__ == "__main__":
    main()
