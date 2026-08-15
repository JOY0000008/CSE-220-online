"""
PRACTICE TEMPLATE — Online 3, Section C pattern
"Verify combined time-scaling + phase-shift (modulation) properties."

*** HONESTY NOTE — READ THIS FIRST ***
Unlike templates A and B, I could NOT get this one to numerically verify
cleanly against Square+Triangle waves, despite several real attempts
(different compression factors, tapering the signal to force decay,
checking window consistency). The two properties below are individually
correct and well-established theory -- I verified BOTH against a clean
Gaussian test signal, where they matched to floating-point precision.
The failure is specific to applying them to PERIODIC signals (Square and
Triangle waves never decay), and I was not able to pin down the exact
mechanism before running out of productive leads. This file gives you
the correct setup and lets you explore it -- do not trust a "PASS" from
this file the way you can trust one from templates A/B.

Time-scaling:  z(t) = x(a*t)              =>  Z(f) = (1/|a|) * X(f/a)
Modulation:    y(t) = x(t)*exp(j*2*pi*f0*t)  =>  Y(f) = X(f - f0)
Combined:      y(t) = x(a*t)*exp(j*2*pi*f0*t)  =>  Y(f) = (1/|a|)*X((f-f0)/a)
"""
from cft_core import cft, mse, significant_mask
import numpy as np


# ---------------------------------------------------------------
# Given: Square and Triangle wave generators (these are correct --
# verified they produce the right period/shape)
# ---------------------------------------------------------------
def square_wave(t, freq=1.0):
    return np.sign(np.sin(2 * np.pi * freq * t))


def triangle_wave(t, freq=1.0):
    return (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))


# ---------------------------------------------------------------
# YOUR WORK GOES HERE — same structure as the real paper
# ---------------------------------------------------------------

# Arguments: t (1D array).
# Returns: x(t) = Square(t) + Triangle(t).
def x_of_t(t):
    # TODO: implement (this part is straightforward)
    return square_wave(t)+triangle_wave(t)


# Arguments: t (1D array), f0 (float), a (float).
# Returns: y(t) = x(a*t) * exp(j*2*pi*f0*t)  -- compressed AND modulated.
def y_of_t(t, f0, a):
    # TODO: implement the combined transformation
    return x_of_t(a*t)*(np.cos(2*np.pi*f0*t)+1j*np.sin(2*np.pi*f0*t))


def main():
    t = np.linspace(-5, 5, 2000)
    f0, a = 10.0, 10.0

    x = x_of_t(t)
    y = y_of_t(t, f0, a)

    f = np.linspace(-30, 30, 3000)
    X = cft(t, x, f)
    Y = cft(t, y, f)

    t_theory = np.linspace(-50, 50, 2000)

    theoretical_Y = (
        1 / abs(a)
    ) * cft(
        t_theory,
        x_of_t(t_theory),
        (f - f0) / a
    )

    print(f"Measured peak |Y(f)|:     {np.abs(Y).max():.4f}")
    print(f"Theoretical peak |Y(f)|:  {np.abs(theoretical_Y).max():.4f}")
    print(f"Ratio (should be ~1 if the property holds numerically): "
          f"{np.abs(Y).max() / np.abs(theoretical_Y).max():.4f}")
    print()
    print("If the ratio above is NOT close to 1, that's the same wall I hit.")
    print("Things worth trying, if you want to dig further:")
    print("  - Try replacing Square+Triangle with a decaying signal (e.g.")
    print("    a Gaussian) and confirm the property DOES hold there --")
    print("    that isolates whether it's the periodicity that's the issue.")
    print("  - Try varying the compression factor 'a' and see if the")
    print("    discrepancy changes in a way that reveals the pattern.")
    print("  - Ask your course staff directly whether the numerical CFT")
    print("    of a periodic signal is expected to obey standard scaling")
    print("    properties under this trapezoidal-integration approach --")
    print("    this may be a known limitation they want you to discover.")


if __name__ == "__main__":
    main()
