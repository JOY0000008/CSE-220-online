import numpy as np


# ============================================================
# BASIC CFT
# ============================================================

def cft(t, x, f):

    phase = 2 * np.pi * np.outer(
        f,
        t
    )

    real_part = np.trapezoid(
        x[None, :] *
        np.cos(phase),
        t,
        axis=-1
    )

    imag_part = -np.trapezoid(
        x[None, :] *
        np.sin(phase),
        t,
        axis=-1
    )

    return real_part + 1j * imag_part


def mse(a, b):
    return np.mean(
        np.abs(a - b) ** 2
    )


# ============================================================
# SIGNAL
# ============================================================

def x_t(t):
    return np.exp(-t ** 2)


def x_derivative(t):
    return -2 * t * np.exp(-t ** 2)


# ============================================================
# Q12 — TIME SHIFT
# ============================================================

def verify_time_shift(
    t,
    x,
    f,
    X,
    t0
):

    direct = cft(
        t,
        x_t(t - t0),
        f
    )

    formula = (
        X *
        np.exp(
            -1j * 2 * np.pi * f * t0
        )
    )

    return direct, formula


# ============================================================
# Q13 — DIFFERENTIATION
# ============================================================

def verify_differentiation(
    t,
    x,
    f,
    X
):

    direct = cft(
        t,
        x_derivative(t),
        f
    )

    formula = (
        1j *
        2 * np.pi *
        f *
        X
    )

    return direct, formula


# ============================================================
# Q14 — COMPLEX MODULATION
# ============================================================

def verify_modulation(
    t,
    x,
    f,
    f0
):

    modulated = (
        x *
        np.exp(
            1j * 2 * np.pi * f0 * t
        )
    )

    direct = cft(
        t,
        modulated,
        f
    )

    formula = cft(
        t,
        x,
        f - f0
    )

    return direct, formula


# ============================================================
# Q15 — COSINE MODULATION
# ============================================================

def verify_cosine_modulation(
    t,
    x,
    f,
    f0
):

    direct = cft(
        t,
        x * np.cos(
            2 * np.pi * f0 * t
        ),
        f
    )

    formula = (
        0.5 * cft(
            t,
            x,
            f - f0
        )
        +
        0.5 * cft(
            t,
            x,
            f + f0
        )
    )

    return direct, formula


# ============================================================
# Q16 — TIME SCALING
# ============================================================

def verify_time_scaling(
    t,
    x,
    f,
    a
):

    if a == 0:
        raise ValueError(
            "a must not be zero"
        )

    direct = cft(
        t,
        x_t(a * t),
        f
    )

    formula = (
        1 / abs(a)
    ) * cft(
        t,
        x,
        f / a
    )

    return direct, formula


# ============================================================
# Q17 — SHIFT + SCALE
# ============================================================

def verify_shift_and_scale(
    t,
    x,
    f,
    a,
    t0
):

    if a == 0:
        raise ValueError(
            "a must not be zero"
        )

    y = x_t(
        a * t - t0
    )

    direct = cft(
        t,
        y,
        f
    )

    formula = (
        1 / abs(a)
    ) * cft(
        t,
        x,
        (f / a)
    ) * np.exp(
        -1j *
        2 * np.pi *
        (f / a) *
        t0
    )

    return direct, formula


# ============================================================
# Q18 — DIFFERENTIATION OF SHIFTED/SCALED SIGNAL
# ============================================================

def verify_shift_scale_derivative(
    t,
    f,
    a=2.0,
    t0=1.0
):

    if a == 0:
        raise ValueError(
            "a must not be zero"
        )

    z = a * t - t0

    y_derivative = (
        -2 *
        a *
        z *
        np.exp(-z ** 2)
    )

    direct = cft(
        t,
        y_derivative,
        f
    )

    X_scaled = (
        1 / abs(a)
    ) * cft(
        t,
        x_t(t),
        f / a
    )

    X_shifted = (
        X_scaled *
        np.exp(
            -1j *
            2 * np.pi *
            (f / a) *
            t0
        )
    )

    formula = (
        1j *
        2 * np.pi *
        f *
        X_shifted
    )

    return direct, formula


# ============================================================
# Q19 — KNOWN FOURIER SERIES COEFFICIENTS
# ============================================================

def known_coefficients():

    # 2 + 3cos(w0t) - 4sin(2w0t)

    return {
        -2: -2j,
        -1: 1.5,
        0: 2.0,
        1: 1.5,
        2: 2j
    }


# ============================================================
# MAIN
# ============================================================

def main():

    t = np.linspace(
        -5,
        5,
        2000
    )

    x = x_t(t)

    f = np.linspace(
        -10,
        10,
        1000
    )

    X = cft(
        t,
        x,
        f
    )

    tests = [

        (
            "Q12",
            verify_time_shift(
                t, x, f, X, 1.0
            )
        ),

        (
            "Q13",
            verify_differentiation(
                t, x, f, X
            )
        ),

        (
            "Q14",
            verify_modulation(
                t, x, f, 3.0
            )
        ),

        (
            "Q15",
            verify_cosine_modulation(
                t, x, f, 3.0
            )
        ),

        (
            "Q16",
            verify_time_scaling(
                t, x, f, 2.0
            )
        ),

        (
            "Q17",
            verify_shift_and_scale(
                t, x, f, 2.0, 1.0
            )
        ),

        (
            "Q18",
            verify_shift_scale_derivative(
                t,
                f,
                2.0,
                1.0
            )
        )
    ]

    for name, (direct, formula) in tests:

        print(
            name,
            "MSE =",
            mse(direct, formula)
        )

    print(
        "Q19:",
        known_coefficients()
    )


if __name__ == "__main__":
    main()
