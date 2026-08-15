import numpy as np


# ============================================================
# BASIC UTILITIES
# ============================================================

def mse(a, b):
    return np.mean(np.abs(a - b) ** 2)


def signal_t(t):
    # x(t) = sin(t) + 2cos(2t)
    return np.sin(t) + 2 * np.cos(2 * t)


def calculate_cn(t, x, n_vals, omega_0, T):
    """
    Complex Fourier Series coefficients:

        c_k = (1/T) ∫ x(t)e^(-jkω0t) dt
    """
    phase = np.outer(n_vals * omega_0, t)

    integrand = x[None, :] * np.exp(-1j * phase)

    return (1 / T) * np.trapezoid(
        integrand,
        t,
        axis=-1
    )


def reconstruct_fs(t, n_vals, cn, omega_0):
    """
    x(t) = Σ c_k e^(jkω0t)
    """
    phase = np.outer(n_vals * omega_0, t)

    components = (
        cn[:, None] *
        np.exp(1j * phase)
    )

    return np.sum(components, axis=0)


# ============================================================
# Q4 — SUBSET RECONSTRUCTION
# ============================================================

def subset_reconstruction(t, n_vals, cn, omega_0, K=5):

    mask = np.abs(n_vals) <= K

    return reconstruct_fs(
        t,
        n_vals[mask],
        cn[mask],
        omega_0
    )


# ============================================================
# Q5 — DOMINANT HARMONICS
# ============================================================

def dominant_reconstruction(
    t,
    n_vals,
    cn,
    omega_0,
    K=5
):

    indices = np.argsort(np.abs(cn))[-K:]

    return reconstruct_fs(
        t,
        n_vals[indices],
        cn[indices],
        omega_0
    )


# ============================================================
# Q7 — FS TIME SHIFT
# ============================================================

def fs_time_shift(t, n_vals, omega_0, T, t0):

    x_shifted = signal_t(t - t0)

    c_direct = calculate_cn(
        t,
        x_shifted,
        n_vals,
        omega_0,
        T
    )

    c_base = calculate_cn(
        t,
        signal_t(t),
        n_vals,
        omega_0,
        T
    )

    # c'_k = c_k e^(-jkω0t0)
    c_formula = (
        c_base *
        np.exp(-1j * n_vals * omega_0 * t0)
    )

    return c_direct, c_formula


# ============================================================
# Q8 — FS TIME REVERSAL
# ============================================================

def fs_time_reversal(
    t,
    n_vals,
    omega_0,
    T
):

    reversed_x = signal_t(-t)

    c_direct = calculate_cn(
        t,
        reversed_x,
        n_vals,
        omega_0,
        T
    )

    c_base = calculate_cn(
        t,
        signal_t(t),
        n_vals,
        omega_0,
        T
    )

    # c'_k = c_(-k)
    c_formula = c_base[::-1]

    return c_direct, c_formula


# ============================================================
# Q9 — FS TIME SCALING
# ============================================================

def fs_time_scaling(
    t,
    n_vals,
    omega_0,
    T,
    a
):

    if a == 0:
        raise ValueError("a must not be zero")

    scaled_x = signal_t(a * t)

    new_omega_0 = abs(a) * omega_0
    new_T = T / abs(a)

    c_direct = calculate_cn(
        t,
        scaled_x,
        n_vals,
        new_omega_0,
        new_T
    )

    c_base = calculate_cn(
        t,
        signal_t(t),
        n_vals,
        omega_0,
        T
    )

    # Coefficients remain the same
    c_formula = c_base

    return c_direct, c_formula


# ============================================================
# Q10 — COMBINED SHIFT + SCALE + REVERSAL
# ============================================================

def fs_combined(
    t,
    n_vals,
    omega_0,
    T,
    a,
    t0
):

    if a == 0:
        raise ValueError("a must not be zero")

    # y(t) = x(at - t0)
    y = signal_t(a * t - t0)

    new_omega_0 = abs(a) * omega_0
    new_T = T / abs(a)

    c_direct = calculate_cn(
        t,
        y,
        n_vals,
        new_omega_0,
        new_T
    )

    c_base = calculate_cn(
        t,
        signal_t(t),
        n_vals,
        omega_0,
        T
    )

    # First shift
    c_formula = (
        c_base *
        np.exp(-1j * n_vals * omega_0 * t0)
    )

    # Then reversal if a < 0
    if a < 0:
        c_formula = c_base[::-1]

    return c_direct, c_formula


# ============================================================
# Q21 — FS DIFFERENTIATION
# ============================================================

def fs_differentiation(
    t,
    n_vals,
    omega_0,
    T
):

    # d/dt [sin(t) + 2cos(2t)]
    # = cos(t) - 4sin(2t)

    derivative_x = (
        np.cos(t)
        - 4 * np.sin(2 * t)
    )

    c_direct = calculate_cn(
        t,
        derivative_x,
        n_vals,
        omega_0,
        T
    )

    c_base = calculate_cn(
        t,
        signal_t(t),
        n_vals,
        omega_0,
        T
    )

    # d_k = jkω0 c_k
    c_formula = (
        1j *
        n_vals *
        omega_0 *
        c_base
    )

    return c_direct, c_formula


# ============================================================
# Q22 — FS INTEGRATION
# ============================================================

def cumulative_trapezoid_manual(t, x):

    result = np.zeros_like(x, dtype=float)

    dt = np.diff(t)

    result[1:] = np.cumsum(
        0.5 *
        (x[:-1] + x[1:]) *
        dt
    )

    return result


def fs_integration(
    t,
    n_vals,
    omega_0,
    T
):

    x = signal_t(t)

    # Direct numerical integral
    integrated_x = cumulative_trapezoid_manual(
        t,
        x
    )

    c_direct = calculate_cn(
        t,
        integrated_x,
        n_vals,
        omega_0,
        T
    )

    c_base = calculate_cn(
        t,
        x,
        n_vals,
        omega_0,
        T
    )

    # c'_k = c_k / (jkω0)
    c_formula = np.zeros_like(
        c_base,
        dtype=complex
    )

    for i, k in enumerate(n_vals):

        if k != 0:
            c_formula[i] = (
                c_base[i] /
                (1j * k * omega_0)
            )

    return c_direct, c_formula


# ============================================================
# Q23 — FS MULTIPLICATION
# ============================================================

def fs_multiplication(
    t,
    n_vals,
    omega_0,
    T
):

    x = np.sin(t)
    y = np.cos(2 * t)

    # Direct
    c_direct = calculate_cn(
        t,
        x * y,
        n_vals,
        omega_0,
        T
    )

    # Individual coefficients
    a = calculate_cn(
        t,
        x,
        n_vals,
        omega_0,
        T
    )

    b = calculate_cn(
        t,
        y,
        n_vals,
        omega_0,
        T
    )

    # c_k = Σ a_p b_(k-p)
    c_formula = np.zeros_like(
        a,
        dtype=complex
    )

    for i, k in enumerate(n_vals):

        for j, p in enumerate(n_vals):

            q = k - p

            matches = np.where(
                n_vals == q
            )[0]

            if len(matches) > 0:

                q_index = matches[0]

                c_formula[i] += (
                    a[j] * b[q_index]
                )
    c_formula = np.zeros_like(a, dtype=complex)

    # for i, k in enumerate(n_vals):

    #     for j, p in enumerate(n_vals):

    #         q = k - p

    #         for m, q_val in enumerate(n_vals):

    #             if q_val == q:
    #                 c_formula[i] += a[j] * b[m]

    return c_direct, c_formula


# ============================================================
# MAIN
# ============================================================

def main():

    t = np.linspace(
        0,
        2 * np.pi,
        2000
    )

    T = t[-1] - t[0]

    omega_0 = 2 * np.pi / T

    N = 20

    n_vals = np.arange(
        -N,
        N + 1
    )

    x = signal_t(t)

    cn = calculate_cn(
        t,
        x,
        n_vals,
        omega_0,
        T
    )

    # Q3
    reconstructed = reconstruct_fs(
        t,
        n_vals,
        cn,
        omega_0
    )

    print(
        "Q3 MSE:",
        mse(x, reconstructed)
    )

    # Q4
    subset = subset_reconstruction(
        t,
        n_vals,
        cn,
        omega_0
    )

    print(
        "Q4 MSE:",
        mse(x, subset)
    )

    # Q5
    dominant = dominant_reconstruction(
        t,
        n_vals,
        cn,
        omega_0
    )

    print(
        "Q5 MSE:",
        mse(x, dominant)
    )

    # Q7
    direct, formula = fs_time_shift(
        t,
        n_vals,
        omega_0,
        T,
        1.0
    )

    print(
        "Q7 MSE:",
        mse(direct, formula)
    )

    # Q8
    direct, formula = fs_time_reversal(
        t,
        n_vals,
        omega_0,
        T
    )

    print(
        "Q8 MSE:",
        mse(direct, formula)
    )

    # Q9
    direct, formula = fs_time_scaling(
        t,
        n_vals,
        omega_0,
        T,
        2.0
    )

    print(
        "Q9 MSE:",
        mse(direct, formula)
    )

    # Q21
    direct, formula = fs_differentiation(
        t,
        n_vals,
        omega_0,
        T
    )

    print(
        "Q21 MSE:",
        mse(direct, formula)
    )


if __name__ == "__main__":
    main()
