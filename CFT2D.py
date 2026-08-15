import numpy as np


# ============================================================
# 2D CONTINUOUS FOURIER TRANSFORM
#
# img       : (Ny, Nx)
# x         : (Nx,)
# y         : (Ny,)
# u         : (Nu,)
# v         : (Nv,)
#
# spectrum  : (Nu, Nv)
# ============================================================

def cft2d(img, x, y, u, v):

    # --------------------------------------------------------
    # Integrate with respect to x
    # --------------------------------------------------------

    exp_x = np.exp(
        -1j *
        2 * np.pi *
        np.outer(u, x)
    )

    # img:
    # (Ny, Nx)
    #
    # img[None, :, :]:
    # (1, Ny, Nx)
    #
    # exp_x[:, None, :]:
    # (Nu, 1, Nx)
    #
    # multiplication:
    # (Nu, Ny, Nx)

    F_u_y = np.trapezoid(
        img[None, :, :] *
        exp_x[:, None, :],
        x,
        axis=-1
    )

    # F_u_y:
    # (Nu, Ny)

    # --------------------------------------------------------
    # Integrate with respect to y
    # --------------------------------------------------------

    exp_y = np.exp(
        -1j *
        2 * np.pi *
        np.outer(v, y)
    )

    # F_u_y[:, None, :]
    # (Nu, 1, Ny)
    #
    # exp_y[None, :, :]
    # (1, Nv, Ny)
    #
    # multiplication:
    # (Nu, Nv, Ny)

    F_u_v = np.trapezoid(
        F_u_y[:, None, :] *
        exp_y[None, :, :],
        y,
        axis=-1
    )

    # (Nu, Nv)

    return F_u_v


# ============================================================
# INVERSE 2D CFT
# ============================================================

def icft2d(
    spectrum,
    u,
    v,
    x,
    y
):

    # --------------------------------------------------------
    # Integrate with respect to u
    # --------------------------------------------------------

    exp_u = np.exp(
        1j *
        2 * np.pi *
        np.outer(x, u)
    )

    # spectrum:
    # (Nu, Nv)
    #
    # spectrum[None, :, :]:
    # (1, Nu, Nv)
    #
    # exp_u[:, :, None]:
    # (Nx, Nu, 1)
    #
    # multiplication:
    # (Nx, Nu, Nv)

    f_x_v = np.trapezoid(
        spectrum[None, :, :] *
        exp_u[:, :, None],
        u,
        axis=1
    )

    # (Nx, Nv)

    # --------------------------------------------------------
    # Integrate with respect to v
    # --------------------------------------------------------

    exp_v = np.exp(
        1j *
        2 * np.pi *
        np.outer(y, v)
    )

    # f_x_v[:, None, :]
    # (Nx, 1, Nv)
    #
    # exp_v[None, :, :]
    # (1, Ny, Nv)
    #
    # multiplication:
    # (Nx, Ny, Nv)

    image = np.trapezoid(
        f_x_v[:, None, :] *
        exp_v[None, :, :],
        v,
        axis=-1
    )

    # image = (Nx, Ny)
    #
    # Original image is (Ny, Nx)

    return image.real.T


# ============================================================
# MSE
# ============================================================

def mse(a, b):
    return np.mean(
        np.abs(a - b) ** 2
    )


# ============================================================
# TEST IMAGE
# ============================================================

def image_2d(X, Y):

    return np.exp(
        -(X ** 2 + Y ** 2)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    x = np.linspace(
        -3,
        3,
        50
    )

    y = np.linspace(
        -3,
        3,
        50
    )

    X, Y = np.meshgrid(
        x,
        y
    )

    img = image_2d(
        X,
        Y
    )

    u = np.linspace(
        -5,
        5,
        50
    )

    v = np.linspace(
        -5,
        5,
        50
    )

    spectrum = cft2d(
        img,
        x,
        y,
        u,
        v
    )

    reconstructed = icft2d(
        spectrum,
        u,
        v,
        x,
        y
    )

    print(
        "Reconstruction MSE:",
        mse(img, reconstructed)
    )


if __name__ == "__main__":
    main()
