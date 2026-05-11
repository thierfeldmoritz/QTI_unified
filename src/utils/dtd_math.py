import numpy as np
from dipy.reconst.qti import from_3x3_to_6x1, from_6x6_to_21x1, from_21x1_to_6x6


# -------------------------------------------------------------------------------------
# inner product for Voigt notation tensors
def voigt_inner_product(a, b):
    return np.sum(a * b, axis=-1)


# -------------------------------------------------------------------------------------
# helper function: convert 1x6_to_3x3 (3D) (arbitrary order! not Voigt):
def pred_1x6_to_3x3(pred):
    f = 1 / np.sqrt(2)
    L = np.zeros((pred.shape[0], 3, 3))
    for i in range(3):
        L[:, i, i] = pred[:, 1 + i]
    # L[:,0,0] = pred[:, 1]; L[:,1,1] = pred[:, 2]; L[:,2,2] = pred[:, 3]
    L[:, 1, 0] = f * pred[:, 4]
    L[:, 2, 0] = f * pred[:, 5]
    L[:, 2, 1] = f * pred[:, 6]
    return L


# -------------------------------------------------------------------------------------
# helper function: convert_3x3_to_1x6 (3D) (Voigt):
def convert_3x3_to_1x6(D):
    if not np.shape(D)[1:] == (3, 3):
        raise Exception("Tensor has wrong shape for Voigt notation.")
    f = np.sqrt(2)
    xx = D[:, 0, 0]
    yy = D[:, 1, 1]
    zz = D[:, 2, 2]
    xy = f * D[:, 0, 1]
    xz = f * D[:, 0, 2]
    yz = f * D[:, 1, 2]
    return np.stack((xx, yy, zz, xy, xz, yz), axis=-1)


# -------------------------------------------------------------------------------------
# DTI signal model for individual diffusion tensors:
def DTI_signal(dtens, btens, log_S=False):
    logS = -1 * voigt_inner_product(btens, dtens)

    if log_S:
        return logS
    else:
        return np.exp(logS)

# -------------------------------------------------------------------------------------
# QTI signal model for individual diffusion tensors:
def QTI_signal(mfs, btens, log_S=False):
    mfs = np.asarray(mfs)
    dtens = mfs[..., 1:7]
    ctens = mfs[..., 7:28]

    if btens.shape[-1] == 6 and btens.ndim == 3 and btens.shape[1] == 1:
        btens = np.squeeze(btens, axis=1)
        
    btens_sq = convert_1x6_to_1x21(btens)

    logS = -1 * voigt_inner_product(btens, dtens)\
        + 0.5 * voigt_inner_product(btens_sq, ctens)

    if log_S:
        return logS
    else:
        return np.exp(logS)

def norm_vector(d_vector):
    norm = np.linalg.norm(d_vector)
    return d_vector / norm


def get_rot_y(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def get_rot_z(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def get_rot_matrix(d_vector):
    d_vector = norm_vector(d_vector)
    d_rot_matrix = np.eye(3)

    if d_vector[0] == 1.0 and d_vector[1] == 0.0 and d_vector[2] == 0.0:
        return d_rot_matrix

    if d_vector[1] == 0 and d_vector[0] != 0 and d_vector[2] != 0:
        eta = np.arctan2(d_vector[2], d_vector[0])
        d_rot_matrix = get_rot_y(-eta)
    elif d_vector[2] == 0 and d_vector[0] != 0 and d_vector[1] != 0:
        phi = np.arctan2(d_vector[1], d_vector[0])
        d_rot_matrix = get_rot_z(phi)
    else:
        phi = np.arctan2(abs(d_vector[1]), d_vector[0])
        if d_vector[1] > 0.0:
            phi *= -1.0
        d_rot_z = get_rot_z(phi)

        d_temp = np.dot(d_rot_z, d_vector)

        eta = np.arctan2(d_temp[2], d_temp[0])
        d_rot_y = get_rot_y(eta)

        d_rot_matrix = np.dot(d_rot_z.T, d_rot_y.T)

    return d_rot_matrix


# -------------------------------------------------------------------------------------
# helper function: self outer product on Voigt tensor (2D):
def convert_1x6_to_1x21(n2):
    xx = n2[..., 0]
    yy = n2[..., 1]
    zz = n2[..., 2]
    xy = n2[..., 3]
    xz = n2[..., 4]
    yz = n2[..., 5]

    t = np.zeros((*n2.shape[:-1], 21))

    t[..., 0:3] = np.sqrt(1) * np.stack((xx * xx, yy * yy, zz * zz), axis=-1)
    t[..., 3:6] = np.sqrt(2) * np.stack((xx * yy, xx * zz, yy * zz), axis=-1)
    t[..., 6:9] = np.sqrt(2) * np.stack((xx * yz, yy * xz, zz * xy), axis=-1)
    t[..., 9:11] = np.sqrt(2) * np.stack((xx * xy, xx * xz), axis=-1)
    t[..., 11:13] = np.sqrt(2) * np.stack((yy * xy, yy * yz), axis=-1)
    t[..., 13:15] = np.sqrt(2) * np.stack((zz * xz, zz * yz), axis=-1)
    t[..., 15:18] = np.sqrt(1) * np.stack((xy * xy, xz * xz, yz * yz), axis=-1)
    t[..., 18:21] = np.sqrt(2) * np.stack((xy * xz, xy * yz, xz * yz), axis=-1)

    return t


# -------------------------------------------------------------------------------------
# compute covariance tensor of dtd:
def compute_cumulant_tensors(dtens):
    # checks:
    if dtens.shape[1:] == (3, 3):
        dtens = convert_3x3_to_1x6(dtens)
    elif dtens.shape[-1] != 6:
        raise ValueError("dtens has wrong shape")

    # compute self outer prod of mean diff tensor:
    avg_dtens = np.mean(dtens, axis=0)
    avg_dtens = avg_dtens[None, :]
    avg_dtens_sq = convert_1x6_to_1x21(avg_dtens)

    # not verified, check that the results are the same

    # compute SELF outer prod of individual diff tensors VECTORIZED VERSION:
    # mean_outer_product = np.mean(outer_products.reshape(-1, outer_products.shape[-2], outer_products.shape[-1]), axis=0) # reshape unnecessary
    outer_products = np.einsum("ij,ik->ijk", dtens, dtens).reshape(-1, 6, 6)
    mean_outer_product = np.mean(outer_products, axis=0)
    ctens = tm_6x6_to_1x21(mean_outer_product) - avg_dtens_sq

    # # compute SELF outer prod of individual diff tensors LOOP VERSION:
    # outer_products = []
    # for i in range(dtens.shape[0]):
    #     outer_products.append(np.outer(dtens[i], dtens[i]))
    # outer_products = np.array(outer_products)
    # mean_outer_product = np.mean(outer_products, axis=0)
    # ctens = tm_6x6_to_1x21(mean_outer_product) - avg_dtens_sq

    return avg_dtens, ctens


"""
    # compute outer prod of individual diff tensors:
    outer_products = []
    for i in range(dtens.shape[0]):
        for j in range(dtens.shape[0]):
            if i == j:
                continue
            outer_products.append(np.outer(dtens[i], dtens[j]))
    outer_products = np.array(outer_products)
    mean_outer_product_a = np.mean(outer_products, axis=0)
    ctens = tm_6x6_to_1x21(mean_outer_product_a) - avg_dtens_sq

    # this is wrong, it takes the outer product of the whole array with itself
    # instead of the outer product of all rows with all other rows
    # outer_products = np.multiply.outer(dtens, dtens)
    # outer_products = np.einsum('ij,kl->ijkl', dtens, dtens).reshape(-1, 2, 2)
    # mean_outer_product = np.mean(outer_products.reshape(-1, outer_products.shape[-2], outer_products.shape[-1]), axis=0)
    # ctens_b = tm_6x6_to_1x21(mean_outer_product) - avg_dtens_sq
"""

"""
#-------------------------------------------------------------------------------------
# helper function: outer product on Voigt tensor, but vectorized for 3D input:
def convert_1x6_to_1x21_3D(n2):

    n2_shape = n2.shape
    if n2_shape[-1] != 6:
        raise Exception("Voigt notation tensor has wrong shape along the last dimension.")
    elif len(n2_shape) == 2:
        n2 = n2[None, :]
        n2_shape = n2.shape
    elif len(n2_shape) == 1:
        n2 = n2[None, None, :]
        n2_shape = n2.shape
    elif len(n2_shape) > 3:
        raise Exception("Voigt notation tensor has number of dimensions > 3")

    xx = n2[:, :, 0]
    yy = n2[:, :, 1]
    zz = n2[:, :, 2]
    xy = n2[:, :, 3]
    xz = n2[:, :, 4]
    yz = n2[:, :, 5]

    t = np.zeros((n2_shape[0], n2_shape[1], 21))

    t[:, :, 0:3] = np.sqrt(1) * np.stack((xx * xx, yy * yy, zz * zz), axis=-1)
    t[:, :, 3:6] = np.sqrt(2) * np.stack((xx * yy, xx * zz, yy * zz), axis=-1)
    t[:, :, 6:9] = np.sqrt(2) * np.stack((xx * yz, yy * xz, zz * xy), axis=-1)  # note: fixed potential error in last entry
    t[:, :, 9:11] = np.sqrt(2) * np.stack((xx * xy, xx * xz), axis=-1)
    t[:, :, 11:13] = np.sqrt(2) * np.stack((yy * xy, yy * yz), axis=-1)
    t[:, :, 13:15] = np.sqrt(2) * np.stack((zz * xz, zz * yz), axis=-1)
    t[:, :, 15:18] = np.sqrt(1) * np.stack((xy * xy, xz * xz, yz * yz), axis=-1)
    t[:, :, 18:21] = np.sqrt(2) * np.stack((xy * xz, xy * yz, xz * yz), axis=-1)

    return np.squeeze(t)
"""


# #-------------------------------------------------------------------------------------
# outer product for Voigt notation tensors
def voigt_outer_product(a, b):
    return np.outer(a, b)


# -------------------------------------------------------------------------------------
# helper function: NOT TESTED! allocation verified
# def tm_6x6_to_1x21(a):
#     t = np.array(a)
#     if t.shape != (6, 6):
#         raise Exception("Tensor has wrong dimensions.")
#
#     t1 = np.sqrt(1) * np.array([t[0, 0], t[1, 1], t[2, 2]])
#     t2 = np.sqrt(2) * np.array([t[0, 1], t[0, 2], t[1, 2]])
#     t3 = np.sqrt(2) * np.array([t[0, 5], t[1, 4], t[2, 3]])  # xxyz, yyxz, zzxy
#     t4 = np.sqrt(2) * np.array([t[0, 3], t[0, 4], t[1, 3]])
#     t5 = np.sqrt(2) * np.array([t[1, 5], t[2, 4], t[2, 5]])
#     t6 = np.sqrt(1) * np.array([t[3, 3], t[4, 4], t[5, 5]])
#     t7 = np.sqrt(2) * np.array([t[3, 4], t[3, 5], t[4, 5]])
#
#     t_out = np.concatenate((t1, t2, t3, t4, t5, t6, t7))
#     return t_out


# helper function: supports input with an additional list index dimension at dimension zero
# vectorized version of tm_6x6_to_1x21 for list input: equality verified
def tm_6x6_to_1x21(a):
    t = np.array(a)
    if t.ndim == 2 and t.shape == (6, 6):
        t = t[None, :, :]  # Add a list dimension if input is a single tensor
    elif t.shape[1:] != (6, 6):
        raise Exception(
            "Tensor has wrong dimensions. Expected shape (N, 6, 6) or (6, 6)."
        )

    t1 = np.sqrt(1) * t[..., [0, 1, 2], [0, 1, 2]]
    t2 = np.sqrt(2) * t[..., [0, 0, 1], [1, 2, 2]]
    t3 = np.sqrt(2) * t[..., [0, 1, 2], [5, 4, 3]]  # xxyz, yyxz, zzxy
    t4 = np.sqrt(2) * t[..., [0, 0, 1], [3, 4, 3]]
    t5 = np.sqrt(2) * t[..., [1, 2, 2], [5, 4, 5]]
    t6 = np.sqrt(1) * t[..., [3, 4, 5], [3, 4, 5]]
    t7 = np.sqrt(2) * t[..., [3, 3, 4], [4, 5, 5]]

    t_out = np.concatenate((t1, t2, t3, t4, t5, t6, t7), axis=-1)
    return t_out


def reorder_mfs(mfs):
    # function to reorder the voigt vector indices to
    # switch between conventions QTIp/DIPY & Westin
    # Reorder indices of the 1x6 d vector
    d_reorder_indices = [0, 1, 2, 5, 4, 3]
    # Reorder indices of the 1x21 c vector
    c_reorder_indices = [
        0,
        1,
        2,
        5,
        4,
        3,
        9,
        10,
        6,
        11,
        7,
        12,
        8,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        19,
    ]

    d_reorder_indices = [i + 1 for i in d_reorder_indices]
    c_reorder_indices = [i + 7 for i in c_reorder_indices]
    reorder_indices = [0] + d_reorder_indices + c_reorder_indices
    mfs_out = mfs[..., reorder_indices]
    return mfs_out


# -------------------------------------------------------------------------------------
# dipy implementation of the QTI parameters
# dipy tensor math differs from own implementation
# perhaps adapt for consistency
# check if voigt notation convention is the same
# -------------------------------------------------------------------------------------


# These tensors are used in the calculation of the QTI parameters
e_iso = np.eye(3) / 3
E_iso = np.eye(6) / 3
E_bulk = from_3x3_to_6x1(e_iso) @ from_3x3_to_6x1(e_iso).T
E_shear = E_iso - E_bulk
E_tsym = E_bulk + 0.4 * E_shear


def S0_hat(params):
    """Estimated signal without diffusion-weighting.

    Returns
    -------
    S0 : numpy.ndarray
    """
    # modified, no log signal for us
    S0 = params[..., 0]
    return S0


def md(params):
    """Mean diffusivity.

    Returns
    -------
    md : numpy.ndarray

    Notes
    -----
    Mean diffusivity is calculated as

        .. math::

            \\text{MD} = \\langle \\mathbf{D} \\rangle :
            \\mathbf{E}_\\text{iso}
    """
    md = np.matmul(params[..., np.newaxis, 1:7], from_3x3_to_6x1(e_iso))[..., 0, 0]
    return md


def v_md(params):
    """Variance of microscopic mean diffusivities.

    Returns
    -------
    v_md : numpy.ndarray

    Notes
    -----
    Variance of microscopic mean diffusivities is calculated as

        .. math::

            V_\\text{MD} = \\mathbb{C} : \\mathbb{E}_\\text{bulk}
    """
    v_md = np.matmul(params[..., np.newaxis, 7::], from_6x6_to_21x1(E_bulk))[..., 0, 0]
    return v_md


def v_shear(params):
    """Shear variance.

    Returns
    -------
    v_shear : numpy.ndarray

    Notes
    -----
    Shear variance is calculated as

        .. math::

            V_\\text{shear} = \\mathbb{C} : \\mathbb{E}_\\text{shear}
    """
    v_shear = np.matmul(params[..., np.newaxis, 7::], from_6x6_to_21x1(E_shear))[
        ..., 0, 0
    ]
    return v_shear


def v_iso(params):
    """Total isotropic variance.

    Returns
    -------
    v_iso : numpy.ndarray

    Notes
    -----
    Total isotropic variance is calculated as

        .. math::

            V_\\text{iso} = \\mathbb{C} : \\mathbb{E}_\\text{iso}
    """
    v_iso = np.matmul(params[..., np.newaxis, 7::], from_6x6_to_21x1(E_iso))[..., 0, 0]
    return v_iso


def d_sq(params):
    """Diffusion tensor's outer product with itself.

    Returns
    -------
    d_sq : numpy.ndarray
    """
    d_sq = np.matmul(params[..., 1:7, np.newaxis], params[..., np.newaxis, 1:7])
    return d_sq


def mean_d_sq(params):
    """Average of microscopic diffusion tensors' outer products with
    themselves.

    Returns
    -------
    mean_d_sq : numpy.ndarray

    Notes
    -----
    Average of microscopic diffusion tensors' outer products with
    themselves is calculated as

        .. math::

            \\langle \\mathbf{D} \\otimes \\mathbf{D} \\rangle =
            \\mathbb{C} +
            \\langle \\mathbf{D} \\rangle \\otimes \\langle \\mathbf{D}
            \\rangle
    """
    mean_d_sq = from_21x1_to_6x6(params[..., 7::, np.newaxis]) + d_sq(params)
    return mean_d_sq


def c_md(params):
    """Normalized variance of mean diffusivities.

    Returns
    -------
    c_md : numpy.ndarray

    Notes
    -----
    Normalized variance of microscopic mean diffusivities is calculated as

        .. math::

            C_\\text{MD} = \\frac{\\mathbb{C} : \\mathbb{E}_\\text{bulk}}
            {\\langle \\mathbf{D} \\otimes \\mathbf{D} \\rangle :
            \\mathbb{E}_\\text{bulk}}
    """
    c_md = (
        v_md(params)
        / np.matmul(
            np.swapaxes(from_6x6_to_21x1(mean_d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_bulk),
        )[..., 0, 0]
    )
    return c_md


def c_mu(params):
    """Normalized microscopic anisotropy.

    Returns
    -------
    c_mu : numpy.ndarray

    Notes
    -----
    Normalized microscopic anisotropy is calculated as

        .. math::

            C_\\mu = \\frac{3}{2} \\frac{\\langle \\mathbf{D} \\otimes
            \\mathbf{D}
            \\rangle : \\mathbb{E}_\\text{shear}}{\\langle \\mathbf{D}
            \\otimes
            \\mathbf{D} \\rangle : \\mathbb{E}_\\text{iso}}
    """
    c_mu = (
        1.5
        * np.matmul(
            np.swapaxes(from_6x6_to_21x1(mean_d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_shear),
        )
        / np.matmul(
            np.swapaxes(from_6x6_to_21x1(mean_d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_iso),
        )
    )[..., 0, 0]
    return c_mu


def ufa(params):
    """Microscopic fractional anisotropy.

    Returns
    -------
    ufa : numpy.ndarray

    Notes
    -----
    Microscopic fractional anisotropy is calculated as

        .. math::

            \\mu\\text{FA} = \\sqrt{C_\\mu}
    """
    ufa = np.sqrt(c_mu(params))
    return ufa


def c_m(params):
    """Normalized macroscopic anisotropy.

    Returns
    -------
    c_m : numpy.ndarray

    Notes
    -----
    Normalized macroscopic anisotropy is calculated as

        .. math::

            C_\\text{M} = \\frac{3}{2} \\frac{\\langle \\mathbf{D} \\rangle
            \\otimes \\langle \\mathbf{D} \\rangle :
            \\mathbb{E}_\\text{shear}}
            {\\langle \\mathbf{D} \\rangle \\otimes \\langle \\mathbf{D}
            \\rangle :
            \\mathbb{E}_\\text{iso}}
    """
    c_m = (
        1.5
        * np.matmul(
            np.swapaxes(from_6x6_to_21x1(d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_shear),
        )
        / np.matmul(
            np.swapaxes(from_6x6_to_21x1(d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_iso),
        )
    )[..., 0, 0]
    return c_m


def fa(params):
    """Fractional anisotropy.

    Returns
    -------
    fa : numpy.ndarray

    Notes
    -----
    Fractional anisotropy is calculated as

        .. math::

            \\text{FA} = \\sqrt{C_\\text{M}}
    """
    fa = np.sqrt(c_m(params))
    return fa


def c_c(params):
    """Microscopic orientation coherence.

    Returns
    -------
    c_c : numpy.ndarray

    Notes
    -----
    Microscopic orientation coherence is calculated as

        .. math::

            C_c = \\frac{C_\\text{M}}{C_\\mu}
    """
    c_c = c_m(params) / c_mu(params)
    return c_c


def mk(params):
    """Mean kurtosis.

    Returns
    -------
    mk : numpy.ndarray

    Notes
    -----
    Mean kurtosis is calculated as

        .. math::

            \\text{MK} = K_\\text{bulk} + K_\\text{shear}
    """
    mk = k_bulk(params) + k_shear(params)
    return mk


def k_bulk(params):
    """Bulk kurtosis.

    Returns
    -------
    k_bulk : numpy.ndarray

    Notes
    -----
    Bulk kurtosis is calculated as

        .. math::

            K_\\text{bulk} = 3 \\frac{\\mathbb{C} :
            \\mathbb{E}_\\text{bulk}}
            {\\langle \\mathbf{D} \\rangle \\otimes \\langle \\mathbf{D}
            \\rangle : \\mathbb{E}_\\text{bulk}}
    """
    k_bulk = (
        3
        * np.matmul(params[..., np.newaxis, 7::], from_6x6_to_21x1(E_bulk))
        / np.matmul(
            np.swapaxes(from_6x6_to_21x1(d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_bulk),
        )
    )[..., 0, 0]
    return k_bulk


def k_shear(params):
    """Shear kurtosis.

    Returns
    -------
    k_shear : numpy.ndarray

    Notes
    -----
    Shear kurtosis is calculated as

        .. math::

            K_\\text{shear} = \\frac{6}{5} \\frac{\\mathbb{C} :
            \\mathbb{E}_\\text{shear}}{\\langle \\mathbf{D} \\rangle
            \\otimes
            \\langle \\mathbf{D} \\rangle : \\mathbb{E}_\\text{bulk}}
    """
    k_shear = (
        6
        / 5
        * np.matmul(params[..., np.newaxis, 7::], from_6x6_to_21x1(E_shear))
        / np.matmul(
            np.swapaxes(from_6x6_to_21x1(d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_bulk),
        )
    )[..., 0, 0]
    return k_shear


def k_mu(params):
    """Microscopic kurtosis.

    Returns
    -------
    k_mu : numpy.ndarray

    Notes
    -----
    Microscopic kurtosis is calculated as

        .. math::

            K_\\mu = \\frac{6}{5} \\frac{\\langle \\mathbf{D} \\otimes
            \\mathbf{D}
            \\rangle : \\mathbb{E}_\\text{shear}}{\\langle \\mathbf{D}
            \\rangle
            \\otimes \\langle \\mathbf{D} \\rangle :
            \\mathbb{E}_\\text{bulk}}
    """
    k_mu = (
        6
        / 5
        * np.matmul(
            np.swapaxes(from_6x6_to_21x1(mean_d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_shear),
        )
        / np.matmul(
            np.swapaxes(from_6x6_to_21x1(d_sq(params)), -1, -2),
            from_6x6_to_21x1(E_bulk),
        )
    )[..., 0, 0]
    return k_mu

def projection_metrics_to_eigvals(d_iso, d_delta):
    """Convert axisymmetric d- or b-tensor projection metrics to eigenvalues.
    d_iso: size part, d_delta: shape part"""#
    # d_iso = 1/3 * d_trace
    # principal axis changed to x (from z, see Lampinen review)

    d_perp = d_iso * (1 - d_delta)

    d_par = d_iso * (1 + 2 * d_delta)

    return d_par, d_perp, d_perp

def trace_shape_to_eigvals(d_trace, d_shape):
    # d_iso = 1/3 * d_trace
    d_par  = (d_trace/3.0) * (1 + 2*d_shape)
    d_perp = (d_trace/3.0) * (1 - d_shape)
    return np.array([d_par, d_perp, d_perp])

def axisym_eigvals_to_projection_metrics(d_par, d_perp):
    # D_iso is prop to the size of an axisymmetric tensor:
    #     D_iso = (D_par + 2*D_perp) / 3
    # d_iso = 1/3 * d_trace

    # Delta quantifies the microscopic anisotropy (shape of the tensor):
    #     Delta = (D_par - D_perp) / (D_par + 2*D_perp)

    d_iso = (d_par + 2 * d_perp) / 3.0
    d_delta = (d_par - d_perp) / (d_par + 2 * d_perp)

    return d_iso, d_delta

def u_from_angles_deg(az_deg, pol_deg):
    phi = np.deg2rad(az_deg); theta = np.deg2rad(pol_deg)
    return (float(np.sin(theta)*np.cos(phi)),
            float(np.sin(theta)*np.sin(phi)),
            float(np.cos(theta)))

def T2_decay_factor(TE, T2):
    return np.exp(-TE / T2)

def scaled_noise_from_snr(clean: np.ndarray, snr, s0, rng: np.random.Generator) -> np.ndarray:
    """
    Rician(-type?) magnitude noise:
      S_noisy = s0 * sqrt( (clean/s0 + z1/snr)^2 + (z2/snr)^2 )
    where z1,z2 ~ N(0,1) i.i.d. (drawn with `rng`).
    `snr` and `s0` can be scalars or arrays broadcastable to `clean.shape`.
    """
    snr = np.asarray(snr, dtype=float)
    s0  = np.asarray(s0,  dtype=float)
    z1  = rng.standard_normal(clean.shape)
    z2  = rng.standard_normal(clean.shape)
    return s0 * np.sqrt((clean / s0 + z1 / snr) ** 2 + (z2 / snr) ** 2)

def rician_complex(sub, snr, s0, rng):
    """Rician magnitude via complex noise; snr/s0 can be scalar or broadcastable to sub."""
    snr = np.asarray(snr, dtype=float)
    s0  = np.asarray(s0,  dtype=float)
    sigma = s0 / snr

    # randomize phase, add i.i.d. Gaussian noise to real/imag, take magnitude
    rand_phase = np.exp(-1j * 2 * np.pi * rng.random(sub.shape))
    z = sub * rand_phase
    noisy_real = z.real + sigma * rng.standard_normal(sub.shape)
    noisy_imag = z.imag + sigma * rng.standard_normal(sub.shape)
    return np.abs(noisy_real + 1j * noisy_imag)


# watson dtd additions
# ---------- ODI <-> kappa ----------

def kappa_from_odi(odi, cap=10000.0, eps=1e-12):
    odi = np.clip(odi, eps, 1.0 - eps)
    kappa = 1.0 / np.tan(0.5*np.pi*odi)   # = cot((π/2)*ODI)
    return np.minimum(kappa, cap)

# ---------- Sphere utilities ----------

def spherical_fibonacci_points(M):
    i = np.arange(M) + 0.5
    phi = 2*np.pi*i / ((1+np.sqrt(5))/2)
    z = 1 - 2*i / M
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack((x, y, z), axis=1)

def random_unit_vectors(n, rng=None):
    if rng is None: rng = np.random.default_rng()
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-15
    return v

# ---------- Discrete Watson sampler (supports ER or SF grid) ----------

def watson_sample_orientations(mu, odi=None, kappa=None, n=1, udirs=None, grid_size=2000, rng=None):
    if rng is None: rng = np.random.default_rng()
    mu = np.asarray(mu, dtype=float)
    mu /= np.linalg.norm(mu) + 1e-15

    if kappa is None:
        if odi is None:
            raise ValueError("Provide either kappa or odi.")
        kappa = kappa_from_odi(odi)

    if udirs is None:
        udirs = spherical_fibonacci_points(grid_size)
    else:
        udirs = np.asarray(udirs, dtype=float)
        udirs /= np.linalg.norm(udirs, axis=1, keepdims=True) + 1e-15

    dots = udirs @ mu
    logits = kappa * (dots**2)
    logits -= logits.max()               # numerical stability
    w = np.exp(logits)
    w /= w.sum()

    idx = rng.choice(len(udirs), size=n, p=w)
    return udirs[idx]

# ---------- Main generators ----------

def generate_fibers_WM_watson(
    n,
    mu=(1,0,0),
    odi=0.2,
    base_d_par=1.7e-9,
    base_d_perp=3.0e-10,
    sigma_iso=0.15,      # relative to base d_iso
    sigma_delta=0.15,    # absolute
    udirs=None,          # optional ER grid (shape [M,3]); if None -> SF
    grid_size=2000,
    rng=None
):
    if rng is None: rng = np.random.default_rng()

    # Sample orientations from Watson
    dirs = watson_sample_orientations(mu=mu, odi=odi, n=n, udirs=udirs, grid_size=grid_size, rng=rng)

    # Sample size/shape around base
    base_iso, base_delta = axisym_eigvals_to_projection_metrics(base_d_par, base_d_perp)
    # d_iso   = rng.normal(loc=base_iso,   scale=sigma_iso * base_iso, size=n)
    # d_delta = rng.normal(loc=base_delta, scale=sigma_delta,          size=n)
    # # Physical clips
    # d_iso   = np.clip(d_iso, 1e-12, None) # here we could clip at free water limit
    # d_delta = np.clip(d_delta, -0.5, 1.0)

    # d_iso   = rng.normal(loc=base_iso,   scale=sigma_iso * base_iso, size=n)
    # d_delta = rng.normal(loc=base_delta, scale=sigma_delta,          size=n)

    d_iso   = truncated_normal(rng, base_iso,   sigma_iso * base_iso, lo=1e-12, hi=3.05e-9, size=n)  # hi ~ free water
    d_delta = truncated_normal(rng, base_delta, sigma_delta,          lo=-0.5,  hi=1.0,    size=n)  # avoids pile-up

    # d_iso   = sample_d_iso_lognormal(base_iso, n, sigma_rel=sigma_iso, rng=rng)
    # d_delta = sample_delta_logitnormal(base_delta, n, sigma=sigma_delta, rng=rng)  # no clipping needed

    params = []
    for i in range(n):
        lam = trace_shape_to_eigvals(3.0 * d_iso[i], d_delta[i])
        lam[1:] = np.maximum(lam[1:], 1e-12)          # d_perp > 0
        lam[0]  = max(lam[0], lam[1] + 1e-12)         # d_par >= d_perp
        u1, u2, u3 = dirs[i]
        params.append({
            "lambda_1": float(lam[0]),
            "lambda_2": float(lam[1]),
            "lambda_3": float(lam[2]),
            "u1": float(u1), "u2": float(u2), "u3": float(u3),
            "d_iso": float(d_iso[i]), "d_delta": float(d_delta[i]),
        })
    return params

def generate_dtd_gauss_random_orientations(
    n,
    base_d_par=1.7e-9,
    base_d_perp=3.0e-10,
    sigma_iso=0.15,
    sigma_delta=0.15,
    orientation_mode="independent",  # "independent" | "shared" | "explicit"
    explicit_dir=None,               # used only when orientation_mode == "explicit"
    rng=None
):
    """
    No Watson. Sample (d_iso, Δ) from Gaussians, convert to eigenvalues,
    and assign either:
      - independent random directions per tensor ("independent"),
      - one shared random direction for all tensors ("shared"),
      - one user-specified direction for all tensors ("explicit", pass explicit_dir).
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- size/shape sampling ----
    base_iso, base_delta = axisym_eigvals_to_projection_metrics(base_d_par, base_d_perp)

    # d_iso   = rng.normal(loc=base_iso,   scale=sigma_iso * base_iso, size=n)
    # d_delta = rng.normal(loc=base_delta, scale=sigma_delta,          size=n)
    # # Physical clips
    # d_iso   = np.clip(d_iso, 1e-12, None)
    # d_delta = np.clip(d_delta, -0.5, 1.0)

    d_iso   = truncated_normal(rng, base_iso,   sigma_iso, lo=1e-12, hi=3.05e-9, size=n)  # hi ~ free water
    d_delta = truncated_normal(rng, base_delta, sigma_delta,          lo=-0.5,  hi=1.0,    size=n)  # avoids pile-up

    #d_iso   = sample_d_iso_lognormal(base_iso, n, sigma_rel=sigma_iso, rng=rng)
    #d_delta = sample_delta_logitnormal(base_delta, n, sigma=sigma_delta, rng=rng)  # no clipping needed


    # ---- orientations ----
    if orientation_mode == "shared":
        dir_shared = random_unit_vectors(1, rng=rng)[0]
        dirs = np.repeat(dir_shared[None, :], n, axis=0)

    elif orientation_mode == "explicit":
        if explicit_dir is None:
            raise ValueError("orientation_mode='explicit' requires explicit_dir=(ux,uy,uz).")
        v = np.asarray(explicit_dir, dtype=float)
        v /= (np.linalg.norm(v) + 1e-15)
        dirs = np.repeat(v[None, :], n, axis=0)

    elif orientation_mode == "independent":
        dirs = random_unit_vectors(n, rng=rng)

    else:
        raise ValueError("orientation_mode must be 'independent', 'shared', or 'explicit'.")

    # ---- assemble tensors ----
    params = []
    for i in range(n):
        # trace_shape_to_eigvals expects d_trace, not d_iso -> pass 3*d_iso
        lam = trace_shape_to_eigvals(3.0 * d_iso[i], d_delta[i])

        # SPD guards
        lam[1:] = np.maximum(lam[1:], 1e-12)
        lam[0]  = max(lam[0], lam[1] + 1e-12)

        u1, u2, u3 = dirs[i]
        params.append({
            "lambda_1": float(lam[0]),
            "lambda_2": float(lam[1]),
            "lambda_3": float(lam[2]),
            "u1": float(u1), "u2": float(u2), "u3": float(u3),
            "d_iso": float(d_iso[i]), "d_delta": float(d_delta[i]),
        })

    return params

def sample_d_iso_lognormal(base_iso, n, sigma_rel=0.15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    # choose mu so E[lognormal] ≈ base_iso
    mu = np.log(base_iso) - 0.5 * (sigma_rel**2)
    return np.exp(rng.normal(loc=mu, scale=sigma_rel, size=n))

def sample_delta_logitnormal(base_delta, n, sigma=0.12, bounds=(-0.5, 1.0), rng=None):
    """ Map Δ in [lo,hi] -> u in (0,1) -> z=logit(u), add Normal noise, map back. """
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = bounds
    eps = 1e-9
    def to_unit(x):  return (np.clip(x, lo+eps, hi-eps) - lo) / (hi - lo)
    def logit(u):    return np.log(u) - np.log(1-u)
    def logistic(z): return 1/(1+np.exp(-z))
    z0 = logit(to_unit(base_delta))
    z  = rng.normal(loc=z0, scale=sigma, size=n)
    u  = logistic(z)
    return lo + u*(hi-lo)   # already within [-0.5, 1]

def truncated_normal(rng, mean, sd, lo, hi, size):
    out = np.empty(size); i = 0
    while i < size:
        k = size - i
        cand = rng.normal(mean, sd, k)
        cand = cand[(cand >= lo) & (cand <= hi)]
        m = len(cand)
        if m:
            out[i:i+m] = cand[:m]; i += m
    return out

