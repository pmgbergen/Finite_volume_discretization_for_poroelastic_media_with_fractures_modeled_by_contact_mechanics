"""
Module for calculating the semi-smooth Newton step in the contact mechanics problem.

Contains one fuction for solving the Tresca friction problem and one for the full
Coulomb problem.
"""
import numpy as np

from utils import l2


def contact_coulomb(T, u, F, bf, ct, cn, M_inv):
    """
    Calculate the semi-smooth Newton step for the Coulomb contact problem.
    The Newton step will appare in the elastic equations as a Robin condition:
    
    mortar_weight * lam + robin_weight * [u] = rhs

    where mortar_weight and robin_weight are matrices, lam the Lagrange multiplier,
    [u] the displacement jump and rhs a right hand side vector.

    Arguments:
        mg: mortar grid
        T: Lagrange multiplier
        u: displacement jump
        F: Coefficient of friction
        bf: Friction bound
        ct: Numerical parameter
        cn: Numerical parameter
        M_inv: Rotation matrix to decompose variables into normal and tangential parts.

    Returns:
       mortar_weight
       robin_weight
       rhs
    """
    num_cells = T.shape[1]
    nd = T.shape[0]
    # Process input
    if np.asarray(F).size==1:
        F = F * np.ones(num_cells)
    
    r_full = np.zeros((nd, num_cells))
    robin_weight = []
    mortar_weight = []
    rhs = np.array([])
    # Change coordinate system to the one alligned to the fractures
    T_hat = np.sum(M_inv * T, axis=1)
    u_hat = np.sum(M_inv * u, axis=1)
    u0_hat = np.sum(M_inv * u, axis=1)

    # Find contact and sliding region
    penetration_bc = active_penetration(T_hat[-1], u_hat[-1], cn)
    sliding_bc = active_sliding(T_hat[:-1], u_hat[:-1], bf, ct)

    zer = np.array([0]*(nd - 1))
    zer1 = np.array([0]*(nd))
    zer1[-1] = 1
    for i in range(T_hat.shape[1]):
        if sliding_bc[i] & penetration_bc[i]: # in contact and sliding
            L, r, v = L_r(T_hat[:-1, i], u_hat[:-1, i], bf[i], ct)
            L = np.hstack((L, np.atleast_2d(zer).T))
            L = np.vstack((L, zer1))
            r = np.vstack((r + bf[i] * v, 0))
            MW = np.eye(nd)
            MW[-1, -1] = 0
            MW[:-1, -1] = -F[i] * v.ravel()
        elif ~sliding_bc[i] & penetration_bc[i]: # In contact and sticking
            mw = -F[i] * u_hat[:-1, i].ravel('F') / bf[i]
            L = np.eye(nd)
            MW = np.zeros((nd, nd))
            MW[:-1, -1] = mw
            r = np.hstack((u_hat[:-1, i], 0)).T
        elif  ~penetration_bc[i]: # not in contact
            L = np.zeros((nd, nd))
            MW = np.eye(nd)
            r = np.zeros(nd)
        else: #should never happen
            raise AssertionError('Should not get here')

        # Rotation
        L = L.dot(M_inv[:, :, i])
        MW = MW.dot(M_inv[:, :, i])
        # Scale equations (helps iterative solver)
        w_diag = np.diag(L) + np.diag(MW)
        W_inv = np.diag(1/w_diag)
        L = W_inv.dot(L)
        MW = W_inv.dot(MW)
        r = r.ravel() / w_diag
        # Append
        robin_weight.append(L)
        mortar_weight.append(MW)
        rhs = np.hstack((rhs, r))
    return mortar_weight, robin_weight, rhs


def contact_tresca(mg, T, u, bf, ct, cn, M_inv):
    """
    Calculate the semi-smooth Newton step for the Tresca contact problem.
    The Newton step will appare in the elastic equations as a Robin condition:
    
    mortar_weight * lam + robin_weight * [u] = rhs

    where mortar_weight and robin_weight are matrices, lam the Lagrange multiplier,
    [u] the displacement jump and rhs a right hand side vector.

    Arguments:
        mg: mortar grid
        T: Lagrange multiplier
        u: displacement jump
        bf: Friction bound
        ct: Numerical parameter
        cn: Numerical parameter
        M_inv: Rotation matrix to decompose variables into normal and tangential parts.

    Returns:
       mortar_weight
       robin_weight
       rhs
    """
    r_full = np.zeros((mg.dim + 1, mg.num_cells))
    robin_weight = []
    mortar_weight = []
    rhs = np.array([])
    # Change coordinate system to the one alligned to the fractures
    T_hat = np.sum(M_inv * T, axis=1)
    u_hat = np.sum(M_inv * u, axis=1)

    # Find contact and sliding region
    penetration_bc = active_penetration(T_hat[-1], u_hat[-1], cn)
    sliding_bc = active_sliding(T_hat[:-1], u_hat[:-1], bf, ct)

    zer = np.array([0]*mg.dim)
    zer1 = np.array([0]*(mg.dim + 1))
    zer1[-1] = 1
    for i in range(T_hat.shape[1]):
        if sliding_bc[i] & penetration_bc[i]: # in contact and sliding
            L, r, _ = L_r(T_hat[:-1, i], u_hat[:-1, i], bf[i], ct)
            L = np.hstack((L, np.atleast_2d(zer).T))
            L = np.vstack((L, zer1))
            r = np.vstack((r, 0))
            MW = np.eye(mg.dim + 1)
            MW[-1, -1] = 0
        elif ~sliding_bc[i] & penetration_bc[i]: # In contact and sticking
            L = np.eye(mg.dim + 1)
            MW = np.zeros((mg.dim + 1, mg.dim + 1))
            r = np.zeros(mg.dim + 1)
        elif  ~penetration_bc[i]: # not in contact
            L = np.zeros((mg.dim + 1, mg.dim + 1))
            MW = np.eye(mg.dim + 1)
            r = np.zeros(mg.dim + 1)
        else: #should never happen
            raise AssertionError('Should not get here')
        robin_weight.append(L.dot(M_inv[:, :, i]))
        mortar_weight.append(MW.dot(M_inv[:, :, i]))
        rhs = np.hstack((rhs, r.ravel()))
    return mortar_weight, robin_weight, rhs


# Active and inactive boundary faces
def active_sliding(Tt, ut, bf, ct):
    return l2(-Tt + ct * ut) - bf > 1e-10


def active_penetration(Tn, un, cn):
    tol = 1e-8 * cn
    return (-Tn  +   cn * un) > tol


# Below here are different help function for calculating the Newton step
def ef(Tt, cut, bf):
    return bf / l2(-Tt + cut)


def Ff(Tt, cut, bf):
    denominator = max(bf, l2(-Tt)) * l2(-Tt + cut)
    # To avoid issues with shapes we multiply the scalar denominator (scalar for each face)
    # into the Lagraniane multiplyer
    numerator = -Tt.dot((-Tt + cut).T)
    return numerator / denominator


def M(Tt, cut, bf):
    Id = np.eye(Tt.shape[0])
    return ef(Tt, cut, bf) * (Id - Ff(Tt, cut, bf))


def hf(Tt, cut, bf):
    return ef(Tt, cut, bf) * Ff(Tt, cut, bf).dot(-Tt + cut)


def L_r(Tt, ut, bf, c):
    if Tt.ndim<=1:
        Tt = np.atleast_2d(Tt).T
        ut = np.atleast_2d(ut).T
    cut = c * ut
    Id = np.eye(Tt.shape[0])
    if bf <= 1e-10:
        return 0*Id, bf * np.ones((Id.shape[0], 1)), (-Tt + cut) / l2(-Tt + cut)
    _M = M(Tt, cut, bf)
    alpha = -Tt.T.dot(-Tt + cut) / (l2(-Tt) * l2(-Tt + cut))
    delta = min(l2(-Tt) / bf, 1)
    if alpha < 0:
        beta = 1 / (1 - alpha * delta)
    else:
        beta = 1
    IdM_inv = np.linalg.inv(Id - beta * _M)

    v = IdM_inv.dot(-Tt + cut) / l2(-Tt + cut)
    return c * (IdM_inv - Id), -IdM_inv.dot(hf(Tt, cut, bf)), v
