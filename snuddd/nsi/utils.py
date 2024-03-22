"""Useful functions for NSI work"""
import numpy as np
import jax.numpy as jnp
from snuddd.models import GeneralNSI


def invert_nsi_u(eps_up, eta, phi):
    """Given the full, unfactorised NSI matrix element for the up-quark, return the flavour-specific and
    charge independent matrix element, eps_{alpha beta}^{eta}."""
    return eps_up / GeneralNSI(0, eta, phi).xi_u


def invert_nsi_d(eps_d, eta, phi):
    """Given the full, unfactorised NSI matrix element for the down-quark, return the flavour-specific and
    charge independent matrix element, eps_{alpha beta}^{eta}."""
    return eps_d / GeneralNSI(0, eta, phi).xi_d

def invert_nsi_e(eps_e, eta, phi):
    """Given the full, unfactorised NSI matrix element for the electron, return the flavour-specific and
    charge independent matrix element, eps_{alpha beta}^{eta}."""
    return eps_e / GeneralNSI(0, eta, phi).xi_e


def invert_nsi_p(eps_p, eta, phi):
    """Given the full, unfactorised NSI matrix element for the proton, return the flavour-specific and
    charge independent matrix element, eps_{alpha beta}^{eta}."""
    return eps_p / (jnp.sqrt(5) * jnp.cos(eta) * jnp.cos(phi))


def eps_matrix_sym(mat):
    """Return the symmetric NSI matrix given an UPPER triangular flavour matrix."""
    return mat - jnp.tril(mat, -1) + jnp.triu(mat, 1).T


def map_eta(eta_prime, phi):
    """Transform the charged to neutral angle from the p-n plane to
    the full 3D framework.
    """

    return jnp.arctan((jnp.sin(phi) + jnp.cos(phi)) * jnp.tan(eta_prime))


def map_eps(eps_prime, eta_prime, phi, c=True):
    """Transform the NSI matrix element from the p-n plane to the full 3D
    Framework
    """

    eta = map_eta(eta_prime, phi)

    if c or eta_prime == 0:
        return eps_prime * (jnp.cos(eta_prime) / jnp.cos(eta) /
                            (jnp.sin(phi) + jnp.cos(phi)))
    return eps_prime * jnp.sin(eta_prime) / jnp.sin(eta)
