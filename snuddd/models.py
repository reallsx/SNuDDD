"""Contains cross sections to be used in targets for any model you like."""
from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp

import snuddd.config as config


if typing.TYPE_CHECKING:
    from snuddd.targets import Nucleus, Electron


class Model(ABC):
    """Provides model interface."""

    @abstractmethod
    def nucleus_cross_section_flavour(self, nucleus: Nucleus, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        pass

    @abstractmethod
    def electron_cross_section_flavour(self, electron: Electron, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        pass


class SM(Model):
    """The standard model neutrino scattering behaviour."""

    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour-breakdown cross section for nucleus. Energy in GeV. model_params = g_x, m_A"""
        Z, m_N, Q_nu_N = nucleus.Z, nucleus.mass, nucleus.Q_nu_N
        cs_SM = Q_nu_N ** 2 / 4

        cs_e = _nuclear_prefactor(nucleus, E_R, E_nu) * cs_SM
        cs_mu = cs_e
        cs_tau = cs_e

        return jnp.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function

    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        y = E_R / E_nu

        cs_e = (2 * electron.mass * config.G_F ** 2 / np.pi *
                ((1 + config.g_L) ** 2 + config.g_R ** 2 * (1 - y) ** 2 - (1 + config.g_L) * config.g_R *
                 (config.m_e * y / E_nu)))
        cs_mu = 2 * electron.mass / np.pi * (config.G_F ** 2 * (
                config.g_L ** 2 + config.g_R ** 2 * (1 - y) ** 2 - config.g_L * config.g_R * config.m_e * y / E_nu))
        cs_tau = cs_mu

        return jnp.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function


class GeneralNSI(Model):
    """A general NSI model, which takes a matrix of NSI couplings and angles."""

    def __init__(self, eps_matrix, eta, phi):
        self.eps_matrix = eps_matrix
        self.eta = eta
        self.phi = phi

    @property
    def xi_p(self):
        """Return proton-rotated charged part of the NSI factorisation."""
        return jnp.sqrt(5) * jnp.cos(self.eta) * jnp.cos(self.phi)

    @property
    def xi_n(self):
        """Return the neutron-rotated charged part of the NSI factorisation."""
        return jnp.sqrt(5) * jnp.sin(self.eta)

    @property
    def xi_u(self):
        """Return the up-quark-rotated charged part of the NSI factorisation."""
        return jnp.sqrt(5) / 3 * (2 * jnp.cos(self.eta) * jnp.cos(self.phi) - jnp.sin(self.eta))

    @property
    def xi_d(self):
        """Return the down-quark-rotated charged part of the NSI factorisation."""
        return jnp.sqrt(5) / 3 * (2 * jnp.sin(self.eta) - jnp.cos(self.eta) * jnp.cos(self.phi))

    @property
    def xi_e(self):
        """Return the electron-rotated charged part of the NSI factorisation."""
        return jnp.sqrt(5) * jnp.cos(self.eta) * jnp.sin(self.phi)

    def G_nucleus_coupling_matrix(self, nucleus):
        """Return the G coupling matrix."""

        return (self.xi_p * nucleus.Z + self.xi_n * nucleus.N) * self.eps_matrix

    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour cross section matrix. Eneregy in GeV"""

        Z, N, m_N, Q_nu_N = nucleus.Z, nucleus.N, nucleus.mass, nucleus.Q_nu_N

        G_matrix = self.G_nucleus_coupling_matrix(nucleus)

        cs_sm = Q_nu_N ** 2 / 4 * jnp.diag(jnp.array([1,1,1]))
        cs_int = - Q_nu_N * G_matrix.real
        cs_bsm = jnp.matmul(G_matrix, G_matrix.conjugate())

        return jnp.tensordot(_nuclear_prefactor(nucleus, E_R, E_nu),
                                 (cs_sm + cs_int + cs_bsm), axes = ((),()))


    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        """Return flavour cross section matrix. Energies in GeV."""

        eps_matrix, eta, phi =  self.eps_matrix, self.eta, self.phi  # This is the change!

        g_L = config.g_L
        g_R = config.g_R
        xi_e = self.xi_e
        xi_p = self.xi_p
        xi_n = self.xi_n 

        GL_matrix = (jnp.array([[1,0,0],[0,0,0],[0,0,0]])   
                    + g_L * jnp.diag(jnp.array([1,1,1]))
                    + 0.5*eps_matrix*xi_e)

        GR_matrix = (g_R*jnp.diag(jnp.array([1,1,1]))
                     + 0.5 * eps_matrix*xi_e)

        prefactor  = 2 * config.G_F ** 2 * config.m_e / np.pi

        Lterm_shape_enhancement = (E_R / E_nu).shape  # To get Lterm to be correct shape for sum

        Lterm = jnp.tensordot(jnp.ones(Lterm_shape_enhancement), jnp.matmul(GL_matrix, GL_matrix.conjugate()), axes = ((),()))
        Rterm = jnp.tensordot((1 - E_R/E_nu)**2, jnp.matmul(GR_matrix, GR_matrix.conjugate()), axes = ((),()))
        LRterm = jnp.tensordot(((config.m_e * E_R)/(2 * E_nu**2)), (jnp.matmul(GL_matrix, GR_matrix.conjugate())
                                    + jnp.matmul(GR_matrix, GL_matrix.conjugate())), axes = ((),()))

        return prefactor * (Lterm + Rterm - LRterm)

        
def _nuclear_prefactor(nucleus, E_R, E_nu):
    """Return commonly used nuclear model prefactor."""
    F_helm = nucleus.form_factor(E_R)
    return config.G_F ** 2 / np.pi * nucleus.mass * (
            1 - nucleus.mass * E_R / (2 * E_nu ** 2)) * F_helm ** 2



