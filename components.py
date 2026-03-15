import numpy as np


I2  = np.eye(2,  dtype=complex)   
X_spatial = np.array([[0, 1],
                      [1, 0]], dtype=complex)

# Polarization projectors: |H><H| and |V><V|
proj_H = np.array([[1, 0],
                   [0, 0]], dtype=complex)

proj_V = np.array([[0, 0],
                   [0, 1]], dtype=complex)


def make_HWP(theta):
    c = np.cos(2 * theta)
    s = np.sin(2 * theta)
    U_pol = np.array([[ c,  s],
                      [ s, -c]], dtype=complex)
    return np.kron(I2, U_pol)

def make_QWP(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    U_pol = np.array([
        [c**2 + 1j*s**2,   (1 - 1j)*s*c  ],
        [(1 - 1j)*s*c,      s**2 + 1j*c**2]
    ], dtype=complex)
    return np.kron(I2, U_pol)

def make_PS(phi):
    U_pol = np.array([
        [1, 0],
        [0, np.exp(1j * phi)]
    ], dtype=complex)
    return np.kron(I2, U_pol)

def make_BS():
    t = 1.0 / np.sqrt(2)
    U_spatial = np.array([
        [t,    1j*t],
        [1j*t, t   ]
    ], dtype=complex)
    return np.kron(U_spatial, I2)

def make_PBS():
    return np.kron(I2, proj_H) + np.kron(X_spatial, proj_V)

def make_NONE():
    return np.eye(4, dtype=complex)


def is_pol_V(photon_idx):
    """Helper: Returns True if the 4D state index corresponds to Polarization V"""
    return photon_idx % 2 == 1

def get_CK_16(theta, ph1, ph2):
    """16x16 Cross-Kerr: Applies phase theta if both photons are Pol V."""
    diag = np.ones(16, dtype=complex)
    for i in range(16):
        p0_idx = i // 4
        p1_idx = i % 4
        idx_list = [p0_idx, p1_idx]
        if is_pol_V(idx_list[ph1]) and is_pol_V(idx_list[ph2]):
            diag[i] = np.exp(1j * theta)
    return np.diag(diag)

def get_CK_64(theta, ph1, ph2):
    """64x64 Cross-Kerr: The ultimate entangler for the Ancilla pipeline."""
    diag = np.ones(64, dtype=complex)
    for i in range(64):
        p0_idx = i // 16
        p1_idx = (i // 4) % 4
        p2_idx = i % 4
        idx_list = [p0_idx, p1_idx, p2_idx]
        if is_pol_V(idx_list[ph1]) and is_pol_V(idx_list[ph2]):
            diag[i] = np.exp(1j * theta)
    return np.diag(diag)


COMP_FUNCS  = {
    'NONE': make_NONE,
    'HWP': make_HWP,
    'QWP': make_QWP,
    'PS' : make_PS,
    'BS' : make_BS,
    'PBS': make_PBS,
    'CK' : None  
}

NEEDS_PARAM = {'NONE': False, 'HWP': True, 'QWP': True, 'PS': True, 'BS': False, 'PBS': False, 'CK': True}
COMP_NAMES  = list(COMP_FUNCS.keys())