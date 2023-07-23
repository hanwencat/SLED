import numpy as np

def cpmg_signal(etl, alpha, delta_te, t2, t1):
    """ 
    Generate multi-echo cpmg signals using epg algorithm
    Based on https://doi.org/10.1002/mrm.23157 and UBC matlab code
    """
    
    # construct initial magnetization after 90 degree RF excitation 
    m0 = np.reshape([1], [1, 1]) 
    m0 = m0.astype(np.complex64)
    M0 = np.zeros([3*etl-1, 1], dtype=np.complex64)
    M0 = np.concatenate([m0, M0], axis=0)

    # create relaxation, rotation, transition matrices for epg operations 
    E = relax(etl, delta_te, t2, t1)
    R = rf_rotate(alpha, etl)
    T = transition(etl)

    # iterate flip_relax_seq for each refocusing RF
    echoes = np.zeros(etl)
    for i in range(etl):
        M0, echo = cpmg_seq(M0, E, R, T)
        echoes[i] = echo
    
    return np.squeeze(echoes)


def cpmg_seq(M, E, R, T):
    """ 
    Combine 3 operations during each delta_te: 
    relax (E), rotate & transition (R & T), and relax (E)
    """
    
    M = E @ T @ R @ E @ M
    echo = abs(M[0])
    return M, echo


def rf_rotate(alpha, etl):
    """Compute the rotation matrix after RF refocus pulse of angle alpha"""
    
    alpha = np.squeeze(alpha)
    rotate_real = np.array([[np.cos(alpha/2)**2, np.sin(alpha/2)**2, 0],
                            [np.sin(alpha/2)**2, np.cos(alpha/2)**2, 0],
                            [0, 0, np.cos(alpha)]])
    rotate_complex = np.array([[0, 0, -np.sin(alpha)],
                               [0, 0, np.sin(alpha)],
                               [-0.5*np.sin(alpha), 0.5*np.sin(alpha), 0]])
    rotate = np.array(rotate_real, dtype=np.complex128) + 1j * np.array(rotate_complex, dtype=np.complex128)
    
    R = np.kron(np.eye(etl,etl), rotate)
    
    return R


def transition(etl):
    """Construct the state transition matrix after each refocusing pulse"""

    # F1* --> F1
    x0 = np.array([0], dtype=np.int64)
    y0 = np.array([1], dtype=np.int64)
    v0 = np.array([1.], dtype=np.float32)

    # F(n)* --> F(n)
    x1 = np.arange(1, 3*etl-4, 3, dtype=np.int64)
    y1 = np.arange(4, 3*etl-1, 3, dtype=np.int64)
    v1 = np.ones(etl-1, dtype=np.float32)

    # F(n) --> F(n+1)
    x2 = np.arange(3, 3*etl-2, 3, dtype=np.int64)
    y2 = np.arange(0, 3*etl-5, 3, dtype=np.int64)
    v2 = np.ones(etl-1, dtype=np.float32)

    # Z(n) --> Z(n)
    x3 = np.arange(2, 3*etl, 3, dtype=np.int64)
    y3 = np.arange(2, 3*etl, 3, dtype=np.int64)
    v3 = np.ones(etl, dtype=np.float32)

    x = np.concatenate([x0, x1, x2, x3])
    y = np.concatenate([y0, y1, y2, y3])
    v = np.concatenate([v0, v1, v2, v3])

    # transition matrix
    T = np.zeros((3*etl, 3*etl), dtype=np.complex64)
    T[x, y] = v

    return T


def relax(etl, delta_te, t2, t1):
    """Compute the relaxation matrix after each refocusing pulse"""
    
    E2 = np.exp(-0.5*delta_te/t2)
    E1 = np.exp(-0.5*delta_te/t1)
    relax = np.array([[E2, 0, 0],
                      [0, E2, 0],
                      [0, 0, E1]])
    E = np.kron(np.eye(etl, dtype=np.complex64), relax)
    
    return E


# main program
if __name__ == "__main__":
    import time
    start_time = time.time()
    s = cpmg_signal(
        etl=120, 
        alpha=120/180*np.pi, 
        delta_te=0.0041, 
        t2=0.1, 
        t1=1,
        )
    print(f'echoes:\n{s}')
    print(f'Elapsed time: {time.time()- start_time:.2f}')
