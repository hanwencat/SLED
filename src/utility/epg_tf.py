import tensorflow as tf
from keras import backend as K

def epg_signal(etl, delta_te, alpha, t2, t1):
    """ 
    Generate multi-echo signals using epg algorithm
    Based on https://doi.org/10.1002/mrm.23157 and UBC matlab code
    """
    
    # construct initial magnetization after 90 degree RF excitation 
    m0 = tf.reshape([1], [1, 1]) 
    m0 = tf.cast(m0, dtype=tf.complex64)
    M0 = tf.zeros([3*etl-1, 1], dtype=tf.complex64)
    M0 = tf.concat([m0, M0], axis=0) # tf object does not support item assignment, so concat is used

    # create relaxation, rotation, transition matrices for epg operations 
    E = relax(etl, delta_te, t2, t1)
    R = rf_rotate(alpha, etl)
    T = transition(etl)

    # iterate flip_relax_seq for each refocusing RF
    echoes = []
    for i in range(etl):
        M0, echo = flip_relax_seq(M0, E, R, T)
        echoes.append(echo)
    
    return tf.squeeze(echoes)


def rf_rotate(alpha, etl):
    """Compute the rotation matrix after RF refocus pulse of angle alpha"""
    
    alpha = tf.squeeze(alpha)
    rotate_real = [[K.cos(alpha/2)**2, K.sin(alpha/2)**2, 0],
                   [K.sin(alpha/2)**2, K.cos(alpha/2)**2, 0],
                   [0, 0, K.cos(alpha)]]
    rotate_complex = [[0, 0, -K.sin(alpha)],
                      [0, 0, K.sin(alpha)],
                      [-0.5*K.sin(alpha), 0.5*K.sin(alpha), 0]]
    rotate = tf.complex(rotate_real, rotate_complex)
    
    R = tf.experimental.numpy.kron(tf.eye(etl,etl), rotate)
    
    return R


def transition(etl):
    """Construct the state transition matrix after each refocusing pulse"""
    
    # F1* --> F1
    x0 = tf.constant(1, shape=[1,], dtype=tf.int64)
    y0 = tf.constant(2, shape=[1,], dtype=tf.int64)
    #v0 = tf.constant(E2, shape=[1,])
    v0 = tf.reshape([1.], shape=[1,])
    
    # F(n)* --> F(n)
    x1 = tf.range(2, 3*etl-3, 3, dtype=tf.int64)
    y1 = tf.range(5, 3*etl, 3, dtype=tf.int64)
    #v1 = E2*tf.ones([etl-1,])
    v1 = 1.*tf.ones([etl-1,])
    
    # F(n) --> F(n+1)
    x2 = tf.range(4, 3*etl-1, 3, dtype=tf.int64)
    y2 = tf.range(1, 3*etl-4, 3, dtype=tf.int64)
    v2 = 1.*tf.ones([etl-1,])

    # Z(n) --> Z(n)
    x3 = tf.range(3, 3*etl+1, 3, dtype=tf.int64)
    y3 = tf.range(3, 3*etl+1, 3, dtype=tf.int64)
    v3 = 1.*tf.ones([etl,])
    
    x = tf.concat([x0,x1,x2,x3],axis=0)
    y = tf.concat([y0,y1,y2,y3],axis=0)
    v = tf.concat([v0,v1,v2,v3],axis=0)

    # transition matrix (indices need to minus 1 because of matlab to python indices conversion)
    T = tf.sparse.SparseTensor(
        indices=tf.stack([x-1,y-1], axis=1), 
        values=v, 
        dense_shape=[3*etl, 3*etl],
        )
    T = tf.sparse.to_dense(tf.sparse.reorder(T))
    T = tf.cast(T, dtype=tf.complex64)
    
    return T


def relax(etl, delta_te, t2, t1):
    """Compute the relaxation matrix after each refocusing pulse"""
    
    E2 = K.exp(-0.5*delta_te/t2)
    E1 = K.exp(-0.5*delta_te/t1)
    relax = [[E2, 0, 0],
             [0, E2, 0],
             [0, 0, E1]]
    E = tf.experimental.numpy.kron(tf.eye(etl, etl), relax)
    E = tf.cast(E, dtype=tf.complex64)
    return E


def flip_relax_seq(M, E, R, T):
    """ 
    Combine 3 operations during each delta_te: 
    relax (E), rotate & transition (R & T), and relax (E)
    """

    M = tf.matmul(E, tf.matmul(T, tf.matmul(R, tf.matmul(E, M))))
    echo = abs(M[0])
    return M, echo