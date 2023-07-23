import numpy as np
import tensorflow as tf
from keras import layers
from keras import backend as K


def signal_model_epg(args):
    """
    signal model (arbitrary number of pools) for multi-echo gradient echo MWI data
    """
    
    # load parameters
    t2s, t1s, amps, angles, te, snr_range = args
    amps = amps[:,:,tf.newaxis] # vectorize to multiply with kernel matrix below
    
    # calculate the kernel matrix for the fitting and generate the signal 
    kernel_matrix = construct_kernel_epg(te, t2s, t1s, angles)
    signal = tf.squeeze(tf.linalg.matmul(kernel_matrix, amps),axis=-1)
    # signal = tf.linalg.matmul(kernel_matrix, amps)

    if snr_range == []:
        return signal
    else:
        # random noise
        snr = tf.random.uniform((tf.shape(signal)[0],1), snr_range[0], snr_range[1]) 
        # use the mean intensity of the first echo as the reference
        scale_factor = tf.reduce_mean(signal, 0)[0] 
        # calculate variance (https://www.statisticshowto.com/rayleigh-distribution/)
        variance = scale_factor*1/(snr * np.sqrt(np.pi/2))
        noise_real = tf.random.normal(tf.shape(signal), 0, variance) # tf.shape used here to handle 'None' shape
        noise_img = tf.random.normal(tf.shape(signal), 0, variance)
        noisy_signal = ((noise_real+signal)**2 + noise_img**2)**0.5 
        
        return noisy_signal 


def construct_kernel_epg(te, t2s, t1s, angles):
    """construct kernel matrix using EPG algorithm"""
    
    # calculate initial parameters for EPG signal production
    etl = te.shape[0]
    delta_te = te[1] - te[0]
    # print(f"t2s shape:{t2s.shape}")
    # print(f"angles shape:{angles.shape}")
    # m = t2s.shape[0]
    # n = t2s.shape[1]
    # kernel_matrix = []
    
    # # iterate each pair of t2 and t1 times to produce the corresponding signal
    # for i in range(m):
    #     for j in range(n):
    #         kernel_matrix.append(
    #             epg_signal(etl, angles[i,0], delta_te, t2s[i,j], t1s[i,j])
    #         )
    # # cast kernel_matrix to its desired shape 
    # kernel_matrix = tf.transpose(
    #     tf.reshape(tf.concat(kernel_matrix, axis=0),[m, n, etl]),
    #     perm=[0,2,1],
    # )
    def func(args):
        t2, t1, angle = args
        # print(f" t2:{t2}\n t1:{t1}\n angle:{angle}")
        return epg_signal(etl, angle, delta_te, t2, t1)

    kernel_my = tf.map_fn(
        fn=func,
        elems=(t2s[:,0], t1s[:,0], angles[:]),
        dtype=tf.float32,
    )
    kernel_my = kernel_my[:,:,tf.newaxis]

    kernel_ie = tf.map_fn(
        fn=func,
        elems=(t2s[:,1], t1s[:,1], angles[:]),
        dtype=tf.float32,
    )
    kernel_ie = kernel_ie[:,:,tf.newaxis]

    kernel_fr = tf.map_fn(
        fn=func,
        elems=(t2s[:,2], t1s[:,2], angles[:]),
        dtype=tf.float32,
    )
    kernel_fr = kernel_fr[:,:,tf.newaxis]

    kernel_matrix = tf.concat([kernel_my, kernel_ie, kernel_fr], axis=2)

    return kernel_matrix


def epg_signal(etl, alpha, delta_te, t2, t1):
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
    
#     M, echo_0 = flip_relax_seq(M0, E, R, T)
#     M, echo_1 = flip_relax_seq(M, E, R, T)
#     M, echo_2 = flip_relax_seq(M, E, R, T)
#     M, echo_3 = flip_relax_seq(M, E, R, T)
#     M, echo_4 = flip_relax_seq(M, E, R, T)
#     M, echo_5 = flip_relax_seq(M, E, R, T)
#     M, echo_6 = flip_relax_seq(M, E, R, T)
#     M, echo_7 = flip_relax_seq(M, E, R, T)
#     M, echo_8 = flip_relax_seq(M, E, R, T)
#     M, echo_9 = flip_relax_seq(M, E, R, T)
#     M, echo_10 = flip_relax_seq(M, E, R, T)
#     M, echo_11 = flip_relax_seq(M, E, R, T)
#     M, echo_12 = flip_relax_seq(M, E, R, T)
#     M, echo_13 = flip_relax_seq(M, E, R, T)
#     M, echo_14 = flip_relax_seq(M, E, R, T)
#     M, echo_15 = flip_relax_seq(M, E, R, T)
#     M, echo_16 = flip_relax_seq(M, E, R, T)
#     M, echo_17 = flip_relax_seq(M, E, R, T)
#     M, echo_18 = flip_relax_seq(M, E, R, T)
#     M, echo_19 = flip_relax_seq(M, E, R, T)
#     M, echo_20 = flip_relax_seq(M, E, R, T)
#     M, echo_21 = flip_relax_seq(M, E, R, T)
#     M, echo_22 = flip_relax_seq(M, E, R, T)
#     M, echo_23 = flip_relax_seq(M, E, R, T)
#     M, echo_24 = flip_relax_seq(M, E, R, T)
#     M, echo_25 = flip_relax_seq(M, E, R, T)
#     M, echo_26 = flip_relax_seq(M, E, R, T)
#     M, echo_27 = flip_relax_seq(M, E, R, T)
#     M, echo_28 = flip_relax_seq(M, E, R, T)
#     M, echo_29 = flip_relax_seq(M, E, R, T)
#     M, echo_30 = flip_relax_seq(M, E, R, T)
#     M, echo_31 = flip_relax_seq(M, E, R, T)
#     M, echo_32 = flip_relax_seq(M, E, R, T)
#     M, echo_33 = flip_relax_seq(M, E, R, T)
#     M, echo_34 = flip_relax_seq(M, E, R, T)
#     M, echo_35 = flip_relax_seq(M, E, R, T)
#     M, echo_36 = flip_relax_seq(M, E, R, T)
#     M, echo_37 = flip_relax_seq(M, E, R, T)
#     M, echo_38 = flip_relax_seq(M, E, R, T)
#     M, echo_39 = flip_relax_seq(M, E, R, T)

#     echoes = tf.concat([echo_0, echo_1, echo_2, echo_3, echo_4, echo_5, echo_6, echo_7, echo_8, echo_9,
#                       echo_10, echo_11, echo_12, echo_13, echo_14, echo_15, echo_16, echo_17, echo_18, echo_19,
#                       echo_20, echo_21, echo_22, echo_23, echo_24, echo_25, echo_26, echo_27, echo_28, echo_29,
#                       echo_30, echo_31, echo_32, echo_33, echo_34, echo_35, echo_36, echo_37, echo_38, echo_39], axis=0)

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


@tf.function
@tf.autograph.experimental.do_not_convert
def flip_relax_seq(M, E, R, T):
    """ 
    Combine 3 operations during each delta_te: 
    relax (E), rotate & transition (R & T), and relax (E)
    """

    M = tf.matmul(E, tf.matmul(T, tf.matmul(R, tf.matmul(E, M))))
    echo = abs(M[0])
    return M, echo


# main program
if __name__ == "__main__":
    
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    # create initial parameters for demonstration
    t2s = tf.constant([[0.015, 0.050, 0.20], [0.015, 0.050, 0.20]], dtype=float)
    t1s = tf.constant([[2., 2., 2.], [2., 2., 2.]], dtype=float)
    # t1s = t2s*5
    amps = tf.constant([[0.2, 0.5, 0.3],[0.2, 0.5, 0.3]], dtype=float)
    te = tf.linspace(0.00675, 0.27, 40)
    angles = tf.constant([[180., 180., 180.,],[120., 120., 120.]])/180*np.pi
    snr_range = []

    # produce signals
    signals = layers.Lambda(signal_model_epg)([t2s, t1s, amps, angles, te, snr_range])
    # signals = signal_model_epg([t2s, t1s, amps, angles, te, snr_range])
    print(signals)
   