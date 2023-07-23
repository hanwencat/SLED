import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import backend as K
import nibabel as nib


def nn_builder(x, nn_layers, batch_norm=False):
    """
    build neural networks
    """
    
    for nn_layer in nn_layers:
        x = layers.Dense(
            nn_layer,
            activation='sigmoid',
            kernel_initializer=keras.initializers.lecun_normal(seed=None),
        )(x)
        
        if batch_norm == True:
            x = layers.BatchNormalization()(x)

    return x


def construct_sled_vary_t2s(
    latent_dim,
    te,
    nn_layers_t2s,
    nn_layers_amps,
    nn_layers_angles,
    range_t2_my,
    range_t2_ie,
    range_t2_fr,
    range_angles,
    snr_range,
    t1_value,
    amps_scaling,
    batch_norm=False,
    ):
    """
    build sled model with variable t2s
    """
   
    # clear session and initialize input
    keras.backend.clear_session()
    x = keras.Input(shape=(te.shape[0],))
    
    # use 3 NNs to encode input data and output each pool's t2 time
    t2_my = nn_builder(x, nn_layers_t2s, batch_norm)
    t2_ie = nn_builder(x, nn_layers_t2s, batch_norm)
    t2_fr = nn_builder(x, nn_layers_t2s, batch_norm)

    # constrain t2s in corresponding ranges
    t2_my = t2_my * (range_t2_my[1] - range_t2_my[0]) + range_t2_my[0]
    t2_ie = t2_ie * (range_t2_ie[1] - range_t2_ie[0]) + range_t2_ie[0]
    t2_fr = t2_fr * (range_t2_fr[1] - range_t2_fr[0]) + range_t2_fr[0]

    # group 3 t2 times into t2s
    t2s = tf.concat([t2_my, t2_ie, t2_fr], axis=1)
    
    # define t1 empirically
    if t1_value == []:
        t1s = t2s*5
    else:
        t1s = tf.constant(t1_value, shape=t2s.shape)

    # use 1 NN to encode input data and output amplitudes of 3 pools
    amps = nn_builder(x, nn_layers_amps, batch_norm=False)
    
    # use 1 NN to encode input data and output refocusing flip angles
    angles = nn_builder(x, nn_layers_angles, batch_norm=False)
    
    # constrain angles in the corresponding range
    angles = angles * (range_angles[1] - range_angles[0]) + range_angles[0]

    # scale the amplitude
    amps = amps * amps_scaling
    
    encoder =  keras.Model(x, [t2s, amps, angles], name = "encoder")
    y = layers.Lambda(signal_model_epg)([latent_dim, t2s, t1s, amps, angles, te, snr_range])
    sled = keras.Model(x, y, name = "sled")

    return encoder, sled


def construct_sled_fix_t2s(
    latent_dim,
    te,
    t2s,
    nn_layers_amps,
    nn_layers_angles,
    range_angles,
    snr_range,
    t1_value,
    amps_scaling,
    batch_norm=False,
    ):
    """
    build sled model with fixed t2s
    """
   
    # clear session and initialize input
    keras.backend.clear_session()
    x = keras.Input(shape=(te.shape[0],))
    
    # define t1 imperially
    if t1_value == []:
        t1s = t2s*5
    else:
        t1s = tf.constant(t1_value, shape=t2s.shape)

    # use 1 NN to encode input data and output amplitudes of 3 pools
    amps = nn_builder(x, nn_layers_amps, batch_norm)
    
    # use 1 NN to encode input data and output refocusing flip angles
    angles = nn_builder(x, nn_layers_angles, batch_norm)
    
    # constrain angles in the corresponding range
    angles = angles * (range_angles[1] - range_angles[0]) + range_angles[0]

    # scale the amplitude
    amps = amps * amps_scaling
    
    encoder =  keras.Model(x, [amps, angles], name = "encoder")
    y = layers.Lambda(signal_model_epg)([latent_dim, t2s, t1s, amps, angles, te, snr_range])
    sled = keras.Model(x, y, name = "sled")

    return encoder, sled


def signal_model_epg(args):
    """
    signal model (arbitrary number of pools) for multi-echo gradient echo MWI data
    """
    
    # load parameters
    latent_dim, t2s, t1s, amps, angles, te, snr_range = args
    amps = amps[:,:,tf.newaxis] # vectorize to multiply with kernel matrix below
    
    # calculate the kernel matrix for the fitting and generate the signal 
    kernel_matrix = construct_kernel_epg(latent_dim, te, t2s, t1s, angles)
    signal = tf.squeeze(tf.linalg.matmul(kernel_matrix, amps), axis=-1)
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


def construct_kernel_epg(latent_dim, te, t2s, t1s, angles):
    """construct kernel matrix using EPG algorithm"""
    
    # calculate initial parameters for EPG signal production
    etl = te.shape[0]
    delta_te = te[1] - te[0]
    
    def func(args):
        t2, t1, angle = args
        return epg_signal(etl, angle, delta_te, t2, t1)

    kernel_matrix=tf.zeros([1, etl, 1])
    for i in range (latent_dim):
        kernel_matrix_col = tf.map_fn(
            fn=func, 
            elems=(t2s[:,i], t1s[:,i], angles[:]), 
            dtype=tf.float32)
        kernel_matrix_col = kernel_matrix_col[:,:,tf.newaxis]
        kernel_matrix = tf.concat([kernel_matrix, kernel_matrix_col], axis=2)
    
    return kernel_matrix[:,:,1:]


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


def train_model(model, data, epochs = 30, return_history=False, verbose=2):
    """
    compile and fit sled model
    use verbose=2 if return_history=True (https://github.com/tensorflow/tensorflow/issues/48033)
    """

    model.compile(
        optimizer=keras.optimizers.Adamax(
            learning_rate=0.0001, 
            clipnorm=1, 
            clipvalue=1,
            ), 
        loss='mse',
    )

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=15),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3),
    ]

    history = model.fit(
        data, 
        data, 
        epochs=epochs, 
        batch_size=256, 
        callbacks=callbacks_list,
        verbose=verbose,
        )
    
    if return_history == True:
        return history


def latent_maps(encoder, data, latent_dim):
    """apply the trained encoder and output latent parameter maps"""
    
    # flatten data and use encoder to get latent parameters
    data_flat = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])
    t2s_flat, amps_flat, angles_flat = encoder.predict(data_flat)
    
    # normalize amps to have sum of 1
    amps_flat = amps_flat/np.sum(amps_flat, axis=1).reshape(data_flat.shape[0], 1)

    # get rid of nan 
    t2s_flat= np.nan_to_num(t2s_flat)
    amps_flat = np.nan_to_num(amps_flat)
    angles_flat = np.nan_to_num(angles_flat)

    # reshape to original shape
    t2s_maps = t2s_flat.reshape(
        np.append(np.asarray(data.shape[:-1]), latent_dim)
        )
    amps_maps = amps_flat.reshape(
        np.append(np.asarray(data.shape[:-1]), latent_dim)
        )  
    angles_map = angles_flat.reshape(
        np.append(np.asarray(data.shape[:-1]), 1)
        )

    return t2s_maps, amps_maps, angles_map      


def latent_maps_fix_t2s(encoder, data, latent_dim):
    """apply the trained encoder and output latent parameter maps"""
    
    # flatten data and use encoder to get latent parameters
    data_flat = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])
    amps_flat, angles_flat = encoder.predict(data_flat)
    
    # normalize amps to have sum of 1
    amps_flat = amps_flat/np.sum(amps_flat, axis=1).reshape(data_flat.shape[0], 1)

    # get rid of nan 
    amps_flat = np.nan_to_num(amps_flat)
    angles_flat = np.nan_to_num(angles_flat)

    # reshape to original shape
    amps_maps = amps_flat.reshape(
        np.append(np.asarray(data.shape[:-1]), latent_dim)
        )  
    angles_map = angles_flat.reshape(
        np.append(np.asarray(data.shape[:-1]), 1)
        )

    return amps_maps, angles_map


def load_data(image_path, mask_path, ETL):
    """
    load nifti dataset and brain mask, return masked image as numpy array
    """
    
    image =  nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    # uncomment the next line if mask erosion is needed.
    # mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=3).astype(mask.dtype)
    mask_4d = np.repeat(mask[:, :, :, np.newaxis], ETL, axis=3)
    mask_4d[mask_4d==0] = np.nan
    masked_image = image*mask_4d
    
    return image, mask, masked_image


def preprocess_data(data):
    """
    flaten 4D dataset and normalize
    """

    data_flat = data.reshape(-1, data.shape[-1])
    data_flat = data_flat[~np.isnan(data_flat)].reshape(-1, data_flat.shape[1])
    data_flat_norm = data_flat/(data_flat[:,0].reshape(data_flat.shape[0],1))
    data_flat_norm = data_flat_norm.astype('float32')

    return data_flat, data_flat_norm


def log_t2s(start, stop, num_elements):
    """generate t2 basis in a logarithmic scale"""

    log_range = tf.linspace(
        tf.math.log(start), 
        tf.math.log(stop), 
        num_elements,
        )
    t2s = tf.exp(log_range)

    return t2s[tf.newaxis,:]


# main program
if __name__ == "__main__":
    
    # physical_devices = tf.config.list_physical_devices('GPU') 
    # for device in physical_devices:
    #     tf.config.experimental.set_memory_growth(device, True)
    
    # load data and preprocess
    data_path = '/export01/data/Hanwen/data/mgre_data/WT_F_21/mese_t2.nii.gz'
    mask_path = '/export01/data/Hanwen/data/mgre_data/WT_F_21/WTF21_mese_mask.nii.gz'
    image, mask, data = load_data(data_path, mask_path, ETL=40)
    data_flat, data_flat_norm = preprocess_data(data)
    data_norm = data/data[:,:,:,0, np.newaxis]

    # define parameters 
    latent_dim = 10
    t2s = log_t2s(0.005, 0.5, latent_dim)
    te = tf.linspace(0.00675, 0.27, 40)
    nn_layers_amps = [128, latent_dim]
    nn_layers_angles = [64, 1]
    # range_angles = tf.constant([90., 180.])/180*np.pi
    range_angles = [0.5*np.pi, np.pi]
    snr_range = []
    t1_value = []
    # amps_scaling = 8
    amps_scaling = np.quantile(data_flat_norm, 0.99, axis=0)[0]
    print(f"amplitude scaling factor = {amps_scaling}")

    # construct sled    
    encoder, sled = construct_sled_fix_t2s(
        latent_dim,
        te,
        t2s,
        nn_layers_amps,
        nn_layers_angles,
        range_angles,
        snr_range,
        t1_value,
        amps_scaling,
        batch_norm=False,
        )
    sled.summary()

    # train sled model
    train_model(sled, data_flat_norm, epochs=5)

    # produce metric maps
    amps_maps, angles_map = latent_maps_fix_t2s(encoder, data_norm, latent_dim)
    print(amps_maps.shape, angles_map.shape)