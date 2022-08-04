import numpy as np
import tensorflow as tf
import keras
from keras import layers
# from keras import regularizers
from keras import backend as K


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


def sled_builder(
    te,
    nn_layers_t2s,
    nn_layers_amps,
    range_t2_my,
    range_t2_ie,
    range_t2_fr,
    snr_range,
    amps_scaling,
    ):

    # clear session and initialize input
    keras.backend.clear_session()
    x = keras.Input(shape=(te.shape[0],))
    
    # use 3 NNs to encode mgre data and output each pool's t2 time
    t2_my = nn_builder(x, nn_layers_t2s, batch_norm=False)
    t2_ie = nn_builder(x, nn_layers_t2s, batch_norm=False)
    t2_fr = nn_builder(x, nn_layers_t2s, batch_norm=False)

    # constrain t2s in corresponding ranges
    t2_my = t2_my * (range_t2_my[1] - range_t2_my[0]) + range_t2_my[0]
    t2_ie = t2_ie * (range_t2_ie[1] - range_t2_ie[0]) + range_t2_ie[0]
    t2_fr = t2_fr * (range_t2_fr[1] - range_t2_fr[0]) + range_t2_fr[0]

    # group 3 t2 times into t2s
    t2s = tf.concat([t2_my, t2_ie, t2_fr], axis=1)

    # use 1 NN to encode mgre data and output amplitudes of 3 pools
    amps = nn_builder(x, nn_layers_amps, batch_norm=False)
    
    # scale the amplitude
    amps = amps * amps_scaling
    
    encoder =  keras.Model(x, [t2s, amps], name = "encoder")
    y =  layers.Lambda(signal_model)([t2s, amps, te, snr_range])
    sled = keras.Model(x, y, name = "sled")

    return encoder, sled


def signal_model(args):
    """Signal model (arbitrary number of pools) for multi-echo gradient echo MWI data"""
    # load and vectorize parameters
    t2s, amps, te, snr_range = args
    t2s = t2s[:,tf.newaxis,:]
    amps = amps[:,:,tf.newaxis]
    te = te[tf.newaxis,:,tf.newaxis]
    
    # calculate the kernel matrix for the fitting and generate the signal 
    kernel_matrix = K.exp(-te/t2s)
    signal = tf.squeeze(tf.linalg.matmul(kernel_matrix, amps))
    
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
        


if __name__ == "__main__":
    # define t2 range of each water pool
    te = tf.range(0.002, 0.05, 0.002) 
    nn_layers_t2s = [256, 128, 1]
    nn_layers_amps = [256, 256, 3]
    range_t2_my = [0.005, 0.015]
    range_t2_ie = [0.045, 0.06]
    range_t2_fr = [0.1, 0.2]
    snr_range = [50., 500.]
    
    encoder, sled = sled_builder(
        te,
        nn_layers_t2s,
        nn_layers_amps,
        range_t2_my,
        range_t2_ie,
        range_t2_fr,
        snr_range,
        amps_scaling=1,
        )
    
    print(sled.summary())

    x = tf.random.uniform([10,24])
    t2s, amps = encoder(x)
    y = sled(x)    

    print(t2s.shape, amps.shape, y.shape)












