import numpy as np
import tensorflow as tf
import keras
from keras import layers
# from keras import regularizers
from keras import backend as K
import nibabel as nib
# import scipy


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
    batch_norm=False,
    ):
    """
    build sled model
    """
   
    # clear session and initialize input
    keras.backend.clear_session()
    x = keras.Input(shape=(te.shape[0],))
    
    # use 3 NNs to encode mgre data and output each pool's t2 time
    t2_my = nn_builder(x, nn_layers_t2s, batch_norm)
    t2_ie = nn_builder(x, nn_layers_t2s, batch_norm)
    t2_fr = nn_builder(x, nn_layers_t2s, batch_norm)

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
    """
    signal model (arbitrary number of pools) for multi-echo gradient echo MWI data
    """
    
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


def train_model(model, data, epochs = 30, return_history=False):
    """compile and fit sled model"""

    model.compile(
        optimizer=keras.optimizers.Adamax(
            learning_rate=0.001, 
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
        )
    
    if return_history == True:
        return history



def latent_maps(encoder, data, latent_dim):
    """apply the trained encoder and output latent parameter maps"""
    
    # flatten data and use encoder to get latent parameters
    data_flat = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])
    t2s_flat, amps_flat = encoder.predict(data_flat)
    
    # normalize amps to have sum of 1
    amps_flat = amps_flat/np.sum(amps_flat, axis=1).reshape(data_flat.shape[0], 1)

    # get rid of nan 
    t2s_flat= np.nan_to_num(t2s_flat)
    amps_flat = np.nan_to_num(amps_flat)

    # reshape to original shape
    t2s_maps = t2s_flat.reshape(
        np.append(np.asarray(data.shape[:-1]), latent_dim)
        )
    amps_maps = amps_flat.reshape(
        np.append(np.asarray(data.shape[:-1]), latent_dim)
        )  

    return t2s_maps, amps_maps      



def load_data(image_path, mask_path, ETL=24):
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



# main program
if __name__ == "__main__":
    # load data and preprocess
    data_path = '/export01/data/Hanwen/SAME-ECOS/SAME-ECOS_code/mGRE_T2star.nii'
    mask_path = '/export01/data/Hanwen/SAME-ECOS/SAME-ECOS_code/mGRE_T2star_bet_mask.nii.gz'
    image, mask, data = load_data(data_path, mask_path)
    data_flat, data_flat_norm = preprocess_data(data)

    # define parameters 
    te = tf.range(0.002, 0.05, 0.002) 
    nn_layers_t2s = [256, 128, 1]
    nn_layers_amps = [256, 256, 3]
    range_t2_my = [0.005, 0.015]
    range_t2_ie = [0.045, 0.06]
    range_t2_fr = [0.1, 0.2]
    snr_range = [50., 500.]
    # amps_scaling = 8
    amps_scaling = np.quantile(data_flat, 0.99, axis=0)[0]
    print(f"amplitude scaling factor = {amps_scaling}")

    # construct sled    
    encoder, sled = sled_builder(
        te,
        nn_layers_t2s,
        nn_layers_amps,
        range_t2_my,
        range_t2_ie,
        range_t2_fr,
        snr_range,
        amps_scaling,
        batch_norm=False,
        )
    sled.summary()

    # train sled model
    train_model(sled, data_flat, epochs=20)

    # produce metric maps
    t2s_maps, amps_maps = latent_maps(encoder, data, latent_dim=3)
    print(t2s_maps.shape, amps_maps.shape)












