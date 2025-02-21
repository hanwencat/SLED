import numpy as np
import tensorflow as tf
from keras.layers import Layer
from utility.epg_tf import epg_signal
from utility.add_noise_tf import add_rayleigh_noise



def build_decoder_epg(config):
    """
    Creates a Keras decoder model.
    """
    t2s_input = tf.keras.layers.Input(shape=config['num_classes'], name='t2s_input')
    amps_input = tf.keras.layers.Input(shape=config['num_classes'], name='amps_input')
    angle_input = tf.keras.layers.Input(shape=(1), name='angle_input')
    signal_output = decoder_epg(config)(inputs=[t2s_input, amps_input, angle_input])
    model = tf.keras.models.Model(
        inputs=[t2s_input, amps_input, angle_input], 
        outputs=signal_output, 
        name=config['name'],
        )
    
    return model


class decoder_epg(Layer):
    """
    A customized Keras layer to host the signal_model_epg function
    """
    def __init__(self, config, **kwargs):
        super(decoder_epg, self).__init__(**kwargs)
        self.nte = config['nte']
        self.delta_te = config['delta_te']
        self.te = np.linspace(
            self.delta_te, 
            self.nte*self.delta_te, 
            self.nte,
            dtype=np.float32,
            )
        self.snr_range = config['snr_range']
        self.fix_t1s = config['fix_t1s']
        if config['fix_t1s']:
            self.t1s = config['fix_t1s_value'] 
        else:
            self.t1_t2_ratio = config['t1_t2_ratio']

    def call(self, inputs):
        t2s, amps, angle = inputs
        # define arguments
        if self.fix_t1s:
            t1s = tf.fill(tf.shape(t2s), self.t1s)
        else:
            t1s = t2s*self.t1_t2_ratio
        args = [t2s, t1s, amps, angle, self.nte, self.delta_te, self.snr_range]
        signal = signal_model_epg(args)

        return signal

    def get_config(self):
        config = super(decoder_epg, self).get_config()
        config.update({'nte': self.nte})
        config.update({'delta_te': self.delta_te})
        config.update({'te': self.te})
        config.update({'snr_range': self.snr_range})
        config.update({'fix_t1s': self.fix_t1s})
        if config['fix_t1s']:
            config.update({'fix_t1s_value': self.t1s})
        else:
            config.update({'t1_t2_ratio': self.t1_t2_ratio})

        return config


def signal_model_epg(args):
    """
    A wrapper function to generate multi-echo signals using epg algorithm.
    Add noise to the signal if snr_range is provided.
    """
    t2s, t1s, amps, angle, nte, delta_te, snr_range = args
    amps = amps[:,:,tf.newaxis] # vectorize to multiply with kernel matrix below
    
    # calculate the kernel matrix for the fitting and generate the signal 
    kernel_matrix = construct_kernel_epg(nte, delta_te, angle, t2s, t1s)
    signal = tf.squeeze(tf.linalg.matmul(kernel_matrix, amps), axis=-1)

    # add noise according to the snr range
    if snr_range == None:
        return signal
    else:
        return add_rayleigh_noise(signal, snr_range)



def construct_kernel_epg(nte, delta_te, angles, t2s, t1s):
    """A wrapper function to construct the kernel matrix"""
    # make kernel matrix for each latent dimension and concatenate them together
    # the t2s and t1s are of shape (batch_size, latent_dim)
    # the kernel matrix is of shape (batch_size, nte, latent_dim)
    # the angles are of shape (batch_size, 1)

    def func(args): 
        t2, t1, angle = args
        # return epg_signal(nte, delta_te, angle, t2, t1)
        return tf.math.real(epg_signal(nte, delta_te, angle, t2, t1))
    
    kernel_matrix = None
    latent_dim = t2s.shape[1]
    for i in range(latent_dim):
        kernel_matrix_col = tf.map_fn(
            fn=func, 
            elems=(t2s[:,i], t1s[:,i], angles), 
            fn_output_signature=tf.TensorSpec(shape=(nte,), dtype=tf.float32)
            # dtype=tf.float32,
            # dtype=tf.complex64,
            )
        kernel_matrix_col = kernel_matrix_col[:,:,tf.newaxis]
        
        if kernel_matrix is None:
            kernel_matrix = kernel_matrix_col
        else:
            kernel_matrix = tf.concat([kernel_matrix, kernel_matrix_col], axis=2)
    
    return kernel_matrix
    






# import yaml
# from tensorflow.keras.layers import Lambda
# if __name__ == "__main__":
   
#     config_path = 'configs/defaults_epg_param.yml' 
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
    
#     decoder = build_decoder_epg_param(config['model']['decoder'])
#     decoder.summary()
#     # print(decoder.get_config()['layers'][3]['config'])
    
#     t2s = np.array([[0.01, 0.05, 0.25], [0.01, 0.05, 0.25]], dtype=np.float32)
#     amps = np.array([[0.3, 0.5, 0.2], [0.3, 0.5, 0.2]], dtype=np.float32)
#     angles = np.array([[1.57], [3.14]], dtype=np.float32)
    
#     print(decoder([t2s, amps, angles]))