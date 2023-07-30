import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras import backend as K


def build_decoder_exp(config):
    """
    Creates a Keras decoder model.
    """
    t2s_input = tf.keras.layers.Input(shape=config['num_classes'], name='t2s_input')
    amps_input = tf.keras.layers.Input(shape=config['num_classes'], name='amps_input')
    signal_output = decoder_exp(config)([t2s_input, amps_input])
    model = tf.keras.models.Model(inputs=[t2s_input, amps_input], outputs=signal_output, name=config['name'])
    
    return model


class decoder_exp(Layer):
    """
    Keras layer for the signal_model_exp function
    """
    def __init__(self, config, **kwargs):
        super(decoder_exp, self).__init__(**kwargs)
        self.nte = config['nte']
        self.delta_te = config['delta_te']
        self.te = np.linspace(
            self.delta_te, 
            self.nte*self.delta_te, 
            self.nte,
            dtype=np.float32,
            )
        self.snr_range = config['snr_range']

    def call(self, inputs):
        t2s, amps = inputs
        args = [t2s, amps, self.te, self.snr_range]
        signal = signal_model_exp(args)
        return signal

    def get_config(self):
        config = super(decoder_exp, self).get_config()
        config.update({'nte': self.nte})
        config.update({'delta_te': self.delta_te})
        config.update({'te': self.te})
        config.update({'snr_range': self.snr_range})
        return config


def signal_model_exp(args):
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
    
    # add noise according to the snr range
    if snr_range == ():
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



import yaml
if __name__ == "__main__":
    t2s = np.array([[0.01, 0.05, 0.25], [0.01, 0.05, 0.25]], dtype=np.float32)
    amps = np.array([[0.3, 0.5, 0.2], [0.3, 0.5, 0.2]],dtype=np.float32)
    
    # te = np.linspace(0.01, 0.32, 32, dtype=np.float32)
    # snr_range = (10, 100)
    # input_shape = t2s.shape[1]
    # input_shape = (3)
    #print(signal_model_exp([t2s, amps, te, snr_range]))
    
    # Load hyperparameters from YAML config file
    config_path = 'configs/defaults.yml' 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    decoder = build_decoder_exp(config['model']['decoder'])
    decoder.summary()

    #print(decoder.get_config())
    print(decoder([t2s, amps]))