import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras import backend as K


def build_decoder_epg(config):
    """
    Creates a Keras decoder model.
    """
    t2s_input = tf.keras.layers.Input(shape=config['num_classes'], name='t2s_input')
    amps_input = tf.keras.layers.Input(shape=config['num_classes'], name='amps_input')
    angle_input = tf.keras.layers.Input(shape=(1,), name='angle_input')
    signal_output = decoder_epg(config)(inputs=[t2s_input, amps_input, angle_input])
    model = tf.keras.models.Model(
        inputs=[t2s_input, amps_input, angle_input], 
        outputs=signal_output, 
        name=config['name'],
        )
    
    return model


class decoder_epg(Layer):
    """
    Keras layer for the signal_model_epg function
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
    t2s, t1s, amps, angle, nte, delta_te, snr_range = args
    amps = amps[:,:,tf.newaxis] # vectorize to multiply with kernel matrix below
    
    # calculate the kernel matrix for the fitting and generate the signal 
    kernel_matrix = construct_kernel_epg(nte, delta_te, angle, t2s, t1s)
    print(f"++++++++++++{kernel_matrix.shape}")
    print(f"++++++++++++++++{amps.shape}")
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


def construct_kernel_epg(nte, delta_te, angle, t2s, t1s):
    
    kernel_matrix = []
    for i in range(3):
        kernel_col = epg_signal(nte, delta_te, angle[:], t2s[:,i], t1s[:,i])
        kernel_matrix.append(kernel_col)
    kernel_matrix = tf.stack(kernel_matrix, axis=1)
    # print(f"+++++++++++{kernel_matrix.shape}") 
    return kernel_matrix 

    # # make the scalar inputs iterable
    # # t2s = tf.transpose(t2s)
    # # t1s = tf.transpose(t1s)
    # # angle = tf.transpose(angle)
    # # nte = tf.fill(tf.shape(t2s), nte) 
    # # delta_te = tf.fill(tf.shape(t2s), delta_te)
    # angle = tf.broadcast_to(angle, tf.shape(t2s))


    # def func(args):
    #     t2, t1, angle = args
    #     return epg_signal(nte, delta_te, angle, t2, t1)
    
    # kernel_matrix = tf.map_fn(
    #     fn=func,
    #     elems=(t2s, t1s, angle),
    #     dtype=tf.float32,
    # )
    
    # # kernel_matrix = tf.transpose(kernel_matrix)
    # print(f"+++++++++++{kernel_matrix.shape}") 
    # return kernel_matrix

    
    
    # kernel_matrix=tf.zeros([1, nte, 1])
    # for i in range (3):
    #     kernel_matrix_col = tf.map_fn(
    #         fn=func, 
    #         elems=(t2s[:,i], t1s[:,i], angle[:]), 
    #         dtype=tf.float32)
    #     kernel_matrix_col = kernel_matrix_col[:,:,tf.newaxis]
    #     print(f"+++++++++++{kernel_matrix_col.shape}")
    #     print(f"+++++++++++{kernel_matrix.shape}")
    #     kernel_matrix = tf.concat([kernel_matrix, kernel_matrix_col], axis=2)
    # print(f"+++++++++++{kernel_matrix.shape}")
    
    # return kernel_matrix[:,:,1:] 
   


def epg_signal(nte, delta_te, angle, t2, t1):
    # nte, delta_te, angle, t2, t1 = args
    return tf.zeros(nte)





import yaml
if __name__ == "__main__":
   
    config_path = 'configs/defaults_epg.yml' 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    decoder = build_decoder_epg(config['model']['decoder'])
    decoder.summary()
    #print(decoder.get_config()['layers'][3]['config'])
    
    t2s = np.array([[0.01, 0.05, 0.25], [0.01, 0.05, 0.25]], dtype=np.float32)
    amps = np.array([[0.3, 0.5, 0.2], [0.3, 0.5, 0.2]], dtype=np.float32)
    angles = np.array([[3.14], [3.14]], dtype=np.float32)
    
    print(decoder([t2s, amps, angles]))