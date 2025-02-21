import numpy as np
import tensorflow as tf

def add_rayleigh_noise(signal, snr_range):
    """Add rayleigh noise to the signal.

    Args:
        signal (a tensor): the pure multi-echo signal
        snr_range (a list): the range of snr

    Returns:
        noisy_signal: signal with rayleigh noise
    """
    # generate random snr in the range
    snr = tf.random.uniform((tf.shape(signal)[0],1), snr_range[0], snr_range[1]) 
    
    # use the mean intensity of the first echo as the scaling factor
    scale_factor = tf.reduce_mean(signal, 0)[0] 
    
    # calculate variance (https://www.statisticshowto.com/rayleigh-distribution/)
    variance = scale_factor*1/(snr * np.sqrt(np.pi/2))
    
    # add rayleigh noise (zero mean gaussian to both real and imaginary parts), then the signal is rician distributed.
    noise_real = tf.random.normal(tf.shape(signal), 0, variance) # tf.shape used here to handle 'None' shape
    noise_img = tf.random.normal(tf.shape(signal), 0, variance)
    noisy_signal = ((noise_real+signal)**2 + noise_img**2)**0.5 
    
    return noisy_signal 