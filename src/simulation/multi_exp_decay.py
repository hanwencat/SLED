import numpy as np

def generate_pretrain_data(config):
    """generate synthetic data for pretraining the model"""
    
    # load the parameters from the config file
    num_samples = config['number_of_samples']
    pools = config['number_of_pools']
    t2_ranges = config['t2_ranges']
    amplitude_ranges = config['amplitude_ranges']
    snr_range = config['snr_range']
    
    # get echo times
    first_echo_time = config['first_echo_time']
    echo_spacing = config['echo_spacing']
    num_echoes = config['number_of_echoes']
    echo_times = np.arange(first_echo_time, first_echo_time + num_echoes * echo_spacing, echo_spacing)
    
    # Generate random amplitudes, t2 times, and SNRs in the specified ranges
    amplitudes = np.array([
        np.random.uniform(low, high, size=num_samples)
        for low, high in amplitude_ranges
    ]).T  # Shape: (num_samples, pools)
    amplitudes /= amplitudes.sum(axis=1, keepdims=True)
    
    t2_times = np.array([
        np.random.uniform(low, high, size=num_samples)
        for low, high in t2_ranges
    ]).T  # Shape: (num_samples, pools)
    
    SNRs = np.random.uniform(snr_range[0], snr_range[1], num_samples)
    
    # Compute decay: sum over pools of amplitudes * exp(-echo_times / t2_times)
    decays = np.sum(
        amplitudes.reshape(num_samples, 1, pools) * np.exp(-echo_times.reshape(1, num_echoes, 1) / t2_times.reshape(num_samples, 1, pools)), 
        axis=2,
        )
    variance = 1 / (SNRs * (np.pi / 2) ** 0.5)
    noise_real = np.random.normal(0, variance[:, np.newaxis], size=(num_samples, num_echoes))
    noise_imag = np.random.normal(0, variance[:, np.newaxis], size=(num_samples, num_echoes))
    decays = np.sqrt((decays + noise_real) ** 2 + noise_imag ** 2)
    decays /= decays[:, 0:1]
    
    return decays, (amplitudes, t2_times)

    
    