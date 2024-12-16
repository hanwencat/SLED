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
    
    # If CSF simulations are requested
    if config['csf_sim']:
        csf_n_samples = config['csf_n_samples']
        
        # Fixed amplitudes for CSF: [0, 0, 1]
        csf_amplitudes = np.tile([0, 0, 1], (csf_n_samples, 1))
        
        # Generate T2 times and SNR for CSF samples
        csf_t2_times = np.array([
            np.random.uniform(low, high, size=csf_n_samples)
            for low, high in t2_ranges
        ]).T
        
        csf_SNRs = np.random.uniform(snr_range[0], snr_range[1], csf_n_samples)
        
        # Compute CSF decays
        csf_decays = np.sum(
            csf_amplitudes.reshape(csf_n_samples, 1, pools) *
            np.exp(-echo_times.reshape(1, num_echoes, 1) / csf_t2_times.reshape(csf_n_samples, 1, pools)),
            axis=2
        )
        
        # Add noise to CSF decays
        csf_variance = 1 / (csf_SNRs * (np.pi / 2)**0.5)
        csf_noise_real = np.random.normal(0, csf_variance[:, np.newaxis], size=(csf_n_samples, num_echoes))
        csf_noise_imag = np.random.normal(0, csf_variance[:, np.newaxis], size=(csf_n_samples, num_echoes))
        csf_decays = np.sqrt((csf_decays + csf_noise_real)**2 + csf_noise_imag**2)
        csf_decays /= csf_decays[:, 0:1]
        
        # Concatenate CSF data with the main dataset
        decays = np.concatenate([decays, csf_decays], axis=0)
        amplitudes = np.concatenate([amplitudes, csf_amplitudes], axis=0)
        t2_times = np.concatenate([t2_times, csf_t2_times], axis=0)

    return decays, (amplitudes, t2_times)

    
    