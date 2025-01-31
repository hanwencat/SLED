# import numpy as np

# def generate_pretrain_data(config):
#     """generate synthetic data for pretraining the model"""
    
#     # load the parameters from the config file
#     num_samples = config['number_of_samples']
#     pools = config['number_of_pools']
#     t2_ranges = config['t2_ranges']
#     amplitude_ranges = config['amplitude_ranges']
#     snr_range = config['snr_range']
    
#     # get echo times
#     first_echo_time = config['first_echo_time']
#     echo_spacing = config['echo_spacing']
#     num_echoes = config['number_of_echoes']
#     echo_times = np.arange(first_echo_time, first_echo_time + num_echoes * echo_spacing, echo_spacing)
    
#     # Generate random amplitudes, t2 times, and SNRs in the specified ranges
#     amplitudes = np.array([
#         np.random.uniform(low, high, size=num_samples)
#         for low, high in amplitude_ranges
#     ]).T  # Shape: (num_samples, pools)
#     amplitudes /= amplitudes.sum(axis=1, keepdims=True)
    
#     t2_times = np.array([
#         np.random.uniform(low, high, size=num_samples)
#         for low, high in t2_ranges
#     ]).T  # Shape: (num_samples, pools)
    
#     SNRs = np.random.uniform(snr_range[0], snr_range[1], num_samples)
    
#     # Compute decay: sum over pools of amplitudes * exp(-echo_times / t2_times)
#     decays = np.sum(
#         amplitudes.reshape(num_samples, 1, pools) * np.exp(-echo_times.reshape(1, num_echoes, 1) / t2_times.reshape(num_samples, 1, pools)), 
#         axis=2,
#         )
#     variance = 1 / (SNRs * (np.pi / 2) ** 0.5)
#     noise_real = np.random.normal(0, variance[:, np.newaxis], size=(num_samples, num_echoes))
#     noise_imag = np.random.normal(0, variance[:, np.newaxis], size=(num_samples, num_echoes))
#     decays = np.sqrt((decays + noise_real) ** 2 + noise_imag ** 2)
#     decays /= decays[:, 0:1]
    
#     # If CSF simulations are requested
#     if config['csf_sim']:
#         csf_n_samples = config['csf_n_samples']
        
#         # Fixed amplitudes for CSF: [0, 0, 1]
#         csf_amplitudes = np.tile([0, 0, 1], (csf_n_samples, 1))
        
#         # Generate T2 times and SNR for CSF samples
#         csf_t2_times = np.array([
#             np.random.uniform(low, high, size=csf_n_samples)
#             for low, high in t2_ranges
#         ]).T
        
#         csf_SNRs = np.random.uniform(snr_range[0], snr_range[1], csf_n_samples)
        
#         # Compute CSF decays
#         csf_decays = np.sum(
#             csf_amplitudes.reshape(csf_n_samples, 1, pools) *
#             np.exp(-echo_times.reshape(1, num_echoes, 1) / csf_t2_times.reshape(csf_n_samples, 1, pools)),
#             axis=2
#         )
        
#         # Add noise to CSF decays
#         csf_variance = 1 / (csf_SNRs * (np.pi / 2)**0.5)
#         csf_noise_real = np.random.normal(0, csf_variance[:, np.newaxis], size=(csf_n_samples, num_echoes))
#         csf_noise_imag = np.random.normal(0, csf_variance[:, np.newaxis], size=(csf_n_samples, num_echoes))
#         csf_decays = np.sqrt((csf_decays + csf_noise_real)**2 + csf_noise_imag**2)
#         csf_decays /= csf_decays[:, 0:1]
        
#         # Concatenate CSF data with the main dataset
#         decays = np.concatenate([decays, csf_decays], axis=0)
#         amplitudes = np.concatenate([amplitudes, csf_amplitudes], axis=0)
#         t2_times = np.concatenate([t2_times, csf_t2_times], axis=0)
#         variance = np.concatenate([variance, csf_variance], axis=0)

#     return decays, (amplitudes, t2_times, variance)

    
    
import numpy as np
import math

def generate_pretrain_data(config):
    """
    Generate synthetic T2 decay data using a T2 basis for Gaussian embedding,
    returning:
      decays_noisy,
      (amplitudes, t2_times, variance, amps_spectrum)

    - decays_noisy: (num_samples, num_echoes)
    - amplitudes:   (num_samples, pools)
    - t2_times:     (num_samples, pools)
    - variance:     (num_samples,)
    - amps_spectrum:(num_samples, number_of_t2_basis)
    """

    # 1) Load parameters
    num_samples       = config['number_of_samples']
    t2_ranges         = config['t2_ranges']                 # e.g. [(0.01,0.02),(0.04,0.06),(0.3,0.5)] (3-pool t2s in second)
    amplitude_ranges  = config['amplitude_ranges']          # e.g. [(0.1,0.5),(0.1,0.5),(0.1,0.4)] (3-pool amplitudes)
    snr_range         = config['snr_range']                 # e.g. [50,150]

    first_echo_time   = config['first_echo_time']           # e.g. 0.01 (in second)
    echo_spacing      = config['echo_spacing']              # e.g. 0.01 (in second)
    num_echoes        = config['number_of_echoes']          # e.g. 32

    # T2 basis range (can be larger than pool-specific T2 ranges)
    t2_basis_min, t2_basis_max = config['t2_basis_range']                   # e.g. (0.005, 2.0)
    number_of_t2_basis         = config['number_of_pools']               # e.g. 100
    gauss_peak_sigma           = config.get('gauss_peak_sigma', 1.0)        # log-scale Gaussian width

    # Echo times array
    echo_times = np.arange(
        first_echo_time,
        first_echo_time + num_echoes * echo_spacing,
        echo_spacing
    )

    # 2) Draw random T2 times & amplitudes
    # -------------------------------------
    # T2 times: shape => (num_samples, pools)
    t2_times = np.array([
        np.random.uniform(low, high, size=num_samples)
        for (low, high) in t2_ranges
    ]).T  # shape: (pools, num_samples) -> transpose -> (num_samples, pools)

    # Amplitudes: shape => (num_samples, pools)
    amplitudes = np.array([
        np.random.uniform(low, high, size=num_samples)
        for (low, high) in amplitude_ranges
    ]).T
    # Normalize each sample's amplitude row to sum=1
    amplitudes /= amplitudes.sum(axis=1, keepdims=True)

    # Random SNR: shape => (num_samples,)
    SNRs = np.random.uniform(snr_range[0], snr_range[1], num_samples)


    # If CSF simulations are requested
    if config['csf_sim']:
        csf_n_samples = config['csf_n_samples']
        
        # Fixed amplitudes for CSF: [0, 0, 1] (only the free water pool)
        csf_amplitudes = np.tile([0, 0, 1], (csf_n_samples, 1))
        
        # Generate T2 times and SNR for CSF samples
        csf_t2_times = np.array([
            np.random.uniform(low, high, size=csf_n_samples)
            for low, high in t2_ranges
        ]).T
        
        csf_SNRs = np.random.uniform(snr_range[0], snr_range[1], csf_n_samples)/10
        
        # Concatenate CSF data with the main dataset
        amplitudes = np.concatenate([amplitudes, csf_amplitudes], axis=0)
        t2_times = np.concatenate([t2_times, csf_t2_times], axis=0)
        SNRs = np.concatenate([SNRs, csf_SNRs], axis=0)
        
    # 3) Build log-spaced T2 basis
    # -----------------------------
    t2_basis = np.logspace(
        np.log10(t2_basis_min),
        np.log10(t2_basis_max),
        number_of_t2_basis
    )

    # 4) Vectorized Gaussian embedding
    # ---------------------------------
    # We interpret t2_basis[i] = t2_basis_min * base^i for i in [0..number_of_t2_basis-1],
    # where base = (t2_basis_max / t2_basis_min)^(1 / (number_of_t2_basis-1)).

    T2_min_basis = t2_basis[0]
    T2_max_basis = t2_basis[-1]
    n_points = number_of_t2_basis

    base = (T2_max_basis / T2_min_basis) ** (1 / (n_points - 1))

    # For each sample & pool, convert T2 => log index
    # shape => (num_samples, pools)
    peak_indices = np.log(t2_times / T2_min_basis) / np.log(base)

    # Create array of T2 basis indices => [0,1,2,...,n_points-1]
    t2_basis_index = np.arange(n_points)

    # Expand dims for broadcasting:
    #   peak_indices => (num_samples, pools)     -> (num_samples, pools, 1)
    #   t2_basis_index => (n_points,)           -> (1, 1, n_points)
    peak_indices_3d   = peak_indices[:, :, np.newaxis]       
    t2_basis_index_3d = t2_basis_index[np.newaxis, np.newaxis, :]

    # Unnormalized Gaussians => shape (num_samples, pools, n_points)
    factor = 1.0 / (gauss_peak_sigma * np.sqrt(2 * math.pi))
    dist_unnorm = factor * np.exp(
        -((t2_basis_index_3d - peak_indices_3d)**2) / (2 * gauss_peak_sigma**2)
    )

    # Threshold small values
    dist_unnorm[dist_unnorm < 1e-4] = 0.0

    # Normalize each (sample, pool) distribution across T2 dimension
    # sum over axis=2 => shape (num_samples, pools, 1)
    sum_unnorm = dist_unnorm.sum(axis=2, keepdims=True)
    sum_unnorm[sum_unnorm == 0] = 1e-12  # avoid divide-by-zero
    dist_norm = dist_unnorm / sum_unnorm  # shape => (num_samples, pools, n_points)

    # Multiply by each pool's amplitude
    # amplitudes => (num_samples, pools) => expand => (num_samples, pools, 1)
    amplitudes_3d = amplitudes[:, :, np.newaxis]
    dist_weighted = dist_norm * amplitudes_3d
      # shape => (num_samples, pools, n_points)

    # Sum across pools => final T2 spectrum => (num_samples, n_points)
    amps_spectrum = dist_weighted.sum(axis=1)

    # 5) Compute decays for each sample at each echo
    # ---------------------------------------------
    # decays[i, j] = sum_k( amps_spectrum[i, k] * exp(- echo_times[j]/t2_basis[k]) )
    # We'll do a broadcast multiply and sum over k.

    echo_times_exp = echo_times[np.newaxis, :, np.newaxis]  # (1, num_echoes, 1)
    t2_basis_exp   = t2_basis[np.newaxis, np.newaxis, :]    # (1, 1, n_points)
    spectra_exp    = amps_spectrum[:, np.newaxis, :]        # (num_samples, 1, n_points)

    # shape => (1, num_echoes, n_points)
    decay_factors = np.exp(-echo_times_exp / t2_basis_exp)

    # shape => (num_samples, num_echoes)
    decays = np.sum(spectra_exp * decay_factors, axis=2)

    # 6) Add Rician noise, normalize by first echo
    # ---------------------------------------------
    # Rician variance => (num_samples,)
    variance = 1 / (SNRs * (np.pi / 2)**0.5)

    # Expand => shape (num_samples, 1)
    variance_2d = variance[:, np.newaxis]

    noise_real = np.random.normal(0, variance_2d, size=decays.shape)
    noise_imag = np.random.normal(0, variance_2d, size=decays.shape)

    decays_noisy = np.sqrt((decays + noise_real)**2 + noise_imag**2)
    decays_noisy /= decays_noisy[:, [0]]  # normalize by first echo

    # 7) Return a dictionary containing all outputs
    return {
        'decays': decays_noisy,
        'amplitudes': amplitudes,
        't2_times': t2_times,
        'variance': variance,
        'amps_spectrum': amps_spectrum
    }

    