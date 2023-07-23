import numpy as np
import math
import itertools
import epg_cpmg as epg
import os
import time
import logging

def max_components_finite_domain(SNR=100, T2_min=7, T2_max=1000):
    """
    Numerically determine the maximum number of T2 components that can be resolved for a given SNR at a certain T2 range.
    Args:
        SNR (float, optional): signal to noise ratio. Defaults to 100.
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 7.
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 1000.
    Returns:
        M: the maximum number of T2 components that can be resolved.
    """    
   
    M = 1
    f = -1
    while f < 0:
        M = M + 0.01
        f = M/np.log(T2_max/T2_min) * np.sinh(np.pi**2*M/np.log(T2_max/T2_min)) - (SNR/M)**2
    return M


def resolution_limit(T2_min=7, T2_max=2000, M=4):
    """
    Finite domain T2 resolution calculation.
    Args:
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 7.
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 2000.
        M (float, optional): maximum number of resolvable T2 components. Defaults to 4.
    Returns:
        resolution: the T2 resolution.
    """    
    
    resolution = (T2_max/T2_min)**(1/M)
    return resolution


def t2s_generator(T2_min=5, T2_max=2000, num_T2=3, resolution=3, scale='log'):
    """
    Randomly generate T2 locations.
    Generated T2 peaks should have uniform distribution (scale = linear or log) at the range of [T2_min, T2_max]
    Args:
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 5. 
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 2000.
        num_T2 (int, optional): number of T2 component. Defaults to 3.
        M_max (int, optional): global maximal number of resolvable T2 components. Defaults to 5.
        scale (str, optional): logarithmic or linear scale over the T2 range. Defaults to 'log'.
    Returns:
        T2_locations: An array of T2 locations (ms)
    """    
    
    T2_location = np.zeros(int(num_T2))
    T2_low = T2_min 
    for i in range(num_T2):
        # uniform distribution on linear scale
        if scale == 'linear': 
            T2_location[i] = np.random.uniform(T2_low, T2_max/(resolution**(num_T2-i-1))) 
        # uniform distribution on logarithmic scale 
        # #(https://stackoverflow.com/questions/43977717/how-do-i-generate-log-uniform-distribution-in-python/43977980)
        if scale == 'log':
            T2_location[i] = np.exp(np.random.uniform(np.log(T2_low),np.log(T2_max/(resolution**(num_T2-i-1)))))
        T2_low = T2_location[i]*resolution # set the new lower boundary        
    T2_location = np.around(T2_location, decimals=3)
    return T2_location
        
    
def T2_resolution_examiner(T2_location, resolution):
    """
    Examine whether two T2 peak locations obey the resolution limit.
    Args:
        T2_location (array): an array of T2 locations (ms)
        resolution (float): the resolution limit
    Returns:
        TF: True or False
    """    
    
    TF = True
    for x, y in itertools.combinations(T2_location,2):
        if x/y > 1/resolution and x/y < resolution:
            TF = False
            break        
    return TF


def required_frequency(T2_location):
    """
    Calculate the required frequency for calculating the amplitudes of t2 peaks.
    Args:
        T2_location (array): an array of T2 locations (ms)
    Returns:
        ratio_min, frequency: the minimal ratio between adjacent components, required frequency.
    """    
    
    T2_location_sort = np.sort(T2_location)
    ratio_min = (T2_location_sort[1:]/T2_location_sort[:-1]).min()
    frequency = np.pi/np.log(ratio_min)   
    return ratio_min, frequency


def minimum_amplitude_calculator(SNR, frequency):
    """
    Calculate the minimum allowable amplitude for all T2 components at a given SNR.
    Args:
        SNR (float): signal to noise ratio
        frequency (float): required frequency of the T2 components
    Returns:
        minimum_amp_T2_peak: the minimal amplitude of T2 components
    """    
    
    amp_noise = 1/SNR
    minimum_amp_T2_peak = amp_noise * np.sqrt(frequency/np.pi * np.sinh(np.pi*frequency))    
    return minimum_amp_T2_peak


def amps_generator(num_T2, minimum_amplitude):
    """
    Randomly generate normalized T2 peak amplitude.
    Generated amplitude should have uniform distribution at range [minimum_amplitude, 1-(num_T2-1)*minimum_amplitude]
    Args:
        num_T2 (int): number of T2 components
        minimum_amplitude (float): the minimal amplitude
    Returns:
        T2_amplitude: an array of T2 amplitudes
    """    
    
    T2_amplitude = np.zeros(num_T2)
    remainder = 1
    for i in range(num_T2-1):
        T2_amplitude[i] = np.random.uniform(minimum_amplitude, remainder-(num_T2-i-1)*minimum_amplitude, 1)
        remainder = 1 - T2_amplitude.sum()
    T2_amplitude[-1] = remainder
    T2_amplitude = T2_amplitude/T2_amplitude.sum()    
    T2_amplitude = np.squeeze(T2_amplitude) 
    if T2_amplitude.shape[0] == 1:
        T2_amplitude = T2_amplitude.reshape(1,) 
    ### amplitude array has a descending trend so it needs to be shuffled 
    np.random.shuffle(T2_amplitude) 
    return T2_amplitude  


def signal_generator(t2s, amps, etl, alpha, delta_te, t1):
    """use EPG algorithm to generate cpmg multi-echo signals"""

    signals = np.zeros(etl)
    for t2, amp in zip(t2s, amps):
        signal = epg.cpmg_signal(
            etl=etl,
            alpha=alpha,
            delta_te=delta_te,
            t2=t2, # unit in seconds
            t1=t1,
            )
        signals += signal*amp # weighted sum of all components
    
    # use list comprehension has similar performance compared with for loops (tested)
    # signals = [epg.cpmg_signal(
    #     etl=etl,
    #     alpha=alpha,
    #     delta_te=delta_te,
    #     t2=t2/1000,
    #     t1=t1,
    # )*amp for t2, amp in zip(t2s.tolist(), amps.tolist())]
    # signals = np.array(signals)
    
    return signals


def add_noise(signal, SNR):
    """
    1. Project pure signal to real and imaginary axis according to a randomly generated phase factor.
    2. Generate noise (normal distribution on real and imaginary axis)
    3. Noise variance is determined by SNR (Rayleigh noise floor).
    Args:
        signal (array): the decay signal
        SNR (float): signal to noise ratio
    Returns:
        signal_with_noise: the signal with added noise on both real and imaginary axis
    """    
    
    Rayleigh_noise_variance = 1/(SNR * math.sqrt(math.pi/2))   
    noise_real = np.random.normal(0, Rayleigh_noise_variance, signal.shape[0])
    noise_imaginary = np.random.normal(0, Rayleigh_noise_variance, signal.shape[0])    
    noisy_signal = ((signal + noise_real)**2 + (noise_imaginary)**2) ** 0.5   
    
    return noisy_signal


def t2_basis_generator(T2_min=7, T2_max=1000, num_basis_T2=40):
    """
    Generate T2 basis set.
    Args:
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 7.
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 1000.
        num_basis_T2 (int, optional): number of basis t2s. Defaults to 40.
    Returns:
        t2_basis: generated basis t2s (ms).
    """    
    
    t2_basis = np.geomspace(T2_min, T2_max, num_basis_T2)    
    return t2_basis


def spectrum_nearest_embedding(T2_location, T2_amplitude, t2_basis):
    """
    This function takes randomly generated t2 peak locations and amplitudes as inputs, and uses basis t2s to represent these T2 peaks.
    Each peak is embedded by two nearest basis T2s.
    Args:
        T2_location (array): an array of T2 locations (ms)
        T2_amplitude (array): an array of T2 amplitudes
        t2_basis (array): the basis t2s (ms)
    Returns:
        spectrum: the T2 spectrum depicted by the basis t2s
    """   
    
    ### create multi-dimensional placeholder (each dimension for each peak)
    spectrum=np.zeros([T2_location.shape[0],t2_basis.shape[0]])  
    ### iterate through each peak and find the nearest couple of t2 basis and assign weighting factors
    for i in range(T2_location.shape[0]):
        for j in range(t2_basis.shape[0]):         
            if abs(t2_basis[j]-T2_location[i])<0.000000001:
                spectrum[i,j] = T2_amplitude[i]            
            elif t2_basis[j-1]<T2_location[i] and t2_basis[j]>T2_location[i]:
                spectrum[i,j-1] = (t2_basis[j]-T2_location[i])/(t2_basis[j]-t2_basis[j-1])*T2_amplitude[i]
                spectrum[i,j] = (T2_location[i]-t2_basis[j-1])/(t2_basis[j]-t2_basis[j-1])*T2_amplitude[i]        
    ### return one dimensional train label
    return spectrum.sum(axis=0)


def spectrum_gaussian_embedding(T2_location, T2_amplitude, t2_basis, sigma=1):
    """
    This function takes randomly generated t2 peak locations and amplitudes as inputs, and uses basis t2s to represent these T2 peaks.
    Each peak generates a gaussian function centered at its peak location (in log space), and then embedded by all basis T2s
    Args:
        T2_location (array): an array of T2 locations (ms)
        T2_amplitude (array): an array of T2 amplitude
        T2_min (float): the T2 lower boundary (ms)
        T2_max (float): the T2 upper boundary (ms)
        t2_basis (array): the basis t2s (ms)
        sigma (float, optional): variance of the Gaussian peaks. Defaults to 1.
    Returns:
        spectrum: the T2 spectrum depicted by basis t2s
    """    
    
    ### create multi-dimensional placeholder (each dimension for each peak)
    spectrum=np.zeros([T2_location.shape[0],t2_basis.shape[0]])
    ### iterate through each peak and assign weighting factors to t2_basis according to normal distribution
    for i in range(T2_location.shape[0]):
        spectrum[i,:] = gaussian_embedding_log_scale(T2_location[i], t2_basis=t2_basis, sigma=sigma) * T2_amplitude[i]  
    ### return one dimensional train label
    return spectrum.sum(axis=0)


def gaussian_embedding_log_scale(peak, t2_basis, sigma):
    """
    This function takes one t2 delta peak as inputs, and returns a normalized gaussian weighted t2_basis labels on log scale.
    Args:
        peak (float): one T2 location
        T2_min (float): T2 lower boundary (ms)
        T2_max (float): T2 upper boundary (ms)
        t2_basis (array): basis t2s (ms)
        sigma (float): variance of the Gaussian peak.
    Returns:
        t2_basis_weights_scaled: the normalized Gaussian peaks
    """    
    
    T2_min = t2_basis.min()
    T2_max = t2_basis.max()
    
    base = (T2_max/T2_min)**(1/t2_basis.shape[0])
    t2_basis_index = np.arange(t2_basis.shape[0])
    peak_index = np.log(peak/T2_min)/np.log(base)
    t2_basis_weights = 1/(sigma*np.sqrt(2*math.pi)) * np.exp(-(t2_basis_index - peak_index)**2/(2*sigma**2))
    t2_basis_weights[t2_basis_weights<1e-4] = 0 # threshold the minimum weight
    t2_basis_weights_scaled = t2_basis_weights/t2_basis_weights.sum()  
    return t2_basis_weights_scaled


def produce_training_data(realizations = 10000,
                          snr_min = 50,
                          snr_max = 500,
                          etl = 32,
                          delta_te = 10,
                          T1 = 1000,
                          num_t2_basis = 40,
                          FA_min = 50,
                          T2_min = 5,
                          T2_max = 2000,
                          gaussian_width = 1,
                          exclude_M_max = False):
    """
    Produce training data via SAME-ECOS simulation pipeline (use a single cpu core).
    Args:
        realizations (int, optional): the number of simulation realizations. Defaults to 10000.
        snr_min (float, optional): lower boundary of SNR. Defaults to 50.
        snr_max (float, optional): upper boundary of SNR. Defaults to 500.
        etl (int, optional): the number of echoes in the echo train. Defaults to 32.
        num_t2_basis (int, optional): the number of basis t2s. Defaults to 40.
        FA_min (float, optional): the minimal refocusing flip angle (degree) for simulation. Defaults to 50.
        gaussian_width (float, optional): the variance of the gaussian peak. Defaults to 1.
        T2_min (float, optional): the overall minimal T2 (ms) of the analysis. Defaults to calculate on the fly if None is given.
        T2_max (float, optional): the overall maximal T2 (ms) of the analysis. Defaults to to 2000ms if None is given.
        exclude_M_max (bool, optional): exclude the M_max if True. Defaults to True.
    Returns:
        data: dictionary collection of the produced training data
    """    
    
    ### Define T2 range, maximum number (M_max) of T2 peaks at the highest SNR, allowable number (N) of T2 peaks for simulation 
    t2_basis = t2_basis_generator(T2_min, T2_max, num_t2_basis)
    M_max = int(np.floor(max_components_finite_domain(snr_max, T2_min, T2_max))) ## M at highest SNR
    #resolution_max = resolution_limit(T2_min_universal, T2_max_universal, M_max) ## resolution at highest SNR
    
    if exclude_M_max == True:
        N = M_max - 1 # M_max is excluded for simulation
    else:
        N = M_max
    
    ### Create placeholders for memory efficiency
    T2_location_all = np.zeros([realizations, N])
    T2_amplitude_all = np.zeros([realizations,N])
    decay_curve_all = np.zeros([realizations, etl])
    decay_curve_with_noise_all = np.zeros([realizations, etl])
    spectrum_nearest_all = np.zeros([realizations,num_t2_basis])
    spectrum_gaussian_all = np.zeros([realizations,num_t2_basis])
    num_T2_SNR_FA_all = np.zeros([realizations,3])
    
    ### For each realization
    for i in range(realizations):        
        
        ### Randomly determine the SNR, the number of T2s, and the flip angle FA.
        SNR = np.random.randint(snr_min, snr_max)
        # SNR = 100 ## for fixed SNR
        M = max_components_finite_domain(SNR, T2_min, T2_max)
        M_floor = int(np.floor(M))
        if exclude_M_max == True:
            M_choices = np.arange(1, M_floor)
        else:
            M_choices = np.arange(1, M_floor+1)
        
        weight = M_choices**0.2 ## weighting factor for each choice
        num_T2 = int(np.random.choice(M_choices, p=weight/weight.sum()))
        FA = np.around(np.random.uniform(FA_min, 180), decimals=2)
        
        ### Calculate the resolution limit
        rl = resolution_limit(T2_min, T2_max, M)
        
        ### Randomly generate T2 peak location with respect to resolution limit.
        T2_location = t2s_generator(T2_min, T2_max, num_T2, rl, scale='log')
        while T2_resolution_examiner(T2_location, rl) == False:
            T2_location = t2s_generator(T2_min, T2_max, num_T2, rl, scale='log')
            
        ### Randomly generate T2 peak amplitude. When two or more peaks, minimal detectable amplitude is calculated
        if num_T2==1:
            T2_amplitude = np.array([1.0])
        else: 
            _ , frequency = required_frequency(T2_location)
            minimum_amplitude = minimum_amplitude_calculator(SNR, frequency)
            T2_amplitude = amps_generator(num_T2, minimum_amplitude)
        
        ### Decay curve generation (weighted sum of each T2 component) (convert units: ms to s, degree to rad)
        decay_curve = signal_generator(T2_location/1000, T2_amplitude, etl, FA/180*np.pi, delta_te/1000, T1/1000)
        
        ### Add noise to decay curve
        decay_curve_with_noise = add_noise(signal=decay_curve, SNR=SNR)
        
        ### T2 basis set embedding (nearest t2_basis neighbors)
        spectrum_nearest = spectrum_nearest_embedding(T2_location, T2_amplitude, t2_basis)
        
        ### T2 basis set embedding (gaussian peaks)
        spectrum_gaussian = spectrum_gaussian_embedding(T2_location, T2_amplitude, t2_basis, gaussian_width)
        
        ### Pad T2_location and T2_amplitude to have uniform size
        T2_location = np.pad(T2_location,[(0, N-int(num_T2))], mode='constant', constant_values=0)
        T2_amplitude = np.pad(T2_amplitude,[(0, N-int(num_T2))], mode='constant', constant_values=0)   
        
        ### Store generated parameters in placeholders       
        T2_location_all[i,:] = T2_location
        T2_amplitude_all[i,:] = T2_amplitude 
        decay_curve_all[i,:] = decay_curve
        decay_curve_with_noise_all[i,:] = decay_curve_with_noise
        spectrum_nearest_all[i,:] = spectrum_nearest
        spectrum_gaussian_all[i,:] = spectrum_gaussian
        num_T2_SNR_FA_all[i,:] = num_T2, SNR, FA

        if i % (realizations/10) == 0:
            print(f"PID {os.getpid()} Progress: {i}/{realizations}")
            logging.debug(f"PID {os.getpid()} Progress: {i}/{realizations}")
    
    ### return a data dict
    data = {'T2_location': T2_location_all, 
            'T2_amplitude': T2_amplitude_all,
            'decay_curve': decay_curve_all,
            'decay_curve_with_noise': decay_curve_with_noise_all,
            'spectrum_nearest': spectrum_nearest_all,
            'spectrum_gaussian': spectrum_gaussian_all,
            'num_T2_SNR_FA': num_T2_SNR_FA_all,
            }      
    
    return data


def mp_func(func, realizations, other_args, ncores):
    """
    use multiprocessing to speed up a function that store its results in a dict
    func must have realizations as its first argument
    """

    ### distribute job to each cpu core
    realizations_pool_list = [realizations//ncores]*ncores 
    if realizations%ncores != 0:
        realizations_pool_list.append(realizations%ncores)
    
    ### start multiprocessing
    import multiprocessing as mp
    pool = mp.Pool(processes=ncores)
    data = pool.starmap(
        func, 
        [(realizations, ) + other_args for realizations in realizations_pool_list],
        )
    pool.close()
    pool.join()
    
    ### concatenate data calculated from each cpu core
    keys = data[0].keys()
    data_all =  {key: None for key in keys}
    for key in keys:
        data_all[key] = np.concatenate([data[x][key] for x in range(len(data))])    
    return data_all


class sim_param:
    def __init__(
            self, realizations, snr_min, snr_max, 
            etl, delta_te, T1_fix, num_t2_basis, FA_min, 
            T2_min, T2_max, gaussian_width, exclude_M_max,
            ):
        self.realizations = realizations
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.etl = etl
        self.delta_te = delta_te
        self.T1_fix = T1_fix
        self.num_t2_basis = num_t2_basis
        self.FA_min = FA_min
        self.T2_min = T2_min
        self.T2_max = T2_max
        self.gaussian_width = gaussian_width
        self.exclude_M_max = exclude_M_max


class dict2obj:
    def __init__(self, dic):
        self.__dict__.update(dic)


class simulation:
    def __init__(self, param, data):
        self.param = param
        self.data = data



# main program

if __name__ == "__main__":
    # load the config file
    import yaml
    with open('configs/same_ecos/se_hyperfine_120echo.yml') as f:
        config = yaml.safe_load(f)

    # np.set_printoptions(precision=3)
    # snr = 200
    # t2_min = 5
    # t2_max = 1000
    
    # print('##############################')
    # print(f'experimental condition: SNR = {snr}, T2_min = {t2_min} ms, T2_max = {t2_max} ms')
    # print('##############################\n')
    
    # m = max_components_finite_domain(
    #     SNR=snr, 
    #     T2_min=t2_min, 
    #     T2_max=t2_max,
    #     )
    # m_floor = int(np.floor(m))
    
    # rl = resolution_limit(
    #     T2_min=t2_min, 
    #     T2_max=t2_max, 
    #     M=m_floor,
    #     )
    
    # t2s = t2s_generator(
    #     T2_min=t2_min, 
    #     T2_max=t2_max, 
    #     num_T2=m_floor, 
    #     resolution=rl,
    #     scale='log',
    #     )
    
    
    # print(f'maximum number of resolvable components: {m:.2f}')
    # print(f'maximum number of resolvable components (round down): {m_floor}')
    # print(f'resolution limit: {rl:.2f}')
    # print(f't2s: {t2s} ms')
    # print(f'resolution limit obeyed: {T2_resolution_examiner(t2s, rl)}\n')

    # ratio_min, f = required_frequency(t2s)
    # amps_min = minimum_amplitude_calculator(SNR=300, frequency=f)
    # amps = amps_generator(num_T2=m_floor, minimum_amplitude=amps_min)
    # print(f'ratio_min:{ratio_min:.2f}, required frequency: {f:.2f}, minimum amplitude: {amps_min:.2f}')
    # print(f'amps: {amps}\n')


    # # produce cpmg sequence signals using epg algorithm
    # np.set_printoptions(precision=3)
    # etl = 40
    # delta_te = 0.00675 # change unit to second
    # t1 = 1 
    # angle = 120 # in degrees
    # alpha = angle/180*np.pi # in radians
    # num_t2_basis = 40
    # print('##############################')
    # print(f'EPG simulation parameters: ETL = {etl}, delta_te = {delta_te*1000} ms, T1 = {t1*1000} ms, alpha = {angle} degrees')
    # print('##############################\n')
    
    # signals = signal_generator(t2s/1000, amps, etl, alpha, delta_te, t1) # unit in seconds
    # noisy_signals = add_noise(signals, snr)
    # print(f'averaged signals: \n{signals}\n')
    # print(f'noisy signals: \n{noisy_signals}\n')


    # t2_basis = t2_basis_generator(t2_min, t2_max, num_t2_basis)
    # spectrum = spectrum_nearest_embedding(t2s, amps, t2_basis)
    # # spectrum = spectrum_gaussian_embedding(t2s, amps, t2_basis, sigma=0.5)
    # print(f'spectrum: \n{spectrum}\n')
 
    
    ### produce many training examples
    ## initialization parameters (relaxation times in ms)
    realizations = config['realizations']
    snr_min = config['snr_min']
    snr_max = config['snr_max']
    etl = config['etl'] 
    delta_te = config['delta_te']
    T1_fix = config['T1_fix']
    num_t2_basis = config['num_t2_basis']
    FA_min = config['FA_min'] 
    T2_min = config['T2_min']
    T2_max = config['T2_max']
    gaussian_width = config['gaussian_width']
    exclude_M_max = config['exclude_M_max']


    # logging the experiment
    import logging
    from datetime import datetime
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Log the config to file
    logging.info(f'Experiment name: {config["name"]}')
    logging.info(f'Experiment begins at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
    logging.info('Config file contents:')
    for key, value in config.items():
        logging.info(f'{key}: {value}')


    print('##############################')
    print(f'SAME-ECOS training data production in progress')
    print('##############################\n')
    logging.info('SAME-ECOS training data production in progress')
    # ## use single processor to produce training data
    # start_time = time.time()
    # data = produce_training_data(
    #     realizations,
    #     snr_min,
    #     snr_max,
    #     etl,
    #     delta_te,
    #     T1_fix,
    #     num_t2_basis,
    #     FA_min,
    #     T2_min,
    #     T2_max,
    #     gaussian_width,
    #     exclude_M_max,
    #     )

    # print(f'data: \n{data.keys()}\n')
    # print(f'Elapsed time: {time.time() - start_time:.2f} seconds\n')
    
    ## use multiprocessing to produce training data
    start_time = time.time()
    ncores = config['ncores']
    other_args = (
        snr_min, 
        snr_max,
        etl,
        delta_te,
        T1_fix,
        num_t2_basis,
        FA_min,
        T2_min,
        T2_max,
        gaussian_width,
        exclude_M_max,
        )
    
    data = mp_func(
        produce_training_data, 
        realizations, 
        other_args, 
        ncores,
        )
    
    logging.info('SAME-ECOS training data production finished')
    print(f'data keys: \n{data.keys()}\n')
    print(f'Elapsed time: {time.time() - start_time:.2f} seconds\n')
    logging.info(f'data keys: {data.keys()}')
    logging.info(f'Elapsed time: {time.time() - start_time:.2f} seconds')


    ## save the training data and parameters into simulation object
    data_obj = dict2obj(data)
    param_obj = sim_param(
        realizations, snr_min, snr_max, 
        etl, delta_te, T1_fix, num_t2_basis, FA_min, 
        T2_min, T2_max, gaussian_width, exclude_M_max,
        )
    simulation_obj = simulation(param_obj, data_obj)

    import pickle
    with open(config['save_path'], 'wb') as f:
        pickle.dump(simulation_obj, f)
    
    logging.info(f'SAME-ECOS training data are saved in \'{config["save_path"]}\'\n')
    # save_folder_path = '/export01/data/Hanwen/sled_mese/same-ecos/'
    # os.chdir(save_folder_path)
    # np.savez('pretrain_data_32echo.npz', **data)

    # np.save('/export01/data/Hanwen/sled_mese/same_ecos/simulation_32echo.npy', simulation_obj)