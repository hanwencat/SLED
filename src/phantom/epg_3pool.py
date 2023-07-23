### Digital phantom produced via the 3-pool exponential decay model

# add src folder to sys.path to avoid error when importing modules in the src folder
import sys
import os
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_folder)

import numpy as np
import same_ecos.epg_cpmg as epg
import matplotlib.pyplot as plt


class phantom_epg_3pools:
    """
    digital phantom for the 3 pool model
    """

    def __init__(self, dims, mwf_range, fa_range, snr_range, fwf, t2s, t1s, nte, delta_te):
        """initialize the phantom
        Args:
            dims (list): size of the 3 dimensions
            mwf_range (list): myelin water fraction range
            fa_range (list): refocusing flip angle range (degrees)
            snr_range (list): SNR range
            fwf: free water fraction
            t2s (list): t2s of the 3 pools (seconds)
            t1s (list): t1s of the 3 pools (seconds)
            nte: number of echoes
            delta_te: echo spacing
        """
        # initialize parameters
        self.x_dim = dims[0]
        self.y_dim = dims[1]
        self.z_dim = dims[2]
        self.mwf_range = mwf_range
        self.fa_range = fa_range
        self.snr_range = snr_range
        self.fwf = fwf
        self.t2s = t2s
        self.t1s = t1s
        self.nte = nte
        self.delta_te = delta_te
        
        # initialize ground truth maps
        self.mwf, self.fa, self.snr = self.ground_truth_maps()
    
        # initialize multiple echo signals
        self.signal, self.noisy_signal = self.produce_signal()


    def ground_truth_maps(self):
        """
        produce ground truth maps
        """

        # create placeholders
        snr = np.zeros([self.x_dim, self.y_dim, self.z_dim])
        mwf = np.zeros([self.x_dim, self.y_dim, self.z_dim])
        fa = np.zeros([self.x_dim, self.y_dim, self.z_dim])

        # produce the ground truth snr map
        for y in range(self.y_dim):
            snr[:,y,:] = self.snr_range[0] + y*(self.snr_range[1]-self.snr_range[0])/self.y_dim

        # produce the ground truth mwf map
        for x in range(self.x_dim):
            mwf[x,:,:] = self.mwf_range[1] - x*(self.mwf_range[1]-self.mwf_range[0])/self.x_dim
            
        # produce the ground truth fa map
        for z in range(self.z_dim):
            fa[:,:,z] = self.fa_range[0] + z*(self.fa_range[1]-self.fa_range[0])/self.z_dim

        return mwf, fa, snr


    def produce_signal(self):
        """produce the decay signals at echo times"""
        
        t2_my = self.t2s[0]
        t2_ie = self.t2s[1]
        t2_fr = self.t2s[2]

        t1_my = self.t1s[0]
        t1_ie = self.t1s[1]
        t1_fr = self.t1s[2] 
        
        signal = np.zeros([self.x_dim, self.y_dim, self.z_dim, self.nte])
        noisy_signal = np.zeros([self.x_dim, self.y_dim, self.z_dim, self.nte])

        for x in range(self.x_dim):
            for y in range(self.y_dim):
                for z in range(self.z_dim):
                    # generate decay signal via epg algorithm (angle in radians)
                    signal_my = epg.cpmg_signal(self.nte, self.fa[x,y,z]/180*np.pi, self.delta_te, t2_my, t1_my)
                    signal_ie = epg.cpmg_signal(self.nte, self.fa[x,y,z]/180*np.pi, self.delta_te, t2_ie, t1_ie)
                    signal_fw = epg.cpmg_signal(self.nte, self.fa[x,y,z]/180*np.pi, self.delta_te, t2_fr, t1_fr)

                    signal[x,y,z,:] = (
                        self.mwf[x,y,z] * signal_my
                        + (1-self.mwf[x,y,z]-self.fwf) * signal_ie
                        + self.fwf * signal_fw
                        )
                    
                    # produce noise (zero-mean Gaussian noise on real and imaginary axes)
                    # noise variance is calculated according to snr
                    variance = 1/(self.snr[x,y,z]*((np.pi/2)**0.5))
                    noise_real = np.random.normal(0, variance, [self.nte,])
                    noise_img = np.random.normal(0, variance, [self.nte,])
                    # add noise to signal
                    noisy_signal[x,y,z,:] = (
                        (signal[x,y,z,:]+noise_real)**2 + noise_img**2
                        )**0.5

        return signal, noisy_signal

    def display_mwf(self, slice=50):
        """
        show ground truth mwf 
        """
        plt.figure(dpi=300)
        plt.xlabel('SNR')
        plt.ylabel('MWF')
        plt.title('Ground truth')
        plt.imshow(
            self.mwf[:,:,slice],
            vmin=self.mwf_range[0],
            vmax=self.mwf_range[1],
            extent=[
                self.snr_range[0], 
                self.snr_range[1], 
                self.mwf_range[0], 
                self.mwf_range[1]
                ],
            aspect=(self.snr_range[1]-self.snr_range[0])/(self.mwf_range[1]-self.mwf_range[0]),
            )
        plt.colorbar()
        plt.show()
    



if __name__ == "__main__":
    import time
    start_time = time.time()

    import yaml
    with open('sled_mese/configs/phantom/pt_epg_3pool.yml') as f:
        config = yaml.safe_load(f)

    # logging the experiment
    import logging
    from datetime import datetime
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Log the parameters in config to file
    logging.info(f'Experiment name: {config["name"]}')
    logging.info(f'Experiment begins at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
    logging.info('Config file contents:')
    for key, value in config.items():
        logging.info(f'{key}: {value}')

    # Make digital phantom
    logging.info('Digital phantom production in progress')
    phantom = phantom_epg_3pools(
        dims=config['dims'],
        mwf_range=config['mwf_range'],
        fa_range=config['fa_range'],
        snr_range=config['snr_range'],
        fwf=config['fwf'],
        t2s=config['t2s'],
        t1s=config['t1s'],
        nte=config['nte'],
        delta_te=config['delta_te'],
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time:0.3f}\n")
    logging.info(f'Digital phantom completed, elapsed time: {elapsed_time:.2f} seconds')

    import pickle
    with open(config['save_path'], 'wb') as f:
        pickle.dump(phantom, f)
    logging.info(f'Digital phantom data are saved in \'{config["save_path"]}\'\n')