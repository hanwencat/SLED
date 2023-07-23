# Digital phantom produced via the 3-pool exponential decay model
import numpy as np
import matplotlib.pyplot as plt

class phantom_exp_3pools:
    """
    digital phantom for the 3 pool model
    """

    def __init__(self, dims, mwf_range, fwf_range, snr_range, t2s, te):
        """initialize the phantom
        Args:
            dims (list): size of the 3 dimensions
            mwf_range (list): myelin water fraction range
            fwf_range (list): free water fraction range
            snr_range (list): SNR range
            t2s (list): t2s of the 3 pools
            te(list): echo times for producing signals
        """
        # initialize parameters
        self.x_dim = dims[0]
        self.y_dim = dims[1]
        self.z_dim = dims[2]
        self.mwf_range = mwf_range
        self.fwf_range = fwf_range
        self.snr_range = snr_range
        self.t2s = t2s
        
        # initialize ground truth maps
        self.mwf, self.iewf, self.fwf, self.snr = self.ground_truth_maps()
    
        # initialize multiple echo signals
        self.signal, self.noise_real, self.noise_img = self.produce_signal(te)


    def ground_truth_maps(self):
        """
        produce ground truth maps
        """

        # create placeholders
        snr = np.zeros([self.x_dim, self.y_dim, self.z_dim])
        mwf = np.zeros([self.x_dim, self.y_dim, self.z_dim])
        fwf = np.zeros([self.x_dim, self.y_dim, self.z_dim])

        # produce the ground truth snr map
        for y in range(self.y_dim):
            snr[:,y,:] = self.snr_range[0] + y*(self.snr_range[1]-self.snr_range[0])/self.y_dim

        # produce the ground truth mwf map
        for x in range(self.x_dim):
            mwf[x,:,:] = self.mwf_range[1] - x*(self.mwf_range[1]-self.mwf_range[0])/self.x_dim
            
        # produce the ground truth fwf map
        for z in range(self.z_dim):
            fwf[:,:,z] = self.fwf_range[0] + z*(self.fwf_range[1]-self.fwf_range[0])/self.z_dim
        
        # produce the ground truth iewf map
        iewf = 1 - mwf - fwf

        return mwf, iewf, fwf, snr


    def produce_signal(self, te):
        """produce the decay signals at echo times"""
        
        t2_my = self.t2s[0]
        t2_ie = self.t2s[1]
        t2_fr = self.t2s[2]
        
        nte = len(te)
        signal = np.zeros([self.x_dim, self.y_dim, self.z_dim, nte])
        noise_real = np.zeros([self.x_dim, self.y_dim, self.z_dim, nte])
        noise_img = np.zeros([self.x_dim, self.y_dim, self.z_dim, nte])

        for x in range(self.x_dim):
            for y in range(self.y_dim):
                for z in range(self.z_dim):
                    # generate decay signal 
                    signal[x,y,z,:] = (
                        self.mwf[x,y,z] * np.exp(-(1/t2_my)*te) 
                        + self.iewf[x,y,z] * np.exp(-(1/t2_ie)*te)
                        + self.fwf[x,y,z] * np.exp(-(1/t2_fr)*te)
                        )
                    
                    # produce noise (zero-mean Gaussian noise on real and imaginary axes)
                    # noise variance is calculated according to snr
                    variance = 1/(self.snr[x,y,z]*((np.pi/2)**0.5))
                    noise_real[x,y,z,:] = np.random.normal(0, variance, [nte,])
                    noise_img[x,y,z,:] = np.random.normal(0, variance, [nte,])
                    # add noise to signal
                    signal[x,y,z,:] = (
                        (signal[x,y,z,:]+noise_real[x,y,z,:])**2 + noise_img[x,y,z,:]**2
                        )**0.5

        return signal, noise_real, noise_img

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

    phantom = phantom_exp_3pools(
        dims=[100, 100, 100],
        mwf_range=[0, 0.5],
        fwf_range=[0.05, 0.05],
        snr_range=[50, 500],
        t2s=[0.01, 0.05, 0.25],
        te=np.arange(0.002, 0.05, 0.002),
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time:0.3f}\n")
    #print(phantom.noise_img[50,50,50,:])
