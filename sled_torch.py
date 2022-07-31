import torch 
import torch.nn as nn
import numpy as np
import time
from cmath import inf

# Construct SLED

def mlp_block(mlp_size):
    """
    the neural network building blocks
    """
    
    layers=[]
    for in_f, out_f in zip(mlp_size[:-2], mlp_size[1:-1]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(nn.BatchNorm1d(out_f))
        # layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())
    layers.append(nn.Linear(mlp_size[-2], mlp_size[-1]))
    # layers.append(nn.ReLU())
    layers.append(nn.Sigmoid())
    
    return nn.Sequential(*layers)
    

class encoder_3pool(nn.Module):
    """
    encoder takes multi-echo data and output latent parameters: t2s and amps of each water pool
    """

    def __init__(self, mlp_size_t2s, mlp_size_amps, range_t2_my, range_t2_ie, range_t2_fr, amps_scaling=1):
        super().__init__()
        self.mlp_size_t2s = mlp_size_t2s
        self.mlp_size_amps = mlp_size_amps
        self.range_t2_my = range_t2_my
        self.range_t2_ie = range_t2_ie
        self.range_t2_fr = range_t2_fr
        self.amps_scaling = amps_scaling
        
        # use 3 MLPs to encode mgre data and output each pool's t2 time
        self.t2_my = mlp_block(mlp_size_t2s)
        self.t2_ie = mlp_block(mlp_size_t2s)
        self.t2_fr = mlp_block(mlp_size_t2s)
        
        # use 1 MLPs to encode mgre data and output amplitudes of 3 pools
        self.amps = mlp_block(mlp_size_amps)
        
        #self.double() # double precision (float64) is usually not required
    
    def print_parameters(self):
        print (f"""
            mlp for t2s: {self.mlp_size_t2s}, 
            mlp for amps:{self.mlp_size_amps}, 
            t2_my range: {self.range_t2_my}s, 
            t2_ie range: {self.range_t2_ie}s, 
            t2_fr range: {self.range_t2_fr}s,
            amps scaling: {self.amps_scaling}
            """)
    
    
    def forward(self, x):
        # t2s are constrained by customized ranges
        t2_my = self.t2_my(x) * (self.range_t2_my[1] - self.range_t2_my[0]) + self.range_t2_my[0]
        t2_ie = self.t2_ie(x) * (self.range_t2_ie[1] - self.range_t2_ie[0]) + self.range_t2_ie[0]
        t2_fr = self.t2_fr(x) * (self.range_t2_fr[1] - self.range_t2_fr[0]) + self.range_t2_fr[0]
        t2s = torch.cat((t2_my,t2_ie, t2_fr), 1)
        
        # amplitude is scaled 
        amps = self.amps_scaling * self.amps(x)
        
        return t2s, amps


class decoder_vpool(nn.Module):
    """ 
    class of decoder with different snr ranges
    decoder takes latent parameters (t2s and amps of each water pool) and outputs multi-echo data
    """
    
    def __init__(self, snr_range):
        super().__init__()
        self.snr_range = snr_range
    
    def print_parameters(self):
        print(f'snr_range used in decoder: {self.snr_range}')
    
    def produce_signal(self, te, t2s_all, amps_all):
        # produce decay signal via nested for loops (slow)
        signal = torch.zeros(t2s_all.shape[0], te.shape[0])
        for i, (t2s, amps) in enumerate(zip(t2s_all, amps_all)):
            for t2, amp in zip(t2s, amps):
                signal[i, :] +=  torch.exp(-te/t2) * amp
        
        return signal
    
    def produce_signal_vectorized(self, te, t2s_all, amps_all):
        # produce decay signal model via matrix vectorization (fast)
        # vectorize parameters
        te = te[None, :, None]
        t2s_all = t2s_all[:, None, :]
        amps_all = amps_all[:, :, None]
        
        # calculate the kernel matrix and generate the signal 
        kernel_matrix = torch.exp(-te/t2s_all)
        signal = torch.squeeze(torch.matmul(kernel_matrix, amps_all))
        
        return signal
    
    def add_noise(self, signal):
        # add noise to signal according to selected snr
        # uniformly sample snr within snr_range for each signal
        snr = torch.distributions.uniform.Uniform(
            self.snr_range[0], self.snr_range[1]).sample((signal.shape[0],1))
        # scaling factor is the mean intensity of the first echo
        scaling = torch.mean(signal, 0)[0]
        # calculate variance (https://www.statisticshowto.com/rayleigh-distribution/)
        variance = scaling * 1/(snr * torch.sqrt(torch.tensor(torch.pi/2)))
        variance = variance.repeat(1, signal.shape[1]) # match the signal size
        # generate gaussian noise for real and imaginary channels
        noise_real = torch.normal(0, variance)
        noise_img = torch.normal(0, variance)
        # add noise to signal assuming signal along real axis
        noisy_signal = ((noise_real+signal)**2 + noise_img**2)**0.5 
        
        return noisy_signal


class sled(nn.Module):
    """
    self-labelled encoder decoder for multi-echo MRI data
    """
    
    def __init__(self, te, encoder, decoder):
        super().__init__()
        self.te = te
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        t2s, amps = self.encoder(x)
        x = self.decoder.produce_signal_vectorized(self.te, t2s, amps)
        x = self.decoder.add_noise(x)
        return x
    
    def latent_maps(self, image, device=torch.device("cpu")):
        # produce latent parameter maps from the encoded layer
        # use cpu at default if gpu doesn't have enough memory for the whole image
        image = torch.tensor(image, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            self.to(device)
            x = image.reshape(-1, image.shape[-1])
            t2s, amps = self.encoder(x)
            amps = amps/amps.sum(dim=-1, keepdim=True) # scale to have sum of 1
        return (t2s.reshape(*image.shape[:-1], t2s.shape[-1]), 
                amps.reshape(*image.shape[:-1], t2s.shape[-1]))
    


def train_model(model,
                device,
                train_loader,
                loss_fn,
                optimizer,
                lr_scheduler,
                epochs,
                load_best_model=True,
                return_loss_time=False):
    """
    training model with pre-defined loss function and optimizer
    """

    start_time = time.time() 
    loss_epoch = []
    best_loss = torch.tensor(inf)
    best_epoch = epochs

    for epoch in range(epochs):
        # training mode
        model.train()
        #print(f"Epoch {epoch+1}\n-------------------------------")
        
        for batch, (xb, yb) in enumerate(train_loader):
                # to device (gpu)
                xb = xb.to(device)
                yb = yb.to(device)
                
                # forward pass the model
                output = model(xb)

                # calculate the loss
                loss_batch = []
                loss = loss_fn(output, yb)
                #print(f"[{(batch+1)*len(xb):>5d}/{size:>5d}]   loss: {loss.item():>0.6f}   lr: {optimizer.param_groups[0]['lr']:>0.6f}")
                loss_batch.append(loss.detach())
                    
                # optimize the model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        loss_epoch.append(sum(loss_batch)/len(loss_batch))
        print(f"Epoch {epoch+1:2}:   learning rate = {optimizer.param_groups[0]['lr']:>0.6f}   average loss = {loss_epoch[-1]:0.6f}")
        lr_scheduler.step(loss)

        if loss_epoch[-1] < best_loss:
            best_loss = loss_epoch[-1]
            best_epoch = epoch+1
            best_state_dict = model.state_dict()

    # load the best model at epoch of the lowest loss during training
    if load_best_model == True:
        model.load_state_dict(best_state_dict)
        print(f"best model found at epoch = {best_epoch} is loaded")
    
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f'Total training time: {elapsed_time:0.3f} seconds')
    
    if return_loss_time == True:
        return loss_epoch, best_epoch, elapsed_time



# Load data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
# import scipy

def load_data(image_path, mask_path, ETL=24):
    """
    load nifti dataset and brain mask, return masked image as numpy array
    """
    
    image =  nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    # uncomment the next line if mask erosion is needed.
    # mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=3).astype(mask.dtype)
    mask_4d = np.repeat(mask[:, :, :, np.newaxis], ETL, axis=3)
    mask_4d[mask_4d==0] = np.nan
    masked_image = image*mask_4d
    
    return image, mask, masked_image


def preprocess_data(data):
    """
    flaten 4D dataset and normalize
    """

    data_flat = data.reshape(-1, data.shape[-1])
    data_flat = data_flat[~np.isnan(data_flat)].reshape(-1, data_flat.shape[1])
    data_flat_norm = data_flat/(data_flat[:,0].reshape(data_flat.shape[0],1))
    data_flat_norm = data_flat_norm.astype('float32')

    return data_flat, data_flat_norm



# Main program
if __name__ == "__main__":
    # load data and preprocess
    data_path = '/export01/data/Hanwen/SAME-ECOS/SAME-ECOS_code/mGRE_T2star.nii'
    mask_path = '/export01/data/Hanwen/SAME-ECOS/SAME-ECOS_code/mGRE_T2star_bet_mask.nii.gz'
    image, mask, data = load_data(data_path, mask_path)
    data_flat, data_flat_norm = preprocess_data(data)

    # cast into torch tensor format
    x = torch.tensor(data_flat, dtype=torch.float32)
    train_data = TensorDataset(x, x)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)

    # construct sled model
    # define t2 range of each water pool
    range_t2_my = [0.005, 0.015]
    range_t2_ie = [0.045, 0.06]
    range_t2_fr = [0.1, 0.2]

    # to device is needed, otherwise error will be raised
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    te = torch.arange(0.002, 0.0481, 0.002).to(device)
    snr_range = torch.tensor([50, 300], dtype=torch.float32).to(device)

    # amps scaling factor
    scaling = torch.quantile(x, 0.98, dim=0)[0] # scaling factor is larger than ~98% first echo intensities.
    #scaling = 10

    encoder_3pool_model = encoder_3pool(
        [24, 256, 128, 1], 
        [24, 256, 256, 3], 
        range_t2_my, 
        range_t2_ie, 
        range_t2_fr, 
        amps_scaling=scaling,
        )
    decoder_vpool_model = decoder_vpool(snr_range)
    sled_3pool = sled(te, encoder_3pool_model, decoder_vpool_model)
    sled_3pool.to(device)

    # training the sled model
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(sled_3pool.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6,
        )
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # loss_epoch = []
    # size = len(train_loader.dataset)

    train_model(
        sled_3pool, 
        device, 
        train_loader, 
        loss_fn, 
        optimizer, 
        lr_scheduler, 
        epochs=10,
        load_best_model=True,
        return_loss_time=False,
        )
    
    # produce metric maps
    t2s_maps, amps_maps = sled_3pool.latent_maps(data)


