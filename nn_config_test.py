import sled_torch as sd
import digital_phantom as pt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import time


# def phantom_make_3d(
#     snr_range, 
#     mwf_range, 
#     exwf_range, 
#     t2s, 
#     x_dim, 
#     y_dim, 
#     z_dim, 
#     echo_time,
#     ):
#     """
#     make 3d phantom
#     """

#     # create placeholders
#     snr = np.zeros([x_dim, y_dim, z_dim])
#     mwf = np.zeros([x_dim, y_dim, z_dim])
#     axwf = np.zeros([x_dim, y_dim, z_dim])
#     exwf = np.zeros([x_dim, y_dim, z_dim])

#     num_echo = len(echo_time)
#     signal = np.zeros([x_dim, y_dim, z_dim, num_echo])
#     noise = np.zeros([x_dim, y_dim, z_dim, num_echo])

#     # produce the ground truth snr map
#     for y in range(y_dim):
#         snr[:, y, :] = snr_range[0] + y*(snr_range[1]-snr_range[0])/y_dim

#     # produce the ground truth mwf map
#     for x in range(x_dim):
#         mwf[x, :, :] = mwf_range[1] - x*(mwf_range[1]-mwf_range[0])/x_dim
        
#     # produce the ground truth exwf map
#     for z in range(z_dim):
#         exwf[:, :, z] = exwf_range[0] + z*(exwf_range[1]-exwf_range[0])/z_dim
    
#     # produce the ground truth axwf map
#     axwf = 1 - mwf - exwf
    
#     # produce the decay signals (no T1 compensation)
#     t2_my = t2s[0]
#     t2_ax = t2s[1]
#     t2_ex = t2s[2]

#     for x in range(x_dim):
#         for y in range(y_dim):
#             for z in range(z_dim):
#                 # produce noise for each echo according to SNR (independent Gaussian noise on real and imaginary axes)
#                 noise_mu = 0  # noise mean is 0
#                 # noise variance is calculated according to snr
#                 noise_sigma = 1/(snr[x, y, z]*((np.pi/2)**0.5))
#                 noise[x, y, z, :] = np.squeeze(
#                     abs((np.random.normal(
#                         noise_mu, noise_sigma, [num_echo, 1])
#                                       + 1j*np.random.normal(
#                                         noise_mu, noise_sigma, [num_echo, 1]))))

#                 # generate signal with noise added
#                 signal[x, y, z, :] = (mwf[x, y, z] * np.exp(-(1/t2_my)*echo_time) 
#                                    + axwf[x, y, z] * np.exp(-(1/t2_ax)*echo_time)
#                                    + exwf[x, y, z] * np.exp(-(1/t2_ex)*echo_time)
#                                    + noise[x, y, z, :]
#                                   )
    
#     return signal, mwf, axwf, exwf, snr, noise



if __name__ == "__main__":
    # ## make the phantom using the following parameters
    # snr_range = [50, 500]
    # mwf_range = [0, 0.5]
    # exwf_range = [0.05, 0.05]
    # t2s = [0.01,0.05,0.25] # unit: seconds
    # x_dim = 100
    # y_dim = 100
    # z_dim = 100
    # echo_time = np.arange(0.002, 0.05, 0.002)

    # class phantom:
    #     pass
    # phantom.signal, phantom.mwf, phantom.axwf, phantom.exwf, phantom.snr, phantom.noise = phantom_make_3d(
    #     snr_range, 
    #     mwf_range, 
    #     exwf_range, 
    #     t2s, 
    #     x_dim, 
    #     y_dim, 
    #     z_dim, 
    #     echo_time,
    #     )

    # phantom.data_flat, phantom_data_norm = sd.preprocess_data(phantom.signal)
    # # print(phantom.data_flat.shape)      
    

    phantom = pt.phantom_3pools(
        dims=[100, 100, 100],
        mwf_range=[0, 0.5],
        fwf_range=[0.05, 0.05],
        snr_range=[50, 500],
        t2s=[0.01, 0.05, 0.25],
        te=np.arange(0.002, 0.05, 0.002),
    )

    phantom.data_flat, phantom_data_norm = sd.preprocess_data(phantom.signal)

    # load phantom data
    x = torch.tensor(phantom.data_flat, dtype=torch.float32)
    train_data = TensorDataset(x, x)
    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)


    ## construct sled model
    # define t2 range of each water pool
    range_t2_my = [0.005, 0.015]
    range_t2_ie = [0.045, 0.06]
    range_t2_fr = [0.2, 0.3]

    # to device is needed, otherwise error will be raised
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    te = torch.arange(0.002, 0.0481, 0.002).to(device)
    snr_range = torch.tensor([50, 500], dtype=torch.float32).to(device)

    # create different hidden layers
    hidden_layer_all = (
        [256],
        [128,64],
        [256,128],
        [128,256,128],
        [256,256,128,64],
    )

    # write results to a csv file
    import csv
    csv_file = "nn_config_test_results.csv"
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(hidden_layer_all))
 
    ## repeated tests
    test_reps = 1    
    test_start_time = time.time()
    
    for rep in range(test_reps):
        loss_all = []
        #training_time_all = []
   
    # test different hidden layers
        for hidden_layer in hidden_layer_all:
            print(f"hidden layers: {hidden_layer}")
            mlp_size_t2s = [24] + hidden_layer + [1]
            mlp_size_amps = [24] + hidden_layer + [3]
            
            # construct sled model
            encoder_3pool_model = sd.encoder_3pool(
                mlp_size_t2s, 
                mlp_size_amps,
                range_t2_my, 
                range_t2_ie, 
                range_t2_fr, 
                amps_scaling=1,
                )
            
            decoder_vpool_model = sd.decoder_vpool(snr_range)
            sled_3pool = sd.sled(
                te, 
                encoder_3pool_model, 
                decoder_vpool_model,
                )
            sled_3pool.to(device)

            # training sled model
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adamax(
                sled_3pool.parameters(), 
                lr=0.001
                )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                factor=0.5, 
                patience=2, 
                min_lr=1e-6,
                )

            loss, training_time = sd.train_model(
                sled_3pool,
                device, 
                train_loader, 
                loss_fn, 
                optimizer, 
                lr_scheduler, 
                epochs=25,
                return_loss_time=True,
                )
            
            loss_all.append(loss[-1].cpu().numpy().astype(np.float32))
            # training_time_all.append(training_time)
            print(f"loss: {loss[-1]:0.6f}\n")

        with open(csv_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(loss_all)
            # writer.writerow(training_time_all)

    test_end_time = time.time() 
    elapsed_time = test_end_time - test_start_time
    print(f'Total test time: {elapsed_time:0.3f} seconds\n')
