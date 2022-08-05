import sled_tf as sd
import tensorflow as tf
import digital_phantom as pt
import numpy as np
import time


if __name__ == "__main__":
    # make the phantom 
    phantom = pt.phantom_3pools(
        dims=[100, 100, 100],
        mwf_range=[0, 0.5],
        fwf_range=[0.05, 0.05],
        snr_range=[50, 500],
        t2s=[0.01, 0.05, 0.25],
        te=np.arange(0.002, 0.05, 0.002),
    )

    # preprocess phantom data
    phantom.data_flat, phantom_data_norm = sd.preprocess_data(phantom.signal)


    # define t2 range of each water pool for the sled model
    te = tf.range(0.002, 0.05, 0.002) 
    range_t2_my = [0.005, 0.015]
    range_t2_ie = [0.045, 0.06]
    range_t2_fr = [0.1, 0.2]
    # snr_range = [50., 500.]
    snr_range = []
    amps_scaling = 1

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
    csv_file = "nn_config_test_tf_results.csv"
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(hidden_layer_all))

    ## repeated tests
    test_reps = 10    
    test_start_time = time.time()
    
    for rep in range(test_reps):
        loss_all = []
        #training_time_all = []

        # test different hidden layers
        for hidden_layer in hidden_layer_all:
            print(f"hidden layers: {hidden_layer}")
            nn_layers_t2s = hidden_layer + [1]
            nn_layers_amps = hidden_layer + [3]


            # construct sled    
            encoder, sled = sd.sled_builder(
                te,
                nn_layers_t2s,
                nn_layers_amps,
                range_t2_my,
                range_t2_ie,
                range_t2_fr,
                snr_range,
                amps_scaling,
                batch_norm=False,
                )

            # train sled model
            history = sd.train_model(
                sled, 
                phantom.data_flat, 
                epochs=20,
                return_history=True,
                verbose=2
                )

            loss_all.append(history.history["loss"][-1])
            print(f"loss: {loss_all[-1]:0.6f}\n")

        with open(csv_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(loss_all)
    
    test_end_time = time.time() 
    elapsed_time = test_end_time - test_start_time
    print(f'Total test time: {elapsed_time:0.3f} seconds\n')








