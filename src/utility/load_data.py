import nibabel as nib
import numpy as np
import utility.image_util as iu

def load_data(config):
    """load and preprocess the data according to the config file"""
    
    # read data and its mask from file
    scan = nib.load(config['io']['data_path'])
    data_4d = scan.get_fdata()
    affine = scan.affine
    header = scan.header
    # query the config file, if there is a mask_path, then load the mask, otherwise, use the whole brain
    if config['io']['mask_path'] == None:
        mask_3d = np.ones(data_4d.shape[0:3])
    else:
        mask_3d = nib.load(config['io']['mask_path']).get_fdata()

    # data preprocessing
    data_4d = data_4d[..., 0:config['sequence']['number_of_echoes']] # select all or the first n echoes as needed
    
    if iu.check_binary(mask_3d) != True: # binarize the mask if it's not binary
        mask_3d = iu.binarize(mask_3d, config['preprocessing']['mask_threshold'])
    data_masked = iu.mask_4D_data(data_4d, mask_3d)
    data_flat, data_flat_norm = iu.flatten_filter_normalize(data_masked)
    
    if config['preprocessing']['normalization'] == True:
        data_input = data_flat_norm
        data_4d = data_4d / data_4d[..., 0:1] # normalize the 4D data, may contain zero division
        amps_scaling = 1
         
    else:
        data_input = data_flat
        amps_scaling = np.quantile(data_input, config['preprocessing']['scaling_quantile'], axis=0)[0] # for scaling the amps NN in the encoder
    
    if config['preprocessing']['remove_negative'] == True:
            data_input[data_input < 0] = 0 # remove negative values

    return data_input, data_4d, mask_3d, affine, header, amps_scaling