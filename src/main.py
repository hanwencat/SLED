import nibabel as nib
import yaml
import utility.image_util as iu
from models.encoder_3pool import build_encoder_3pool, apply_encoder
from models.decoder_exp import build_decoder_exp
from models.sled import build_sled
from models.train import train_model
import keras
import numpy as np


def main(config):
    # read data from file
    scan = nib.load(config['io']['data_path'])
    data_4d = scan.get_fdata()
    affine = scan.affine
    header = scan.header
    # query the config file, if there is a mask_path, then load the mask, otherwise, use the whole brain
    if config['io']['mask_path'] == None or False:
        mask_3d = np.ones(data_4d.shape[0:3])
    else:
        mask_3d = nib.load(config['io']['mask_path']).get_fdata()

    # data preprocessing
    if iu.check_binary(mask_3d) != True: # binarize the mask if it's not binary
        mask_3d = iu.binarize(mask_3d, config['preprocessing']['mask_threshold'])
    data_masked = iu.mask_4D_data(data_4d, mask_3d)
    data_flat, data_flat_norm = iu.flatten_filter_normalize(data_masked)
    if config['preprocessing']['normalization'] == True:
        data_input = data_flat_norm
    else:
        data_input = data_flat
    amps_scaling = np.quantile(data_input, config['preprocessing']['scaling_quantile'], axis=0)[0] # for scaling the amps NN in the encoder

    # build SLED
    encoder = build_encoder_3pool(config['model']['encoder'], amps_scaling)
    decoder = build_decoder_exp(config['model']['decoder'])
    sled = build_sled(encoder=encoder, decoder=decoder)
    sled.summary()

    # train SLED with preprocessed data
    train_model(sled, config['training'], data_input, data_input)
    # TODO load the best model (need to be confirmed)
    if config['training']['save_best_only']:
        sled.load_weights(config['training']['save_model_path'])

    # extract latent parameter maps after training
    t2s_map, amps_map = apply_encoder(encoder, data_4d)
    amps_map = iu.amps_sum2one(amps_map)
    mwf_map = iu.mwf_production(t2s_map, amps_map, config['postprocessing']['mwf_cutoff'])

    # save parameter maps to nifti files and dump the configs as a nifti extension (code=6 specifies a comment as a convention) 
    extension = nib.nifti1.Nifti1Extension(6, yaml.dump(config).encode()) # https://nipy.org/nibabel/devel/biaps/biap_0003.html
    header.extensions.append(extension)
    header['descrip']=config['io']['descrip']
    nib.save(nib.Nifti1Image(t2s_map, affine, header), config['io']['save_path']+'t2s.nii.gz')
    nib.save(nib.Nifti1Image(amps_map, affine, header), config['io']['save_path']+'amps.nii.gz')
    nib.save(nib.Nifti1Image(mwf_map, affine, header), config['io']['save_path']+'mwf.nii.gz')

    # clear session
    keras.backend.clear_session()


if __name__ == '__main__':

    with open('configs/hyperfine.yml') as f:
        config = yaml.safe_load(f)
    
    main(config)
