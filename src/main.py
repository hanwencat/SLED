import yaml
import keras
from models.encoder import build_encoder
from models.decoder_exp import build_decoder_exp
from models.decoder_epg import build_decoder_epg
from models.sled import build_sled
from simulation.multi_exp_decay import generate_pretrain_data
from train.custom_train_sled import custom_train_sled
from utility.load_data import load_data 
from utility.postprocessing import calculate_maps, save_results

def main():
    # get configuration for the fitting from the config file
    with open('configs/hyperfine_defaults.yaml') as f:
        config = yaml.safe_load(f)
    
    # load and preproces data
    data_input, data_4d, mask_3d, affine, header, amps_scaling = load_data(config)
    data_input = data_input[1000:1200, :] # for testing purpose
    
    # build SLED
    encoder = build_encoder(config['model']['encoder'], amps_scaling)
    if config['fitting']['decay_model'] == 'epg':
        decoder = build_decoder_epg(config['model']['decoder'])
    else:
        decoder = build_decoder_exp(config['model']['decoder'])
    sled = build_sled(encoder=encoder, decoder=decoder, config=config['model']['sled'])
    sled.summary(expand_nested=True)
    
    # # Train SLED with synthetic data
    # if config['fitting']['pretrain_model'] == True:
    #     data_sim = generate_pretrain_data(config['pretrain'])
    #     y = {
    #         'multiecho': data_sim['decays'],
    #         'amps': data_sim['amplitudes'] if config['fitting']['fitting_model']=='parametric' else data_sim['amps_spectrum'],
    #         'sigma': data_sim['variance'],
    #     }
    #     custom_train_sled(sled, data_sim['decays'], y, config['training'])

    # # Train SLED with real data
    # y = {'multiecho': data_input} # make y a label dictionary
    # custom_train_sled(sled, data_input, y, config['training'])
    
    # calculate maps once SLED is trained
    results = calculate_maps(sled, data_4d, mask_3d, config['postprocessing'])

    # save calcualted maps to nifti files
    save_results(results, affine, header, config)

    # clear session
    keras.backend.clear_session()


if __name__ == '__main__':  
    main()
