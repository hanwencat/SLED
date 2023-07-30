import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Activation, Add, Input
import yaml
import numpy as np


def build_encoder_3pool(config, amps_scaling):
    
    # Set up the model architecture
    inputs = Input(shape=(config['input_shape'],))
    x = inputs

    # use 3 NNs to estimate 3 t2 times
    if config['base_nn_t2s']['name'] == 'mlp':
        t2_my = mlp(config['base_mlp_t2'], x)
        t2_ie = mlp(config['base_mlp_t2'], x)
        t2_fr = mlp(config['base_mlp_t2'], x)
    
    if config['base_nn_t2s']['name'] == 'resnet':
        t2_my = resnet(config['base_resnet_t2'], x)
        t2_ie = resnet(config['base_resnet_t2'], x)
        t2_fr = resnet(config['base_resnet_t2'], x)

    # constrain t2s in corresponding ranges
    t2_my = t2_my * (config['range_t2_my'][1] - config['range_t2_my'][0]) + config['range_t2_my'][0]
    t2_ie = t2_ie * (config['range_t2_ie'][1] - config['range_t2_ie'][0]) + config['range_t2_ie'][0]
    t2_fr = t2_fr * (config['range_t2_fr'][1] - config['range_t2_fr'][0]) + config['range_t2_fr'][0]

    # group 3 t2 times into t2s
    t2s = tf.concat([t2_my, t2_ie, t2_fr], axis=1)

    # use 1 NN to estimate 3 amplitudes
    if config['base_nn_amps']['name'] == 'mlp':
        amps = mlp(config['base_mlp_amps'], x) * amps_scaling
    if config['base_nn_amps']['name'] == 'resnet':
        amps = resnet(config['base_resnet_amps'], x) * amps_scaling

    # build the encoder model
    encoder =  keras.Model(x, [t2s, amps], name = "encoder")
    
    # name the two output layers
    encoder.layers[-2]._name = 't2s'
    encoder.layers[-1]._name = 'amps'
    
    return encoder


def mlp(config, x):
    # Set up the model architecture
    for layer_size in config['hidden_layers']:
        x = Dense(layer_size, activation=config['activation'])(x)
    x = Dense(config['num_classes'], activation=config['activation_last_layer'])(x)
    
    return x


def resnet(config, x):
    # Define a residual block
    def residual_block(x, units_list, activation='relu'):
        shortcut = x
        for i, units in enumerate(units_list):
            x = Dense(units)(x)
            x = BatchNormalization()(x)
            if i == len(units_list) - 1:
                # Skip activation for the last layer
                if shortcut.shape[-1] != units:
                    shortcut = Dense(units)(shortcut)
                x = Add()([shortcut, x])
            else:
                x = Activation(activation)(x)
        return x

    # Set up the model architecture
    for layer_size in config['hidden_layers_head']:
        x = Dense(layer_size, activation=config['activation'])(x)
    for i in range(config['num_res_blocks']):
        x = residual_block(x, config['res_block_size'], activation=config['activation'])
    for layer_size in config['hidden_layers_tail']:
            x = Dense(layer_size, activation=config['activation'])(x)
    x = Dense(config['num_classes'], activation=config['activation_last_layer'])(x)
    
    return x


# def apply_encoder(encoder, volume):
#     """Apply the trained encoder to process the volume dataset (exclude NaN and zero voxels)"""
    
#     flattened_volume = volume.reshape(-1, volume.shape[-1])
#     mask = np.isnan(flattened_volume) | (flattened_volume == 0)
#     valid_indices = ~(mask.any(axis=-1))
#     valid_flattened_volume = flattened_volume[valid_indices]

#     t2s, amps = encoder.predict(valid_flattened_volume)

#     output_shape_flat = flattened_volume.shape[:-1] + (t2s.shape[-1],)
#     t2s_map_flat = np.zeros(output_shape_flat)
#     amps_map_flat = np.zeros(output_shape_flat)

#     t2s_map_flat[valid_indices] = t2s
#     amps_map_flat[valid_indices] = amps

#     output_shape = volume.shape[:-1] + (t2s.shape[-1],)
#     t2s_map = t2s_map_flat.reshape(output_shape)
#     amps_map = amps_map_flat.reshape(output_shape)

#     return t2s_map, amps_map


def apply_encoder(encoder, volume):
    """Apply the trained encoder"""
    flattened_volume = volume.reshape(-1, volume.shape[-1])
    t2s, amps = encoder.predict(flattened_volume)

    t2s_map = t2s.reshape(volume.shape[:-1] + (t2s.shape[-1],))
    amps_map = amps.reshape(volume.shape[:-1] + (amps.shape[-1],))

    return t2s_map, amps_map


if __name__ == "__main__":
    
    # Load hyperparameters from YAML config file
    config_path = 'configs/defaults.yml' 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    encoder = build_encoder_3pool(config['model']['encoder'], amps_scaling=1)
    encoder.summary()