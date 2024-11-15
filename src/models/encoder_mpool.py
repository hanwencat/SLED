import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Activation, Add, Input, Lambda
from keras.initializers import Constant, Zeros
import yaml
import numpy as np


def build_encoder_mpool(config, amps_scaling=1):
    
    # Set up the model input
    x = Input(shape=(config['input_shape'],))

    # use 1 NN to estimate the many pool amplitudes
    if config['base_nn_amps']['name'] == 'mlp':
        amps = mlp(config['base_mlp_amps'], x) * amps_scaling
    if config['base_nn_amps']['name'] == 'resnet':
        amps = resnet(config['base_resnet_amps'], x) * amps_scaling

    # Directly output the fixed logarithmically spaced T2* values regardless of the input
    t2s_values = np.exp(
        np.linspace(
            np.log(config['t2s_range'][0]), 
            np.log(config['t2s_range'][1]), 
            config['latent_shape']
        )
    )
    
    # # Add a lambda layer to output the fixed T2* values
    # t2s = Lambda(lambda x: tf.constant(t2s_values, dtype=tf.float32))(x)
    
    # Add a dense layer with zero weights and fixed biases
    t2s = Dense(config['latent_shape'], use_bias=True, trainable=False,
                    kernel_initializer=Zeros(),  # Weights are zeroed
                    bias_initializer=Constant(t2s_values)  # Biases are fixed to logarithmic_samples
                )(x)

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
    config_path = 'configs/mpool.yaml' 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    encoder = build_encoder_mpool(config['model']['encoder'], amps_scaling=1)
    encoder.summary()