import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Activation, Add, Input, Lambda
from tensorflow.keras import regularizers
from keras.initializers import Constant, Zeros
import yaml
import numpy as np


def build_encoder_3pool(config, amps_scaling=1):
    
    # Set up the model input
    x = Input(shape=(config['input_shape'],))

    if config['fitting_model'] == 'nonparametric': # a non-parametric fitting like NNLS
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
        
        # Add a dense layer with fixed weights and biases
        t2s_layer = Dense(config['latent_shape'], use_bias=True,
                kernel_initializer=Zeros(),
                bias_initializer=Constant(t2s_values),
                name='t2s')
        t2s = t2s_layer(x)
        t2s_layer.trainable = False  # Make the entire layer untrainable
        
    else: # a 3-pool model
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

        #  Group 3 t2 times into t2s and assign name
        t2s = tf.keras.layers.Concatenate(name='t2s')([t2_my, t2_ie, t2_fr])

    # use 1 NN to estimate 3 amplitudes
    if config['base_nn_amps']['name'] == 'mlp':
        amps = mlp(config['base_mlp_amps'], x)
    if config['base_nn_amps']['name'] == 'resnet':
        amps = resnet(config['base_resnet_amps'], x)
    
    # Multiply by amps_scaling using a Lambda layer and assign name 'amps'
    amps = Lambda(lambda x: x * amps_scaling, name='amps')(amps)
    
    # use 1 NN to estimate the noise variance sigma
    if config['base_nn_amps']['name'] == 'mlp':
        sigma = mlp(config['base_mlp_sigma'], x, name='sigma')
    if config['base_nn_amps']['name'] == 'resnet':
        sigma = resnet(config['base_resnet_sigma'], x, name='sigma')
    if config['fix_sigma'] == True:
        sigma = sigma * 0 + config['sigma_value']
    # Multiply by amps_scaling using a Lambda layer and assign name 'sigma'
    # sigma = Lambda(lambda x: x * amps_scaling, name='sigma')(sigma)
    
    # Build the encoder model with named outputs
    encoder = keras.Model(inputs=x, outputs={'t2s': t2s, 'amps': amps, 'sigma': sigma}, name="encoder")
    
    return encoder


def mlp(config, x, name=None):
    # Set up the model architecture
    for layer_size in config['hidden_layers']:
        x = Dense(layer_size, activation=config['activation'])(x)
    x = Dense(
        config['num_classes'], 
        activation=config['activation_last_layer'], 
        kernel_regularizer=regularizers.l1(config['l1_reg']), 
        # kernel_regularizer=regularizers.l2(config['l2_reg']), 
        name=name,
        )(x)
    
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
    x = Dense(config['num_classes'], activation=config['activation_last_layer'], kernel_regularizer=regularizers.l1(config['l1_reg']))(x)
    
    return x


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