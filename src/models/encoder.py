import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Dense,
    BatchNormalization,
    Activation,
    Add,
    Input,
    Lambda,
    Concatenate
)
from keras import regularizers
from keras.initializers import Constant, Zeros

#####################################
#  Branch-Building Submodel Utility #
#####################################

def build_nn_submodel(network_type: str, config: dict, name_prefix: str = None, transform_fn=None) -> keras.Model:
    """
    Build a submodel for a given branch (e.g., 't2', 'amps', 'sigma', or 'fa') using either an MLP or ResNet.
    An optional transform function can be applied inside the submodel via a Lambda layer.
    
    Args:
        network_type (str): One of 't2', 'amps', 'sigma', or 'fa'.
        config (dict): Configuration dictionary containing:
            - 'nn_base': either 'mlp' or 'resnet'
            - Keys like 'base_mlp_<network_type>' or 'base_resnet_<network_type>'
        name_prefix (str, optional): Prefix for naming layers (e.g., "amps" or "t2_my").
        transform_fn (callable, optional): A function that takes a tensor and returns a transformed tensor.
    
    Returns:
        keras.Model: A nested model for the branch.
    """
    branch_input = Input(shape=(config['input_shape'],), name=f"{name_prefix}_input")
    
    # Select configuration based on the chosen base network.
    key = f"base_mlp_{network_type}" if config['nn_base'] == 'mlp' else f"base_resnet_{network_type}"
    nn_config = config[key]
    
    # Build the branch using the chosen architecture.
    if config['nn_base'] == 'mlp':
        branch_output = mlp(nn_config, branch_input, name_prefix=name_prefix)
    elif config['nn_base'] == 'resnet':
        branch_output = resnet(nn_config, branch_input, name=name_prefix)
    else:
        raise ValueError("Invalid 'nn_base' value in config.")
    
    # If a transform function is provided, apply it inside the submodel.
    if transform_fn is not None:
        branch_output = Lambda(transform_fn, name=f"{name_prefix}_transform")(branch_output)
    
    return keras.Model(inputs=branch_input, outputs=branch_output, name=name_prefix)

def build_t2_branch_submodel(config: dict) -> keras.Model:
    """
    Build a nested submodel for the T2 branch that includes three sub‑branches (t2_my, t2_ie, t2_fr),
    each with an internal Lambda layer for scaling, and then concatenates their outputs.
    
    Args:
        config (dict): Encoder configuration containing T2 range values and submodel settings.
    
    Returns:
        keras.Model: A submodel that takes an input and outputs the concatenated T2 estimates.
    """
    t2_input = Input(shape=(config['input_shape'],), name='t2s_input')
    
    # Build each T2 sub-branch with its own scaling inside the submodel.
    t2_my_model = build_nn_submodel(
        't2', config, name_prefix='t2_my',
        transform_fn=lambda t: t * (config['range_t2_my'][1] - config['range_t2_my'][0]) + config['range_t2_my'][0]
    )
    t2_ie_model = build_nn_submodel(
        't2', config, name_prefix='t2_ie',
        transform_fn=lambda t: t * (config['range_t2_ie'][1] - config['range_t2_ie'][0]) + config['range_t2_ie'][0]
    )
    t2_fr_model = build_nn_submodel(
        't2', config, name_prefix='t2_fr',
        transform_fn=lambda t: t * (config['range_t2_fr'][1] - config['range_t2_fr'][0]) + config['range_t2_fr'][0]
    )
    
    # Apply each submodel on the same T2 branch input.
    t2_my = t2_my_model(t2_input)
    t2_ie = t2_ie_model(t2_input)
    t2_fr = t2_fr_model(t2_input)
    
    # Concatenate the three outputs inside this T2 branch submodel.
    concatenated = Concatenate(name='t2s')([t2_my, t2_ie, t2_fr])
    
    return keras.Model(inputs=t2_input, outputs=concatenated, name='t2s')

##########################
#  Network Architectures #
##########################

def mlp(config: dict, x: tf.Tensor, name_prefix: str = None) -> tf.Tensor:
    """
    Build an MLP and name its hidden layers using the provided prefix.
    
    Args:
        config (dict): MLP configuration including:
            - 'hidden_layers': list of integers for hidden layer sizes.
            - 'activation': activation function for hidden layers.
            - 'num_classes': number of output neurons.
            - 'activation_last_layer': activation for the output layer.
            - 'l1_reg': L1 regularization factor.
        x (tf.Tensor): Input tensor.
        name_prefix (str, optional): Prefix for naming layers.
    
    Returns:
        tf.Tensor: Output tensor of the MLP.
    """
    for i, layer_size in enumerate(config["hidden_layers"]):
        layer_name = f"{name_prefix}_hidden_layer_{i+1}" if name_prefix else None
        x = Dense(layer_size, activation=config["activation"], name=layer_name)(x)
    
    # Name the output layer using the prefix.
    output_layer_name = f"{name_prefix}_output_layer" if name_prefix else None
    x = Dense(
        config["num_classes"],
        activation=config["activation_last_layer"],
        kernel_regularizer=regularizers.l1(config["l1_reg"]),
        name=output_layer_name,
    )(x)
    return x

def resnet(config: dict, x: tf.Tensor, name: str = None) -> tf.Tensor:
    """
    Build a ResNet-like network.
    
    Args:
        config (dict): ResNet configuration including:
            - 'hidden_layers_head': list of integers for head layers.
            - 'num_res_blocks': number of residual blocks.
            - 'res_block_size': list of integers defining each residual block.
            - 'hidden_layers_tail': list of integers for tail layers.
            - 'num_classes': number of output neurons.
            - 'activation': activation function.
            - 'activation_last_layer': activation for the output layer.
            - 'l1_reg': L1 regularization factor.
        x (tf.Tensor): Input tensor.
        name (str, optional): Name to apply (via a Lambda layer) at the final output.
    
    Returns:
        tf.Tensor: Output tensor of the ResNet.
    """
    def residual_block(x: tf.Tensor, units_list: list, activation: str = "relu") -> tf.Tensor:
        shortcut = x
        for i, units in enumerate(units_list):
            x = Dense(units)(x)
            x = BatchNormalization()(x)
            if i == len(units_list) - 1:
                if shortcut.shape[-1] != units:
                    shortcut = Dense(units)(shortcut)
                x = Add()([shortcut, x])
            else:
                x = Activation(activation)(x)
        return x

    # Head
    for layer_size in config['hidden_layers_head']:
        x = Dense(layer_size, activation=config['activation'])(x)
    # Residual blocks
    for _ in range(config['num_res_blocks']):
        x = residual_block(x, config['res_block_size'], activation=config['activation'])
    # Tail
    for layer_size in config['hidden_layers_tail']:
        x = Dense(layer_size, activation=config['activation'])(x)
    # Final output layer
    x = Dense(
        config['num_classes'],
        activation=config['activation_last_layer'],
        kernel_regularizer=regularizers.l1(config['l1_reg'])
    )(x)
    if name is not None:
        x = Lambda(lambda t: t, name=name)(x)
    return x

##############################
#     Encoder Construction   #
##############################

def build_encoder(config: dict, amps_scaling: float = 1) -> keras.Model:
    """
    Build the encoder model with nested submodels for each branch.
    
    Branches include:
      - T2 branch:
          - For 'nonparametric': a fixed Dense layer with preset T2* values.
          - Otherwise: a nested T2 submodel that internally builds and scales three T2 branches and concatenates them.
      - Amplitude branch (amps) with scaling applied inside its submodel.
      - Noise variance branch (sigma).
      - Optionally, an FA branch.
    
    Args:
        config (dict): Encoder configuration.
        amps_scaling (float): Scaling factor for the amplitude branch.
    
    Returns:
        keras.Model: The constructed encoder model.
    """
    encoder_input = Input(shape=(config['input_shape'],), name='encoder_input')
    
    # --- T2 Branch ---
    if config['fitting_model'] == 'nonparametric':
        t2s_values = np.exp(
            np.linspace(
                np.log(config['t2s_range'][0]),
                np.log(config['t2s_range'][1]),
                config['latent_shape']
            )
        )
        t2s_layer = Dense(
            config['latent_shape'],
            use_bias=True,
            kernel_initializer=Zeros(),
            bias_initializer=Constant(t2s_values),
            name='t2s'
        )
        t2s = t2s_layer(encoder_input)
        t2s_layer.trainable = False
    else:
        # Use a nested T2 branch submodel that handles scaling and concatenation internally.
        t2_branch_model = build_t2_branch_submodel(config)
        t2s = t2_branch_model(encoder_input)
    
    # --- Amplitude Branch ---
    # Build amplitude branch submodel that applies scaling internally.
    amps_model = build_nn_submodel('amps', config, name_prefix='amps', transform_fn=lambda t: t * amps_scaling)
    amps = amps_model(encoder_input)
    
    # --- Noise Variance (sigma) Branch ---
    sigma_model = build_nn_submodel('sigma', config, name_prefix='sigma')
    sigma = sigma_model(encoder_input)
    if config.get('fix_sigma', False):
        sigma = Lambda(lambda t: t * 0 + config['sigma_value'], name='sigma_fixed')(sigma)
    
    outputs = {'t2s': t2s, 'amps': amps, 'sigma': sigma}
    
    # --- FA Branch (optional) ---
    if config.get('decay_model') == 'epg':
        fa_model = build_nn_submodel('fa', config, name_prefix='fa')
        fa = fa_model(encoder_input)
        outputs['fa'] = fa
    
    return keras.Model(inputs=encoder_input, outputs=outputs, name="encoder")

#########################
#    Inference Helper   #
#########################

def apply_encoder(encoder: keras.Model, volume: np.ndarray) -> tuple:
    """
    Apply the trained encoder to a volume.
    
    Args:
        encoder (keras.Model): The encoder model.
        volume (np.ndarray): Input volume with the last dimension corresponding to features.
    
    Returns:
        tuple: (t2s_map, amps_map) reshaped to the volume's spatial dimensions.
    """
    flattened_volume = volume.reshape(-1, volume.shape[-1])
    predictions = encoder.predict(flattened_volume)
    
    t2s = predictions["t2s"]
    amps = predictions["amps"]
    
    t2s_map = t2s.reshape(volume.shape[:-1] + (t2s.shape[-1],))
    amps_map = amps.reshape(volume.shape[:-1] + (amps.shape[-1],))
    
    return t2s_map, amps_map

#########################################
# Example usage (for testing purposes)  #
#########################################
# if __name__ == "__main__":
#     with open("configs/defaults.yml", "r") as file:
#         config = yaml.safe_load(file)
#
#     encoder = build_encoder(config["model"]["encoder"], amps_scaling=1)
#     encoder.summary(expand_nested=True)