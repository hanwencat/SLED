from keras.models import Model
import numpy as np

def build_sled(encoder, decoder, config):
    """
    Build a SLED model by connecting an encoder and a decoder.
    
    Args:
        encoder: A Keras model that takes an input and outputs a dictionary of representations.
        decoder: A Keras model that expects a list of two inputs ([t2s, amps]) and outputs the reconstructed multiecho signal.
                 (It is assumed that any necessary wrapping into a nested submodel has been done outside this function.)
        config: A configuration dictionary with keys like 'decay_model', 'load_pretrained_model', etc.
    
    Returns:
        A Keras model that connects the encoder and the decoder.
    """
    # Use the encoder's input (assuming a single input tensor)
    encoder_input = encoder.inputs
    encoder_outputs = encoder(encoder_input)
    
    # Depending on the decay model, extract the appropriate outputs from the encoder.
    if config['decay_model'] == 'epg':
        t2s = encoder_outputs['t2s']
        amps = encoder_outputs['amps']
        sigma = encoder_outputs['sigma']
        fa = encoder_outputs['fa']
        multiecho = decoder([t2s, amps])
        outputs = {
            'multiecho': multiecho,
            't2s': t2s,
            'amps': amps,
            'sigma': sigma,
            'fa': fa
        }
    elif config['decay_model'] == 'exp':
        t2s = encoder_outputs['t2s']
        amps = encoder_outputs['amps']
        sigma = encoder_outputs['sigma']
        multiecho = decoder([t2s, amps])
        outputs = {
            'multiecho': multiecho,
            't2s': t2s,
            'amps': amps,
            'sigma': sigma
        }
    else:
        raise ValueError("Unknown decay_model: {}".format(config['decay_model']))
    
    # Create the final SLED model.
    sled_model = Model(inputs=encoder_input, outputs=outputs, name='SLED')
    
    # Optionally load pretrained weights.
    if config.get('load_pretrained_model', False):
        sled_model.load_weights(config['pretrained_model_path'])
        print("##################################################")
        print("Pretrained model loaded from:", config['pretrained_model_path'])
        print("##################################################")
    
    return sled_model


def apply_sled_to_volume(sled, volume):
    """
    Apply the trained SLED model to a 4D volume.
    
    Args:
        sled: The trained SLED model. It outputs a dictionary with keys 'multiecho', 't2s', 'amps', and 'sigma'.
        volume: A 4D numpy array of shape (X, Y, Z, T).
    
    Returns:
        multiecho_map: A numpy array of shape (X, Y, Z, T) with the fitted multiecho signals.
        t2s_map: A numpy array of shape (X, Y, Z, num_classes) with T2 times.
        amps_map: A numpy array of shape (X, Y, Z, num_classes) with amplitudes.
        sigma_map: A numpy array of shape (X, Y, Z, 1) with the noise standard deviation.
    """
    # Flatten the volume from (X, Y, Z, T) to (N, T), where N = X * Y * Z.
    flattened_volume = volume.reshape(-1, volume.shape[-1])
    
    # Predict using the SLED model.
    preds = sled.predict(flattened_volume, verbose=0)
    
    multiecho = preds['multiecho']  # Shape: (N, T)
    t2s = preds['t2s']              # Shape: (N, num_classes)
    amps = preds['amps']            # Shape: (N, num_classes)
    sigma = preds['sigma']          # Shape: (N, 1)
    fa = preds['fa']          # Shape: (N, 1)
    
    # Reshape predictions back to the original volume dimensions.
    multiecho_map = multiecho.reshape(volume.shape)
    t2s_map = t2s.reshape(volume.shape[:-1] + (t2s.shape[-1],))
    amps_map = amps.reshape(volume.shape[:-1] + (amps.shape[-1],))
    sigma_map = sigma.reshape(volume.shape[:-1] + (1,))
    fa_map = fa.reshape(volume.shape[:-1] + (1,))
    
    return multiecho_map, t2s_map, amps_map, sigma_map, fa_map