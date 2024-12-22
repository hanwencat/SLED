from keras.models import Model
import tensorflow as tf
import yaml


def build_sled(encoder, decoder):
    """
    Builds a SLED model by connecting an encoder and decoder model.

    Args:
        encoder: A Keras model that takes in an input and outputs a compressed representation.
        decoder: A Keras model that takes in a compressed representation and outputs a reconstructed output.

    Returns:
        A Keras model that connects the encoder and decoder.
    """

    # Define the inputs and outputs of the model
    input = encoder.inputs
    t2s, amps, sigma = encoder.output['t2s'], encoder.output['amps'], encoder.output['sigma']
    fitted_signals = decoder([t2s, amps])

    # Concatenate along the last axis (-1)
    fitted_signals_with_sigma = tf.concat([fitted_signals, sigma], axis=-1)
    
    # Create a Keras model that connects the encoder and decoder
    sled = Model(
        inputs=input, 
        outputs={'fitted_signals_with_sigma':fitted_signals_with_sigma, 't2s':t2s, 'amps':amps}, 
        name='SLED',
        )

    return sled


def apply_sled_to_volume(sled, volume):
    """
    Apply the trained SLED model to a 4D volume.
    
    Args:
        sled: The trained SLED model. It should output a dictionary with
              keys 'fitted_signals', 't2s', and 'amps'.
        volume: A 4D numpy array of shape (X, Y, Z, T).

    Returns:
        fitted_signals_map: A numpy array of shape (X, Y, Z, T) with the fitted signals.
        t2s_map: A numpy array of shape (X, Y, Z, num_classes) with T2 times.
        amps_map: A numpy array of shape (X, Y, Z, num_classes) with amplitudes.
        sigma_map: A numpy array of shape (X, Y, Z, 1) with the noise standard deviation.
    """
    # Flatten the volume from (X, Y, Z, T) to (N, T)
    # where N = X * Y * Z
    flattened_volume = volume.reshape(-1, volume.shape[-1])
    
    # Predict using the SLED model
    preds = sled.predict(flattened_volume, verbose=0)
    
    # Extract predictions
    fitted_signals_with_sigma = preds['fitted_signals_with_sigma']  # shape: (N, T)
    t2s = preds['t2s']                        # shape: (N, num_classes)
    amps = preds['amps']                      # shape: (N, num_classes)
    
    fitted_signals = fitted_signals_with_sigma[..., :-1]  # shape: (N, T)
    sigma = fitted_signals_with_sigma[...,-1]                    # shape: (N, 1)
    
    # Reshape back to original volume dimensions
    # fitted_signals_map: (X, Y, Z, T)
    fitted_signals_map = fitted_signals.reshape(volume.shape)
    
    # t2s_map: (X, Y, Z, num_classes)
    t2s_map = t2s.reshape(volume.shape[:-1] + (t2s.shape[-1],))
    
    # amps_map: (X, Y, Z, num_classes)
    amps_map = amps.reshape(volume.shape[:-1] + (amps.shape[-1],))
    
    # sigma_map: (X, Y, Z, 1)
    sigma_map = sigma.reshape(volume.shape[:-1] + (1,))

    return fitted_signals_map, t2s_map, amps_map, sigma_map


if __name__ == "__main__":
    from encoder_mpool import build_encoder_mpool
    from decoder_exp import build_decoder_exp

    # Load hyperparameters from YAML config file
    config_path = 'configs/mpool.yaml' 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    encoder = build_encoder_mpool(config['model']['encoder'], amps_scaling=1)
    # encoder.summary()

    decoder = build_decoder_exp(config['model']['decoder'])
    # decoder.summary()

    sled = build_sled(encoder=encoder, decoder=decoder)
    sled.summary()