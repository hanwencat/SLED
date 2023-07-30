from keras.models import Model
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
    output = decoder(encoder.outputs)

    # Create a Keras model that connects the encoder and decoder
    sled = Model(inputs=input, outputs=output, name='SLED')

    return sled


if __name__ == "__main__":
    from encoder_3pool import build_encoder_3pool
    from decoder_exp import build_decoder_exp

    # Load hyperparameters from YAML config file
    config_path = 'configs/defaults.yml' 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    encoder = build_encoder_3pool(config['model']['encoder'])
    # encoder.summary()

    decoder = build_decoder_exp(config['model']['decoder'])
    # decoder.summary()

    sled = build_sled(encoder=encoder, decoder=decoder)
    sled.summary()