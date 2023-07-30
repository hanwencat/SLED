from tensorflow import keras
from tensorflow.keras.layers import Dense, Input

def build_mlp(config):
    # Set up the model architecture
    inputs = Input(shape=(config['input_shape'],))
    x = inputs
    for layer_size in config['hidden_layers']:
        x = Dense(layer_size, activation=config['activation'])(x)
    x = Dense(config['num_classes'], activation=config['activation_last_layer'])(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    return model
    
    