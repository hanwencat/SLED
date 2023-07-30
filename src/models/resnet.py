from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Add, Input

def build_resnet(config):
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
    inputs = Input(shape=(config['input_shape'],))
    x = inputs
    for layer_size in config['hidden_layers_head']:
        x = Dense(layer_size, activation=config['activation'])(x)
    for i in range(config['num_res_blocks']):
        x = residual_block(x, config['res_block_size'], activation=config['activation'])
    for layer_size in config['hidden_layers_tail']:
            x = Dense(layer_size, activation=config['activation'])(x)
    x = Dense(config['num_classes'], activation=config['activation_last_layer'])(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    return model
    


if __name__ == "__main__":
    from train import train_model
    import yaml
    config_path = 'configs/models/resnet.yml'
    with open('config_path') as f:
        config = yaml.safe_load(f)
    model = build_resnet(config)
    

    # Load the data and train the model
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    history = train_model(model, config_path, x_train, y_train)
    
    #model.fit(x_train, y_train, epochs=config['num_epochs'], batch_size=config['batch_size'])

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
