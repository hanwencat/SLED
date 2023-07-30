from tensorflow import keras
import tensorflow.keras.optimizers as optimizers
import time
from datetime import datetime
import logging
from tensorflow.keras.callbacks import Callback


def train_model(model, config, x, y):
    
    # Configure the logging settings
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f'Experiment for subject: {config["name"]} begins at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
    logging.info('Model configurations:')
    for key, value in config.items():
        logging.info(f'{key}: {value}')

    # Get the optimizer name and parameters from the YAML file
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config.pop('name')
    
    # Create the optimizer object based on the name and parameters
    optimizer_class = getattr(optimizers, optimizer_name.capitalize())
    optimizer = optimizer_class(**optimizer_config) 
    
    # Compile the model 
    model.compile(
        loss=config['loss_function'],
        optimizer=optimizer,
        metrics=config['metric'],
    )

    callbacks_list = [
        keras.callbacks.TensorBoard(log_dir=config['TensorBoard_log_path'], 
                                    histogram_freq=config['TensorBoard_hist_freq']),
        keras.callbacks.EarlyStopping(monitor=config['EarlyStopping_monitor'], 
                                      patience=config['EarlyStopping_patience']),
        keras.callbacks.ReduceLROnPlateau(monitor=config['ReduceLROnPlateau_monitor'], 
                                          factor=config['ReduceLROnPlateau_factor'], 
                                          patience=config['ReduceLROnPlateau_patience']),
        keras.callbacks.ModelCheckpoint(filepath=config['save_model_path'],
                                        monitor=config['Checkpoint_monitor'],
                                        save_best_only=config['save_best_only']),
        CustomCallback(),
    ]

    # train the model
    start_time = time.time()
    logging.info(f'Training in progress')
    
    history = model.fit(
        x, 
        y,
        shuffle=config['shuffle'], 
        epochs=config['epochs'], 
        batch_size=config['batch_size'], 
        callbacks=callbacks_list,
        verbose=config['verbose'],
        )

    logging.info(f'Training finished, elapsed time: {time.time() - start_time:.2f} seconds')
    logging.info(f'model is saved in \'{config["save_model_path"]}\'\n')

    return history 


class CustomCallback(Callback):
    """Keras callback to log the learning rate, loss, and accuracy during training."""
    # TODO: The custom callback got warning: Callback method `on_train_batch_end` is slow compared to the batch time 

    # def on_epoch_begin(self, epoch, logs=None):
    #     lr = self.model.optimizer.lr.numpy().item()
    #     logging.debug(f'Epoch {epoch} - Learning rate: {lr:.6f}')
    
    def on_epoch_end(self, epoch, logs=None):
        formatted_logs = {key: f"{value:.6f}" for key, value in logs.items()}
        logging.debug(f"Epoch {epoch+1} - {formatted_logs} ")

    