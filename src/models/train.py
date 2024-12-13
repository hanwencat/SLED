import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.optimizers as optimizers
import time
from datetime import datetime
import logging
from tensorflow.keras.callbacks import Callback
# import tensorflow_probability as tfp
# from tensorflow.keras.losses import Loss


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
        # loss=config['loss'],
        loss=RicianNLLLoss(sigma=0.1, name="rician_nll_loss"),
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




# from scipy.special import i0
import numpy as np
import tensorflow as tf

def log_i0_stable(x):
    # x: tf.Tensor
    # Choose a threshold; e.g., 10 as a cutoff between "small" and "large" x
    absx = tf.abs(x)
    threshold = 10.0

    # For small x, use a truncated series expansion for i0(x):
    # i0(x) = sum_{k=0}^\infty ((x/2)^{2k}) / (k!)^2
    # We'll only do a few terms.
    def i0_series(z, terms=10):
        z2 = tf.square(z/2.0)
        result = tf.ones_like(z)
        term = tf.ones_like(z)
        factorial = 1.0
        for k in range(1, terms):
            factorial *= k
            term = term * z2 / (factorial**2)
            result = result + term
        return result

    def small_x_branch():
        val = i0_series(x, terms=10)
        # add small epsilon to avoid log(0)
        return tf.math.log(val + 1e-30)

    def large_x_branch():
        # log i0(x) ~ x - 0.5*log(2πx)
        # Add small epsilons to avoid log(0)
        return x - 0.5*tf.math.log(2.0 * np.pi * absx + 1e-30)

    return tf.where(absx < threshold, small_x_branch(), large_x_branch())

class RicianNLLLoss(tf.keras.losses.Loss):
    def __init__(self, sigma=0.02, name="rician_nll_loss"):
        super().__init__(name=name)
        self.sigma = sigma

    def call(self, y_true, y_pred):
        s = tf.maximum(y_true, 1e-8)
        nu = tf.maximum(y_pred, 1e-8)
        sigma = self.sigma

        argument = (s * nu) / (sigma**2)

        # Use the stable approximation for log(i0(x))
        log_i0_val = log_i0_stable(argument)

        term = (tf.math.log(sigma**2 + 1e-30)
                - tf.math.log(s + 1e-30)
                + (tf.square(s) + tf.square(nu)) / (2.0 * sigma**2)
                - log_i0_val)

        nll = tf.reduce_sum(term, axis=-1)
        return tf.reduce_mean(nll, axis=0)

# class RicianNLLLoss(Loss):
#     """
#     A custom Keras loss that computes the negative log-likelihood (NLL)
#     under a Rician noise model for magnitude-only data.

#     Attributes:
#         sigma (float): Standard deviation of the Gaussian noise in each
#                        of the real and imaginary components.
#     """
#     def __init__(self, sigma=0.02, name="rician_nll_loss"):
#         super().__init__(name=name)
#         self.sigma = sigma

#     def call(self, y_true, y_pred):
#         """
#         Compute the Rician negative log-likelihood loss.

#         Args:
#             y_true: Observed noisy magnitude data. Shape: (batch, 80)
#             y_pred: Model predicted fitted signals (nu). Same shape as y_true.

#         Returns:
#             A scalar loss (the mean NLL over the batch).
#         """
#         # Ensure observed magnitude is > 0 to avoid log(0)
#         s = tf.maximum(y_true, 1e-12)
#         nu = y_pred
#         sigma = self.sigma

#         # argument = (s * nu) / sigma^2
#         argument = (s * nu) / (sigma**2)

#         # Compute log(I0(argument)):
#         # i0(x) = i0e(x)*exp(|x|)
#         i0e_val = tfp.math.bessel_i0e(argument)
#         log_i0 = tf.math.log(i0e_val + 1e-300) + tf.abs(argument)

#         # NLL terms:
#         # term = log(sigma^2) - log(s) + (s^2 + nu^2)/(2 sigma^2) - log_i0
#         term = (tf.math.log(sigma**2)
#                 - tf.math.log(s)
#                 + (tf.square(s) + tf.square(nu)) / (2.0 * sigma**2)
#                 - log_i0)

#         # Sum over the time dimension and then mean over the batch
#         nll = tf.reduce_sum(term, axis=-1)    # sum over time points (80)
#         return tf.reduce_mean(nll, axis=0)    # mean over the batch
    