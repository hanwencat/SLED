import numpy as np
import tensorflow as tf
from .loss_rician import RicianNLLLoss

def pretrain_sled(config, sled, decays, amps, t2s):
    """Pretrain the SLED model with synthetic data"""
    
    # map the loss name to the loss class
    loss_mapping = {
        "rician_nll_loss": RicianNLLLoss  # your custom loss class defined somewhere
    }
    output_loss = config['loss']  # This is a dictionary with keys 't2s', 'amps', 'fitted_signals'
    final_loss = {}
    for output_name, loss_name in output_loss.items():
        if loss_name in ["mse", "categorical_crossentropy", "mae"]: 
            # It's a built-in Keras loss
            final_loss[output_name] = loss_name
        else:
            # It's a custom rician loss
            loss_class = loss_mapping[loss_name]
            final_loss[output_name] = loss_class(sigma=config['loss_rician_sigma'])

    
    # Compile the model
    sled.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=config['learning_rate']),
        loss=final_loss,
        loss_weights=config['loss_weights'],
        metrics=config['metrics'],
    )

    # Prepare training data
    y_train = {
        't2s': t2s,
        'amps': amps,
        'fitted_signals': decays
    }

    # Train the model
    history = sled.fit(
        decays,
        y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        verbose=config['verbose'],
    )

    return history


