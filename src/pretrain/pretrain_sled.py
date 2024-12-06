import tensorflow as tf

def pretrain_sled(config, sled, decays, amps, t2s):
    """Pretrain the SLED model with synthetic data"""
    
    # Compile the model
    sled.compile(
        optimizer=tf.keras.optimizers.legacy.Adamax(learning_rate=config['learning_rate']),
        loss=config['loss'],
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
        verbose=config['verbose'],
    )

    return history
