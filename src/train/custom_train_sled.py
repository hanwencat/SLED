import tensorflow as tf
from .loss_gaussian_nll import loss_gaussian_nll
from .loss_rician_nll import loss_rician_nll

def custom_train_sled(model, x, y, config):
    """
    Custom training loop with support for multiple labels, ReduceLROnPlateau scheduler, and metric monitoring.

    Parameters:
    - model: Keras Model to train
    - x: NumPy array of shape (N, echoes), input data
    - y: Dictionary containing labels:
        - 'multiecho': NumPy array of shape (N, echoes)
        - 't2s': NumPy array of shape (N,)
        - 'amps': NumPy array of shape (N,)
        - 'sigma': NumPy array of shape (N,)
    - config: Configuration dictionary containing training parameters and callback settings
    """
    
    # Ensure all data is float32
    x = x.astype('float32')
    y = {key: val.astype('float32') for key, val in y.items()}
    
    # Create a tf.data.Dataset from the data
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=len(x)) \
                     .batch(config['batch_size']) \
                     .prefetch(tf.data.AUTOTUNE)
    
    # Initialize optimizer without built-in LR scheduling
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['optimizer']['learning_rate'],
        clipnorm=config['optimizer'].get('clipnorm', None),  
        clipvalue=config['optimizer'].get('clipvalue', None),
    )
    
    # Define loss functions for each label
    loss_functions = {
        'multiecho': loss_gaussian_nll if config['noise_model'] == 'gaussian' else loss_rician_nll,
        # 'multiecho': tf.keras.losses.CategoricalCrossentropy(),
        # 'multiecho': tf.keras.losses.MeanSquaredError(),
        't2s': tf.keras.losses.MeanAbsoluteError(),
        'amps': tf.keras.losses.MeanAbsoluteError(),
        'sigma': tf.keras.losses.MeanAbsoluteError(),
    }
    
    # Determine which labels to train on based on y's keys
    train_labels = list(y.keys())
    
    # Optionally, assign weights to different losses
    # Fetch loss_weights from config; if not present, default to 1.0 for all labels
    config_loss_weights = config.get('loss_weights', {})
    loss_weights = {label: config_loss_weights.get(label, 1.0) for label in train_labels}
    
    # Initialize metrics for each label you want to monitor
    metrics = {}
    for label in train_labels:
        metrics[f'{label}_mae'] = tf.keras.metrics.MeanAbsoluteError(name=f'{label}_mae')
    
    # Initialize ReduceLROnPlateau parameters
    reduce_lr_config = config.get('ReduceLROnPlateau', {})
    lr_monitor = reduce_lr_config.get('monitor', 'epoch_loss')  # Monitor training loss
    lr_factor = reduce_lr_config.get('factor', 0.1)
    lr_patience = reduce_lr_config.get('patience', 5)
    lr_min = reduce_lr_config.get('min_lr', 1e-6)
    lr_verbose = reduce_lr_config.get('verbose', 1)
    lr_patience_counter = 0
    best_lr_monitor = float('inf')  # Assuming lower is better (e.g., loss)
    
    # Initialize ModelCheckpoint parameters
    checkpoint = config.get('ModelCheckpoint', {})
    save_folder_path = checkpoint.get('save_folder_path', 'model/')
    prefix = checkpoint.get('prefix', 'best')
    checkpoint_filepath = save_folder_path + '/' + prefix + '_model.h5'
    checkpoint_monitor = checkpoint.get('monitor', 'epoch_loss')
    checkpoint_save_best_only = checkpoint.get('save_best_only', True)
    checkpoint_verbose = checkpoint.get('verbose', 1)
    best_checkpoint_loss = float('inf')
    
    # Training loop
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        # Reset training metrics at the start of each epoch
        for metric in metrics.values():
            metric.reset_states()
        
        print(f"Starting Epoch {epoch+1}/{config['epochs']}")
        
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                outputs_dict = model(x_batch, training=True)
                total_loss = 0.0
                
                # Compute loss for each selected label
                for label in train_labels:
                    if label not in y_batch:
                        raise ValueError(f"Label '{label}' not found in y_batch.")
                    
                    if label == 'multiecho':
                        # Use appropriate loss function based on noise model
                        loss_fn = loss_functions[label]
                        loss = loss_fn(y_batch['multiecho'], outputs_dict)
                        # loss = loss_fn(y_batch['multiecho'], outputs_dict['multiecho'])
                    else:
                        # For other labels, use standard loss functions
                        loss_fn = loss_functions[label]
                        # Ensure that outputs_dict contains the corresponding output
                        if label not in outputs_dict:
                            raise ValueError(f"Model output does not contain '{label}' for loss computation.")
                        loss = loss_fn(y_batch[label], outputs_dict[label])
                    
                    # Apply loss weight if specified
                    if label in loss_weights:
                        loss = loss_weights[label] * loss
                    
                    # Accumulate to total loss
                    total_loss += loss
                
                # Optionally, normalize the total loss by the number of labels
                # total_loss = total_loss / len(train_labels)
            
            # Compute gradients
            grads = tape.gradient(total_loss, model.trainable_weights)
            
            # Apply gradients
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            # Update metrics with current batch
            for label in train_labels:
                if label not in y_batch or label not in outputs_dict:
                    continue  # Skip if label is not present in current batch/output
                y_true = y_batch[label]
                y_pred = outputs_dict[label]
                metrics[f'{label}_mae'].update_state(y_true, y_pred)
            
            # Accumulate loss
            epoch_loss += total_loss.numpy()
            num_batches += 1
        
        # Compute average loss for the epoch
        avg_loss = epoch_loss / num_batches
        
        # Compute average training metrics for the epoch
        avg_metrics = {metric.name: metric.result().numpy() for metric in metrics.values()}
        
        # Print training metrics
        metrics_str = ', '.join([f"{name}: {value:.4f}" for name, value in avg_metrics.items()])
        print(f"Epoch {epoch+1}/{config['epochs']}, Average Loss: {avg_loss:.4f}, {metrics_str}, LR: {optimizer.learning_rate.numpy():.6f}")
        
        # Update ReduceLROnPlateau logic based on monitored metric (training loss)
        current_monitor = avg_loss  # Since we're monitoring training loss
        if current_monitor < best_lr_monitor - 1e-4:  # A small delta to account for floating point precision
            best_lr_monitor = current_monitor
            lr_patience_counter = 0
            # if lr_verbose:
            #     print(f"Epoch {epoch+1}: {lr_monitor} improved to {current_monitor:.4f}. Resetting LR patience counter.")
        else:
            lr_patience_counter += 1
            if lr_patience_counter >= lr_patience:
                old_lr = optimizer.learning_rate.numpy()
                new_lr = max(old_lr * lr_factor, lr_min)
                optimizer.learning_rate.assign(new_lr)
                # if lr_verbose:
                #     print(f"Epoch {epoch+1}: {lr_monitor} did not improve for {lr_patience} epochs. Reducing learning rate to {new_lr:.6f}.")
                lr_patience_counter = 0  # Reset counter after reducing LR
        
        # ModelCheckpoint logic based on training loss
        if checkpoint_save_best_only:
            if current_monitor < best_checkpoint_loss - 1e-4:  # A small delta
                best_checkpoint_loss = current_monitor
                model.save_weights(checkpoint_filepath)
                if checkpoint_verbose:
                    print(f"Model checkpoint saved at epoch {epoch+1} with {checkpoint_monitor}: {current_monitor:.4f}")
        else:
            model.save_weights(checkpoint_filepath)
            if checkpoint_verbose:
                print(f"Model checkpoint saved at epoch {epoch+1}.")

    # Load best model weights if save_best_only is True
    if checkpoint_save_best_only:
        model.load_weights(checkpoint_filepath)
    
                

# def custom_train_sled(model, x, y, epochs=10, lr=1e-3):
#     # Ensure all data is float32
#     x = x.astype('float32')
#     y = {key: val.astype('float32') for key, val in y.items()}
    
#     # Create a tf.data.Dataset from the data
#     dataset = tf.data.Dataset.from_tensor_slices((x, y['multiecho']))
#     dataset = dataset.shuffle(buffer_size=len(x)) \
#                      .batch(512) \
#                      .prefetch(tf.data.AUTOTUNE)
    
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         num_batches = 0
#         for step, (x_batch, y_batch) in enumerate(dataset):
#             # Debug: Print batch shapes and dtypes
#             # print(f"Epoch {epoch+1}, Step {step+1}")
#             # print(f"x_batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")
#             # print(f"y_batch shape: {y_batch.shape}, dtype: {y_batch.dtype}")

#             with tf.GradientTape() as tape:
#                 # Forward pass
#                 outputs_dict = model(x_batch, training=True)
#                 # print(f"outputs_dict keys: {list(outputs_dict.keys())}")
#                 # print(f"multiecho shape: {outputs_dict['multiecho'].shape}, dtype: {outputs_dict['multiecho'].dtype}")
#                 # print(f"sigma shape: {outputs_dict['sigma'].shape}, dtype: {outputs_dict['sigma'].dtype}")

#                 # Compute loss
#                 loss_value = loss_gaussian_nll(y_batch, outputs_dict)
#                 # print(f"Loss value: {loss_value.numpy()}")

#             # Compute gradients
#             grads = tape.gradient(loss_value, model.trainable_weights)

#             # Apply gradients
#             optimizer.apply_gradients(zip(grads, model.trainable_weights))

#             # Accumulate loss
#             epoch_loss += loss_value.numpy()
#             num_batches += 1

#         # Compute average loss for the epoch
#         avg_loss = epoch_loss / num_batches
#         print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}\n")