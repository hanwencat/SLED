import tensorflow as tf

# def loss_gaussian_nll(y_true, outputs_dict):
#     """
#     y_true: shape (batch, echoes)  -- observed signals
#     outputs_dict: a dictionary with keys like:
#         'multiecho': shape (batch, echoes) => mu
#         'sigma': shape (batch, 1) or shape (batch,) => log_sigma or sigma
#     """

#     # extract the outputs
#     y_predict = outputs_dict['multiecho'] # shape (batch,echoes)
#     sigma = outputs_dict['sigma']  # shape (batch,1) or (batch,)
#     amps = outputs_dict['amps']  # shape (batch,latent_shape)

#     # shape alignment
#     # mu: (batch,echoes), y_true: (batch,echoes)
#     # sigma: (batch,1). We'll broadcast across time dimension
#     sigma_2d = tf.reshape(sigma, (-1, 1))  # (batch,1)
    
#     diff = y_true - y_predict  # shape (batch,echoes)

#     # negative log-likelihood
#     # NLL = ln(sigma^2) + (diff^2)/(2*sigma^2)
#     nll = tf.math.log(sigma_2d**2) + tf.square(diff) / (2.0 * sigma_2d**2)

#     # sum across the echoes time points, then average over batch
#     nll = tf.reduce_sum(nll, axis=-1)  # shape (batch,)
#     return tf.reduce_mean(nll)         # scalar


def loss_gaussian_nll(
    y_true, 
    outputs_dict,
    smoothness_lambda=0
):
    """
    y_true: shape (batch, echoes)  -- observed signals
    outputs_dict: a dictionary with keys:
        'multiecho': shape (batch, echoes) => mu (predicted signals)
        'sigma': shape (batch, 1) or shape (batch,) => predicted noise std
        'amps': shape (batch, 40) => amps latent shape
    smoothness_lambda: float
        Weight for the smoothness penalty on consecutive T2 element's amps.
    """
    # 1) Extract outputs
    y_pred = outputs_dict['multiecho']   # shape (batch, echoes)
    sigma = outputs_dict['sigma']        # shape (batch,) or (batch,1)
    amps = outputs_dict['amps']           # shape (batch, 40)

    # 2) Standard Gaussian NLL
    sigma_2d = tf.reshape(sigma, (-1, 1))  # (batch, 1)
    diff = y_true - y_pred                # shape (batch, echoes)
    nll = tf.math.log(sigma_2d**2) + tf.square(diff) / (2.0 * sigma_2d**2)
    # nll = tf.math.log(sigma_2d**2) + tf.square(diff) / (2.0 * sigma_2d**2)
    nll = tf.reduce_sum(nll, axis=-1)     # sum across echoes => shape (batch,)
    nll_loss = tf.reduce_mean(nll)        # average across batch => scalar

    # 3) Smoothness Penalty for amps
    #    We penalize consecutive differences: sum_i (amps[:, i+1] - amps[:, i])^2
    #    amps has shape (batch, 40). We'll compute differences along axis=1.
    diffs = amps[:, 1:] - amps[:, :-1]   # shape (batch, 39)
    smooth_term_per_sample = tf.reduce_sum(tf.square(diffs), axis=1)  # shape (batch,)
    smooth_term = tf.reduce_mean(smooth_term_per_sample)  # scalar

    # 4) Combine
    total_loss = nll_loss + smoothness_lambda * smooth_term
    return total_loss