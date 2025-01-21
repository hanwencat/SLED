import tensorflow as tf
import numpy as np

def log_i0_stable(x):
    """
    Computes a stable approximation of log(I0(x)) where I0 is the modified Bessel function of the first kind of order zero.

    Parameters:
    - x: tf.Tensor of any shape

    Returns:
    - tf.Tensor of the same shape as x containing log(I0(x))
    """
    # Absolute value of x for symmetry
    absx = tf.abs(x)
    threshold = 10  # Threshold to switch between series expansion and asymptotic expansion

    def i0_series(z, terms=10):
        z2 = tf.square(z / 2.0)
        result = tf.ones_like(z)
        term = tf.ones_like(z)
        factorial = 1.0
        for k in range(1, terms):
            factorial *= k
            term = term * z2 / (factorial ** 2)
            result = result + term
        return result

    def small_x_branch():
        val = i0_series(x, terms=10)
        return tf.math.log(val + 1e-30)  # Add epsilon to prevent log(0)

    def large_x_branch():
        # Asymptotic expansion: log(I0(x)) ≈ x - 0.5*log(2πx)
        return x - 0.5 * tf.math.log(2.0 * np.pi * absx + 1e-30)

    return tf.where(absx < threshold, small_x_branch(), large_x_branch())


def loss_rician_nll(y_true, outputs_dict):
    """
    Computes the Rician Negative Log-Likelihood loss in a numerically stable manner.

    Parameters:
    - y_true: Tensor of shape (batch_size, echoes), dtype float32
              Observed signals (r)
    - outputs_dict: Dictionary with keys:
        - 'multiecho': Tensor of shape (batch_size, echoes), dtype float32
                            Predicted signals (v)
        - 'sigma': Tensor of shape (batch_size, 1), dtype float32
                  Predicted scale parameter (sigma)

    Returns:
    - Scalar tensor representing the mean Rician NLL over the batch
    """
    
    # Extract predictions
    r = y_true                            # Observed signals, shape (batch_size, echoes)
    v = outputs_dict['multiecho']    # Predicted signals, shape (batch_size, echoes)
    sigma = outputs_dict['sigma']         # Predicted scale parameter, shape (batch_size, 1)
    
    # Ensure sigma has shape (batch_size, echoes) for broadcasting
    sigma = tf.tile(sigma, [1, tf.shape(r)[1]])  # Shape: (batch_size, echoes)
    
    # Add a small epsilon to sigma_sq to prevent division by zero
    epsilon = 1e-5
    sigma_sq = tf.square(sigma) + epsilon  # Shape: (batch_size, echoes)
    
    # Compute the argument for the Bessel function and clip it to prevent overflow
    bessel_arg = (r * v) / sigma_sq         # Shape: (batch_size, echoes)
    bessel_arg = tf.clip_by_value(bessel_arg, -10.0, 10.0)  # Prevent overflow in Bessel function
    
    # Compute the modified Bessel function of the first kind of order zero using the stable function
    log_i0 = log_i0_stable(bessel_arg)      # Shape: (batch_size, echoes)
    
    # Compute the Rician NLL
    # NLL = -ln(r / sigma^2) + (r^2 + v^2) / (2 * sigma^2) - ln(I0(rv / sigma^2))
    r_over_sigma_sq = r / sigma_sq + epsilon  # Avoid log(0)
    log_r_over_sigma_sq = tf.math.log(r_over_sigma_sq)  # Shape: (batch_size, echoes)
    
    nll = -log_r_over_sigma_sq + (tf.square(r) + tf.square(v)) / (2.0 * sigma_sq) - log_i0
    
    # Sum the NLL over the echoes features for each sample
    nll_sum = tf.reduce_sum(nll, axis=-1)  # Shape: (batch_size,)
    
    # Compute the mean NLL over the batch
    mean_nll = tf.reduce_mean(nll_sum)     # Scalar
    
    return mean_nll

