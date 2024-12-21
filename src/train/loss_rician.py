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
    # Rician negative log-likelihood loss
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