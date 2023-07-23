import tensorflow as tf
import numpy as np

class emd_mse_cosine_entropy(tf.keras.losses.Loss):
    """weighted loss that combines earth movers distance, mse, cosine and entropy losses"""

    def __init__(
            self, 
            emd_weight=0.3, 
            mse_weight=0.3, 
            cosine_weight=0.2, 
            entropy_weight=0.2, 
            name='weighted_loss',
            ):
        super().__init__(name=name)
        self.emd_weight = emd_weight
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight  
        self.entropy_weight = entropy_weight

    def call(self, y_true, y_pred):
        emd_loss = self.emd_loss(y_true, y_pred)
        mse_loss = self.mse_loss(y_true, y_pred)
        cosine_loss = self.cosine_loss(y_true, y_pred)
        entropy_loss = self.entropy_loss(y_true, y_pred)
        return self.emd_weight * emd_loss \
            + self.mse_weight * mse_loss \
            + self.cosine_weight * cosine_loss \
            + self.entropy_weight * entropy_loss

    def emd_loss(self, y_true, y_pred):
        # implementation of Earth Mover's Distance (EMD) loss between y_true and y_pred.
        cdf_true = tf.math.cumsum(y_true, axis=-1)
        cdf_pred = tf.math.cumsum(y_pred, axis=-1)
        emd = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(cdf_true - cdf_pred), axis=-1)))
        return emd

    def mse_loss(self, y_true, y_pred):
        # implementation of mean squared error loss
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    def cosine_loss(self, y_true, y_pred):
        # implementation of cosine similarity loss
        return tf.keras.losses.cosine_similarity(y_true, y_pred)
    
    def entropy_loss(self, y_true, y_pred):
        # implementation of categorical crossentropy loss
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

