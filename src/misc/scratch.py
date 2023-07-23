import tensorflow as tf

def construct_kernel_epg(nte, delta_te, angle, t2s, t1s):
    
    angle = tf.broadcast_to(angle, tf.shape(t2s))
    nte = tf.shape(t2s)[0]
    t2s = tf.expand_dims(t2s, axis=2)
    t2s = tf.repeat(t2s, repeats=nte, axis=2)
    t1s = tf.expand_dims(t1s, axis=2)
    t1s = tf.repeat(t1s, repeats=nte, axis=2)
    angle = tf.expand_dims(angle, axis=2)
    angle = tf.repeat(angle, repeats=nte, axis=2)

    def func(args):
        t2, t1, angle = args
        return epg_signal(nte, delta_te, angle, t2, t1)
    
    kernel_matrix = tf.map_fn(
        fn=func,
        elems=(t2s, t1s, angle),
        dtype=tf.float32,
    )
    
    print(f"+++++++++++{kernel_matrix.shape}") 
    return kernel_matrix

def epg_signal(nte, delta_te, angle, t2, t1):
    return tf.zeros((nte, nte, 3))  # Return a tensor of shape (nte, nte, 3)

# Example usage
net = 24
t2s = tf.keras.Input(shape=(None, 3), dtype=tf.float32)
t1s = tf.keras.Input(shape=(None, 3), dtype=tf.float32)
angle = tf.keras.Input(shape=(None, 1), dtype=tf.float32)
result = construct_kernel_epg(net, 0.1, angle, t2s, t1s)
print(result.shape)  # Output: (None, 24, 24, 3)
