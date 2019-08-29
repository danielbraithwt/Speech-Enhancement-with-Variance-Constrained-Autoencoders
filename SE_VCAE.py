import tensorflow as tf
import numpy as np
import math, os
import soundfile as sf

# Set seed for reproducable results
tf.set_random_seed(1234)
np.random.seed(1234)

## USER DEFINED PAREMETERS ##
# Specify the GPU for Tensorflow to use, prevents Tensorflow
# from locking all GPUs on a system.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load Dataset, Only if this python file is being run dirrectly
if __name__ == '__main__':
    train_X = np.load('./_data/train_X_noisy_p200_w600.npz')['arr_0']
    testing_X = np.load('./_data/test_X_noisy_p200_w600.npz')['arr_0']

    train_y = np.load('./_data/train_y_clean_p200_w600.npz')['arr_0']
    testing_y = np.load('./_data/test_y_clean_p200_w600.npz')['arr_0']

# Set the batch size used during training
bs = 200

# The name of the file to save the model paramaters in
model_name = './_models/DN_VCAE_330lf_w600.ckpt'

# Set the variance of the additive noise in each latent dimention
sigma = 0.05

# Set the number of latent features
z_dim = 330

# Set the desired variance of the latent feature distrubution.
v = z_dim

X_size = 1000
X_enh_size = 600

# If you change the "X_enh_size" you will have to adjust the folowing accordingly:
# Divide X_enh_size by 2 the number of times there is a convoluional layer with a
# a stride of 2 in the decoder.
# For example: with the default decoder, and X_enh_size = 600, we have
# d1 = ((600/2)/2)/2 = 75
d1 = 75 # length of maximally reduced slice
## ## ## ## ## ## ## ## ## ##


## Define the Encoder, Decoder and f networks ##
dec_in_channels = 128
channels_wf = 128

reshaped_dim = [-1, d1, 1, dec_in_channels]
n_latent_max = d1 * dec_in_channels


def lrelu(x, alpha=0.1):
    """The Leaky ReLU activation function."""
    return tf.maximum(x, alpha * x)


def encoder(x, z_dim, reuse=False):
    """Creates the computation graph for the encoder network. Takes
        as input:
          - x: A tensor of speech samples with shape [batch_size, X_size]
          - z_dim: The number of latent features
          - reuse (optional): Tells tensorflow that the encoder has allready
                                been constructed and that it should reuse the existing variables"""
    with tf.variable_scope('encoder') as vs:
        if reuse:
            vs.reuse_variables()

        # The data needs to be reshaped into a format that can be processed by the convolutional layers
        x = tf.reshape(x, [-1, X_size, 1])
        conv1 = tf.layers.conv1d(x, kernel_size=31, filters=int(dec_in_channels / 4), strides=1, padding='same', activation=lrelu)
        conv2 = tf.layers.conv1d(conv1, kernel_size=31, filters=int(dec_in_channels / 4), strides=2, padding='same', activation=lrelu)
        conv3 = tf.layers.conv1d(conv2, kernel_size=31, filters=int(dec_in_channels / 2), strides=2, padding='same', activation=lrelu)
        conv4 = tf.layers.conv1d(conv3, kernel_size=31, filters=int(dec_in_channels), strides=2, padding='same', activation=lrelu)
        conv5 = tf.layers.conv1d(conv4, kernel_size=31, filters=int(dec_in_channels), strides=1, padding='same', activation=None)

        flat = tf.layers.flatten(conv5)

        mu = tf.layers.dense(flat, units=z_dim, activation=None)
        return mu


def decoder(z, reuse=False):
    """Creates the computation graph for the decoder network. Takes
        as input:
          - z: A tensor of latent feature vectors
          - reuse (optional): Tells tensorflow that the encoder has allready
                                been constructed and that it should reuse the existing variables."""
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()

        z = tf.layers.dense(z, units=n_latent_max)
        z = tf.reshape(z, reshaped_dim)

        # As of constructing this, tensorflow did not have a conv1d_transpose implementation.
        # Consequently, we implemented the 1D transpose convolutional layer using conv2d_transpose.
        deconv1 = tf.layers.conv2d_transpose(z, kernel_size=(31, 1), filters=int(dec_in_channels / 2), strides=(2, 1), padding='same', activation=lrelu)
        deconv2 = tf.layers.conv2d_transpose(deconv1, kernel_size=(31, 1), filters=int(dec_in_channels / 4), strides=(2, 1), padding='same', activation=lrelu)
        deconv3 = tf.layers.conv2d_transpose(deconv2, kernel_size=(31, 1), filters=int(dec_in_channels / 8), strides=(2, 1), padding='same', activation=lrelu)
        deconv4 = tf.layers.conv2d_transpose(deconv3, kernel_size=(31, 1), filters=int(dec_in_channels / 8), strides=(1, 1), padding='same', activation=lrelu)
        deconv5 = tf.layers.conv2d_transpose(deconv4, kernel_size=(31, 1), filters=1, strides=1, padding='same', activation=None)

        x_hat = tf.layers.flatten(deconv5)
        x_hat = tf.layers.dense(x_hat, units=X_enh_size, activation=None)
        return x_hat


def wf_net_x(v, reuse=False):
    """Create the computation graph for the wasserstein distance function. Takes
        as input:
          - v: A tensor of either clean or enhanced speech, has shape [?, X_enh_size]
          - reuse (optional): Tells tensorflow that the encoder has allready
                                been constructed and that it should reuse the existing variables. """
    with tf.variable_scope('wf') as vs:
        if reuse:
            vs.reuse_variables()
        x = tf.reshape(v, [-1, X_enh_size, 1])
        conv1 = tf.layers.conv1d(x, kernel_size=31, filters=int(channels_wf / 4), strides=2, padding='same', activation=None)
        bn1 = tf.layers.batch_normalization(conv1)
        bn1 = lrelu(bn1)
        conv2 = tf.layers.conv1d(bn1, kernel_size=31, filters=int(channels_wf / 2), strides=2, padding='same', activation=None)
        bn2 = tf.layers.batch_normalization(conv2)
        bn2 = lrelu(bn2)
        conv3 = tf.layers.conv1d(bn2, kernel_size=31, filters=int(channels_wf), strides=2, padding='same', activation=None)
        bn3 = tf.layers.batch_normalization(conv3)
        bn3 = lrelu(bn3)

        flat = tf.layers.flatten(bn3)
        dense = tf.layers.dense(flat, units=1, activation=None)
    return dense
## ## ## ## ## ## ## ## ## ## ##


def de_emph(y, coeff=0.95):
    """Function defining a de-emphesis operation, the inverse of the pre-emphesis filter.
        Takes as input:
          y: The signal to apply the filter to
          coeff: The coefficient of the de-emphesis filter"""

    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

# Hann window used for smoothing the enhanced blocks together
hann_window = np.hanning(X_enh_size)

if __name__ == '__main__':
    # Generate a permutation of the indicies of the training set, this is used
    # to randomise the order of the training items
    permutation = np.random.permutation(train_X.shape[0])

    X = tf.placeholder(tf.float32, shape=[None, X_size])
    y = tf.placeholder(tf.float32, shape=[None, X_enh_size])

    # Encode noisy speech and decode to clean speech
    Z_mu = encoder(X, z_dim)
    Z = Z_mu + tf.random_normal(tf.shape(Z_mu), stddev=sigma, dtype='float32')
    X_hat = decoder(Z)

    # Apply Wasserstein function to clean and enhanced data
    wf_x = wf_net_x(y)
    wf_x_hat = wf_net_x(X_hat, reuse=True)

    # Split tranable varables by whether they are
    wf_var = [v for v in tf.trainable_variables() if 'wf' in v.name]
    net_var = [v for v in tf.trainable_variables() if not 'wf' in v.name]

    ## Construct the loss function
    # Compute the L1 reconstruction error
    L1_err = tf.reduce_mean(tf.reduce_sum(tf.abs(y - X_hat), axis=[1]))

    # Compute variance regulriser which constrains the
    # variance of the latent distrubution.
    moments = tf.nn.moments(Z, axes=[0])
    var = moments[1]
    var_err = tf.reduce_sum(var) - v
    var_reg = tf.abs(tf.reduce_sum(var) - v)

    # Compute Wasserstein distance for X
    wass_dist_x = (tf.reduce_mean(wf_x) - tf.reduce_mean(wf_x_hat))

    # Compute L1 Regulriser loss
    l1_reg = tf.contrib.layers.l1_regularizer(scale=0.000001)
    reg_penalty = tf.contrib.layers.apply_regularization(l1_reg, net_var)

    # Write out loss function.
    # Note: the scaling term (1.0/100.0) is needed so the variance regulriser
    #       term does not make everything else insignificant.
    loss = L1_err + (1.0 / 100.0) * var_reg + wass_dist_x + reg_penalty

    # Gradient Penalty for Wasserstein distance function
    gradients = tf.gradients(wf_net_x(X_hat, reuse=True), [X_hat])
    _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    # Construct training operations
    train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, var_list=net_var)
    train_op_wass_x = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-wass_dist_x + _gradient_penalty, var_list=wf_var)

    # Construct object used to save and load the model from disk.
    saver = tf.train.Saver()

    # Create and initilise tensorflow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Enable the line below to load a pre-trained model.
    saver.restore(sess, model_name)


    def get_batch(pos, bs):
        """Function that returns the next batch. Takes as input:
            pos: The current position in the training data"""

        batch_x = train_X[permutation[pos:pos + bs]]
        batch_y = train_y[permutation[pos:pos + bs]]

        # Update position and return to 0 if we have reached the end of the data
        pos = pos + bs
        if pos >= len(train_X):
            pos = 0

        return batch_x, batch_y, pos


    # Main training loop
    pos = 0
    for i in range(10000000 + 1):
        # Get the next batch
        batch_x, batch_y, p = get_batch(pos, bs)
        pos = p
        feed_dict = {X: batch_x, y: batch_y}

        # Train the neural networks using the constructed batch
        _, _, L1_err_l, var_reg_l, was_dist_x_l = sess.run([train_op_wass_x, train_op, L1_err, var_reg, wass_dist_x], feed_dict=feed_dict)

        # Every 100 batches print out the training infomation
        if i % 100 == 0:
            print(i)
            print("Negative log likelihood is %f, representation loss is %f, Wasserstein distance is %f" % (L1_err_l, var_reg_l, was_dist_x_l))
            print()

        # Every 10,000 batches enhance a portion of the testing set and save it.
        if i % 10000 == 0:
            # Save the model to disk
            save_path = saver.save(sess, model_name)
            print("Saved: ", save_path)

            # Enhance a portion of the testing data and save it to file.
            recon_test = sess.run(X_hat, feed_dict={X: testing_X[0:3000]})

            # Apply hann window to enhanced blocks
            batch_hann = recon_test * hann_window

            # Create empty array to store enhanced signal
            enhanced = np.zeros(300 * len(recon_test) + 300)

            # Join the enhanced blocks together
            idx = 0
            cs = X_enh_size
            for j in range(len(batch_hann)):
                enhanced[idx:idx + cs] = enhanced[idx:idx + cs] + batch_hann[j]
                idx += 300

            enhanced = de_emph(enhanced)
            sf.write('testing_audio_de_noised.wav', enhanced, 16000)

            # Save the coresponding noisy testing audio that was enhanced.
            sf.write('testing_audio_noisy.wav', de_emph(np.concatenate(testing_y[0:3000][:, 0:300], axis=0)), 16000)


    save_path = saver.save(sess, model_name)
    print("Saved: ", save_path)

