import tensorflow as tf
import numpy as np
import math, os
import soundfile as sf

from SE_VCAE import encoder, decoder, de_emph


import os
from os import listdir
from os.path import isfile, join

# Set seed for reproducable results
tf.set_random_seed(1234)
np.random.seed(1234)

## USER DEFINED PAREMETERS ##
# Specify the GPU for Tensorflow to use, prevents Tensorflow
# from locking all GPUs on a system.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# The path to the pre-trained SE-VCAE
model_name = './_models/DN_VCAE_330lf_w600.ckpt'

# Set the dimentionality of the latent space
z_dim = 330

# Specify the paths to the files to be enhanced (read_path) and the
# location for the enhanced files to be stored (save_path).
read_path = './_data/noisy_testset_wav_16k'
save_path = './enhanced/'

# Set the amount of padding of either side of the enhancement window
peek = 200

# Set the size of the enhancement window
cs = 600
## ## ## ## ## ## ## ## ## ##

# First get the names of all files in the provided directory
files = []
def listdir(d, files):
    if not os.path.isdir(d):
        if d[-3:] == 'wav':
            files.append(d)
    else:
        for item in os.listdir(d):
            listdir((d + '/' + item) if d != '/' else '/' + item, files)

listdir(read_path, files)

# Read all the audio files inside the given folder.
import numpy as np
import soundfile as sf
raw_audio = []
for f in files:
    print(f)
    a, sr = sf.read(f)
    raw_audio.append(a)


# Then process the loaded audio into something the trained SE-VCAE can process.
# This involves applying a pre emphesis filter and splitting the files into chunks.
def pre_emph(x, coeff=0.95):
    x0 = np.reshape(x[0], [1,])
    diff = x[1:] - coeff * x[:-1]
    concat = np.concatenate([x0, diff], axis=0)
    return concat


def get_chunk_with_margin(x, i, cs, peek):
    min_i = max([i-peek, 0])
    max_i = min([len(x), i + cs + peek])
    chunk = x[min_i: max_i]

    # Check if the we need to pad with 0's
    if i - peek < 0:
        diff = np.abs(i - peek)
        chunk = np.concatenate([np.zeros(diff), chunk], axis=0)
    if i + cs + peek >= len(x):
        diff = np.abs(len(x) - i - peek - cs)
        chunk = np.concatenate([chunk, np.zeros(diff)], axis=0)

    return chunk

# Set the amount of overalp between adjacent windows to 50%
overlap_p = 0.5
overlap_c = int(cs * overlap_p)

clip_X = []
for ra in raw_audio:
    # Split sound files into chunks.
    X = []
    ra = pre_emph(ra)

    # Iterate through the audio file's samples constructing blocks.
    for i in range(0, len(ra), cs - overlap_c):
        raw_chunk = ra[i:i+cs]
        # Check if we need to pad raw_chunk on the right, i.e. are we
        # at the end of the file.
        if len(raw_chunk) < cs:
            diff = cs - len(raw_chunk)
            raw_chunk = np.concatenate([raw_chunk, np.zeros(diff)], axis=0)

        padded_chunk = get_chunk_with_margin(ra, i, cs, peek)
        
        X.append(padded_chunk)
    clip_X.append(X)

hann_window = np.hanning(600)
if __name__ == '__main__':
    X_n = tf.placeholder(tf.float32, shape=[None, 1000])
    X = tf.placeholder(tf.float32, shape=[None, 600])

    Z_mu = encoder(X_n, z_dim)
    Z = Z_mu

    X_hat = decoder(Z)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_name)

    de_noised = []
    for i in range(len(clip_X)):
        # Enhance all the windows for a given audio file
        batch_X_n  = clip_X[i]
        batch_e = sess.run(X_hat, feed_dict={X_n:batch_X_n})

        # Apply the hann window to the enhanced chunks
        batch_hann = batch_e * hann_window

        # Combine the enhanced windows into a single audio file
        enhanced = np.zeros(overlap_c * len(batch_e) + overlap_c)

        idx = 0
        for j in range(len(batch_hann)):
            enhanced[idx:idx + cs] = enhanced[idx:idx + cs] + batch_hann[j]
            idx += overlap_c

        # Clip the joined windows and apply de-emphesis operation
        enhanced = enhanced[0:len(raw_audio[i])]
        enhanced = de_emph(enhanced)

        # Save the file
        fn = files[i].split('/')[-1]
        print(fn)
        sf.write(save_path + '/' + fn, enhanced, 16000)