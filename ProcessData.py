import os
from os import listdir
from os.path import isfile, join
import math

## USER DEFINED PAREMETERS ##
# Set the class of audio which is to be processed ('clean'/'noisy')
audio_class = 'clean'

# Set the audio set ot be enhanced ('train'/'test')
audio_set = 'test'

# The coefficient for the preemphesis filter
pre_emph_coeff = 0.95

# Set the amount of padding of either side of the enhancement window
peek = 200

# Set the size of the enhancement window
cs = 600

# Set the percentge of overlap between the blocks (0 <= overlap_p <= 1)
overlap_p = 0.5
## ## ## ## ## ## ## ## ## ##

# Construct path to the desired audio folder
path = './_data/' + audio_class + '_' + audio_set + 'set_wav_16k'

files = []
def listdir(d, files):
    if not os.path.isdir(d):
        if d[-3:] == 'wav':
            files.append(d)
    else:
        for item in os.listdir(d):
            listdir((d + '/' + item) if d != '/' else '/' + item, files)

listdir(path, files)

import numpy as np
import soundfile as sf
raw_audio = []
for f in files:
    print(f)
    a, sr = sf.read(f)
    raw_audio.append(a)

def get_chunk_with_margin(a, i, cs, peek):
    min_i = max([i-peek, 0])
    max_i = min([len(a), i + cs + peek])
    chunk = a[min_i : max_i]

    # Check if the we need to pad with 0's
    if i - peek < 0:
        diff = np.abs(i - peek)
        chunk = np.concatenate([np.zeros(diff), chunk], axis=0)
    if i + cs + peek >= len(a):
        diff = np.abs(len(a) - i - peek - cs)
        chunk = np.concatenate([chunk, np.zeros(diff)], axis=0)

    return chunk

def pre_emph(x, coeff=0.95):
    x0 = np.reshape(x[0], [1,])
    diff = x[1:] - coeff * x[:-1]
    concat = np.concatenate([x0, diff], axis=0)
    return concat


# Split sound files into chunks.
X = []
y = []

# Compute the number of samples overlapping between windows.
overlap_c = int(cs * overlap_p)

# For each raw audio file that has been read in:
# - Apply a pre-emphesis filter.
# - Split into chunks of desired size, one with an additional amount of
#   padding on either end.
for ra in raw_audio:
    ra = pre_emph(ra, pre_emph_coeff)

    for i in range(0, len(ra), cs - overlap_c):
        raw_chunk = ra[i:i+cs]
        # Check if we need to pad raw_chunk on the right
        if len(raw_chunk) < cs:
            diff = cs - len(raw_chunk)
            raw_chunk = np.concatenate([raw_chunk, np.zeros(diff)], axis=0)

        padded_chunk = get_chunk_with_margin(ra, i, cs, peek)

        assert len(raw_chunk) == cs
        assert len(padded_chunk) == cs + 2*peek
        
        X.append(padded_chunk)
        y.append(raw_chunk)

# Once all files have been processed, save result into compressed file.
X = np.array(X)
y = np.array(y)
np.savez_compressed('./_data/' + audio_set + '_X_' + audio_class + '.npz', X)
np.savez_compressed('./_data/' + audio_set + '_y_' + audio_class + '.npz', y)