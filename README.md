# Documentation for Speech Enhancement VCAE

## Setup
A version of the SE-VCAE that is ready to run out-of-the-box is available in this git repository, which includes:

 - The python scripts described in below.
 - The pre-trained SE-VCAE model (available at: https://www.dropbox.com/sh/o252isrv2eh7bdl/AAB0B7NpTY0XMq3x8_6tM7Tja?dl=0).
 - The noisy testing set.
 - The enhanced version of the noisy training set (using the pre-trained model included).

Since the data files are very large, they are available separately on Google Drive: [https://drive.google.com/open?id=1-77yP500p0UaUgvaVFsPkkLaPHm3dMTi].

Lastly, there is an additional ZIP file (available on Google Drive: [https://drive.google.com/open?id=1yeLgpkJ17G4CBn-u3b31fKnbgDNRJ3aE]) folder called _"Original Audio.zip"_ which contains the raw audio files (both clean and noisy), this can be used to generate new training/testing data with a different configuration (see the section about Training Data).

**The Python version we used is 3.7.3 and the TensorFlow version was 1.12.0**

## Python Script Documentation
In this section we discuss the three Python scripts available in the archive, _"SE-VCAE.zip"_.

### Training Data
#### Default Data:
The data used to train/test to train the model is available online (as discussed in the Setup section), and has the following format:

 - The files have been processed by a pre-emphesis filter (with coefficient $0.95$).
 - The noisy input data is in blocks of 1000 samples.
 - The desired clean output data is in blocks of 600 samples (corresponding to the central 600 samples of the noisy input).

Under the current implementation the training data files should be placed directly into a folder called *\_data*.


#### Generating New Data:
If instead a different configuration is desired, a script has been included to construct new training and testing sets from the original audio files (the file "Original Audio.zip" available on Google Drive). The python file *ProcessData.py* pre-process a set of audio files into a format on which SE-VCAE can operate. There are six parameters (at the top of the Python file) that can be adjusted:
 - *audio_class*: Controls whether its noisy or clean speech being processed.
 - *audio_set*: Controls whether its the train or testing data being processed
 - *pre_emph_coeff*: The coefficient for the pre-emphesis filter that is applied to the data.
 - *peek*: The amount of padding on both sides of the enhancement window.
 - *cs*: The size of the enhancement window.
 - *overlap\_p*: The percentage of overlap between enhancement windows.

It is necessary to run the script four times to pre-process the clean training data, noisy training data, clean testing data, and noisy testing data.


#### Training the Model:
The python file *SE_VCAE.py* trains and saves a speech enhancement VCAE model. Using the default training and testing data, this will run out-of-the-box. **Note: by default, the python script SE_VCAE.py will load the pre-trained model and continue to train it. To train a model from scratch, comment out line 217 of SE_VCAE.py before running**. We note that SE_VCAE.py can also be modified to use a different data setup (i.e. newly generated training/testing data generated using *ProcessData.py*). The user defined parameters in *SE_VCAE.py* are:

 -  *CudaVisibleDevices*: Specifies which GPUs TensorFlow should use. For example, setting to "1" would mean that TensorFlow can only see the GPU with id 1. To find the GPU ids for a system, run the command \textit{nvidia-smi}
 -  *Datasets*: Specify the compressed dataset files to be used for training, leave as default if using the ones available in the Google Drive folder.
 -  *bs*: The batch size to be used during training.
 -  *model_name*: The file in which the trained model weights should be saved.
 -  *sigma*: The variance of the additive noise in each latent dimension.
 -  *v*: The desired summed variance of the latent distribution.
 -  *X_size*: The size of the input window
 -  *X_enh_size*: The size of the enhanced output window.

During the training process a portion of the testing set is enhanced and saved to disk every 10,000 batches, **we note that the current set up for doing this assumes that the overlap is 50\%.**

## Using a Trained SE-VCAE for Enhancement
The python file *SE_VCAE_extract_results.py* enhances a folder of audio files using a trained SE-VCAE model. The python script reads in the audio files and splits them into blocks (which overlap by 50%). Then, the trained SE-VCAE enhances each block. The enhanced blocks are joined together using a Hann window to ensure a smooth output signal. The default setup will use the pre-trained model to enhance the noisy testing set.
