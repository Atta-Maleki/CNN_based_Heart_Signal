from scipy import signal
import pywt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.signal import resample
from scipy.signal import convolve as sig_convolve
from skimage.measure import block_reduce
from PIL import Image
#from ecgdetectors import Detectors
from math import floor, ceil
import random
import csv


# Define butterworth filter and widths for cwt
sos = signal.butter(11, (0.014, 0.3), 'bandpass', output='sos')
# 5, (0.02, 0.3)

# Initialise peak detection algorithms
#detectors = Detectors(300)

# cwt version 1 widths
widths = np.arange(1, 151)

# cwt version 2 widths
fs = 300
w = 24
freq = np.exp2(np.linspace(0, 6, 299)) + 3.7
widths2 = w*fs / (2*freq*np.pi)


# Length randomisation for data augmentation
def randomise_length(sig):
    signal_length = len(sig)
    # Choose target length
    if signal_length >= 9000:
        target_length = random.randint(5000, 9000)
    else:
        target_length = random.randint(round(signal_length * 0.6), signal_length)
    # Randomise signal length
    if target_length < signal_length:
        interval_start = random.randint(0, signal_length - target_length)
        return sig[interval_start:interval_start + target_length]
    else:
        return sig


# Time scale randomisation for data augmentation
def randomise_time(sig, magnitude=0.0):
    if magnitude == 0.0:
        return sig
    signal_length = len(sig)
    target_length = random.randint(int(signal_length * (1.0 - magnitude)), int(signal_length * (1.0 + magnitude)))
    return np.floor(resample(sig, target_length))


# Randomise polarity of the signal for data augmentation
def randomise_polarity(sig):
    flip = random.randint(0, 1)
    if flip == 0:
        return sig
    else:
        return sig * -1


# Length randomisation, reza version
def randomise_length1(sig):
    signal_length = len(sig)
    # Cut random length from start
    cut_start = random.randint(0, round(signal_length * 0.25))
    cut_end = random.randint(0, round(signal_length * 0.25))
    return sig[cut_start:signal_length - cut_end]


def flip_horizontally(sig):
    flip = random.randint(0, 1)
    if flip == 0:
        return sig
    else:
        return np.flip(sig)


# Data augmentation
def data_augment(ID_list, class_list, new_data_dir='training2017_augmented/', nsr_copies=0, length_rand_mode=0, time_rand=0.0, polarity_flip=True, time_flip=True):
    # Load the reference file as a dataframe
    reference = pd.read_csv('training2017/REFERENCE.csv')
    # Get the count of each class
    count_af = 0
    count_nsr = 0
    count_other = 0
    count_noisy = 0
    for i in class_list:
        if i == 'A':
            count_af += 1
        elif i == 'N':
            count_nsr += 1
        elif i == 'O':
            count_other += 1
        else:
            count_noisy += 1

    # How many copies of other signals to make:
    af_copies = floor((nsr_copies + 1) * count_nsr / count_af) - 1
    other_copies = floor((nsr_copies + 1) * count_nsr / count_other) - 1
    noisy_copies = floor((nsr_copies + 1) * count_nsr / count_noisy) - 1

    # Define length randomisation modes
    def length_randomizer(sig, mode):
        if mode == 0:
            return randomise_length(sig)    # Length randomisation method 1
        elif mode == 1:
            return randomise_length1(sig)   # Method 2 (Reza version)
        else:
            return sig                      # No length randomisation

    # Create files and update dataframe
    for i, ID in enumerate(ID_list):
        # Load file
        sig = (loadmat('training2017/' + ID + '.mat'))["val"][0]
        # Save a copy in augmented data directory
        savemat(new_data_dir + ID + '.mat', {'val': np.expand_dims(sig, 0)})

        # Make copies of Normal signals
        if class_list[i] == 'N' and nsr_copies != 0:
            for j in range(1, nsr_copies + 1):
                # Flip polarity randomly
                if polarity_flip:
                    sig = randomise_polarity(sig)
                # Flip along timescale
                if time_flip:
                    sig = flip_horizontally(sig)
                # Cut signal
                new_sig = length_randomizer(sig, length_rand_mode)
                # Stretch signal
                new_sig = randomise_time(new_sig, time_rand)
                # Save signal
                savemat(new_data_dir + 'A' + str(j) + ID[2:6] + '.mat', {'val': np.expand_dims(new_sig, 0)})
                new_entry = {
                    "Entry": 'A' + str(j) + ID[2:6],
                    "Class": 'N'
                }
                reference = reference.append(new_entry, ignore_index=True)

        # Make copies of AF signals
        elif class_list[i] == 'A' and af_copies != 0:
            for j in range(1, af_copies + 1):
                if polarity_flip:
                    sig = randomise_polarity(sig)
                if time_flip:
                    sig = flip_horizontally(sig)
                new_sig = length_randomizer(sig, length_rand_mode)
                new_sig = randomise_time(new_sig, time_rand)
                savemat(new_data_dir + 'A' + str(j) + ID[2:6] + '.mat', {'val': np.expand_dims(new_sig, 0)})
                new_entry = {
                    "Entry": 'A' + str(j) + ID[2:6],
                    "Class": 'A'
                }
                reference = reference.append(new_entry, ignore_index=True)

        # Make copies of Other signals
        elif class_list[i] == 'O' and other_copies != 0:
            for j in range(1, other_copies + 1):
                if polarity_flip:
                    sig = randomise_polarity(sig)
                if time_flip:
                    sig = flip_horizontally(sig)
                new_sig = length_randomizer(sig, length_rand_mode)
                new_sig = randomise_time(new_sig, time_rand)
                savemat(new_data_dir + 'A' + str(j) + ID[2:6] + '.mat', {'val': np.expand_dims(new_sig, 0)})
                new_entry = {
                    "Entry": 'A' + str(j) + ID[2:6],
                    "Class": 'O'
                }
                reference = reference.append(new_entry, ignore_index=True)

        # Make copies of TN signals
        elif class_list[i] == '~' and noisy_copies != 0:
            for j in range(1, noisy_copies + 1):
                if polarity_flip:
                    sig = randomise_polarity(sig)
                if time_flip:
                    sig = flip_horizontally(sig)
                new_sig = length_randomizer(sig, length_rand_mode)
                new_sig = randomise_time(new_sig, time_rand)
                savemat(new_data_dir + 'A' + str(j) + ID[2:6] + '.mat', {'val': np.expand_dims(new_sig, 0)})
                new_entry = {
                    "Entry": 'A' + str(j) + ID[2:6],
                    "Class": '~'
                }
                reference = reference.append(new_entry, ignore_index=True)
        else:
            pass

    # Write the dataframe as a new csv file
    reference.to_csv(new_data_dir + 'REFERENCE_AUGMENTED.csv', index=False)


# Normalise signal length
def normalise_length(sig, NL):
    signal_length = len(sig)
    if signal_length > NL:
        interval_start = (signal_length - NL) / 2
        return sig[floor(interval_start):signal_length - ceil(interval_start)]
    elif signal_length < NL:
        pad_length = (NL - signal_length) / 2
        return np.pad(sig, (floor(pad_length), ceil(pad_length)), 'wrap')
    else:
        return sig


# Normalise signal length
def normalise_length_zero_pad(sig, NL):
    signal_length = len(sig)
    if signal_length > NL:
        interval_start = (signal_length - NL) / 2
        return sig[floor(interval_start):signal_length - ceil(interval_start)]
    elif signal_length < NL:
        pad_length = (NL - signal_length) / 2
        return np.pad(sig, (floor(pad_length), ceil(pad_length)), 'constant')
    else:
        return sig


# Apply bandpass filter and voltage normalisation between (0, 1)
def apply_filter_0_1(sig_array):
    filtered = signal.sosfiltfilt(sos, sig_array)
    # 0-1 normalisation
    filtered = filtered - np.amin(filtered)
    return filtered / np.amax(filtered)


# Bandpass and voltage normalisation between (-1, 1)
def apply_filter(sig_array):
    filtered = signal.sosfiltfilt(sos, sig_array)
    filtered = (filtered - np.amin(filtered)) / (np.amax(filtered)- np.amin(filtered))
    # if np.amin(filtered) < -1.0:
    #     filtered = filtered / abs(np.amin(filtered))
    return filtered


# Bandpass filter designed by Reza
def new_filt(new_sample, cutoff, order, fs, norm_mode=1):
    # Median filter
    temp1 = signal.medfilt(new_sample, kernel_size=np.int64(fs/5+1))
    temp2 = signal.medfilt(temp1, kernel_size=np.int64(2*fs/3+1))
    new_sample_subt = new_sample-temp2
    # FIR filter
    b = signal.firwin(order,  2*cutoff/fs , window='hamming')
    filtered = (sig_convolve(np.expand_dims(new_sample_subt,axis=0), b[np.newaxis, :], mode='valid'))[0]

    if norm_mode == 1:
        # Normalise voltage between (-1, 1)
        filtered = filtered / np.amax(filtered)
        if np.amin(filtered) < -1.0:
            filtered = filtered / abs(np.amin(filtered))
    else:
        # 0-1 normalisation
        filtered = filtered - np.amin(filtered)
        filtered = filtered / np.amax(filtered)

    return filtered


# Resampling the signal:
def resample_timescale(sig, initial_freq, final_freq):
    target_length = int(len(sig) * final_freq / initial_freq)
    sig_resampled = resample(sig, target_length)
    return sig_resampled


# Create filtered signal arrays
def create_filtered_signals(ID_list, data_dir, image_dir, filt_type=0, length_norm=9000):
    for i, ID in enumerate(ID_list):
        # Load file
        sig = (loadmat(data_dir + ID + '.mat'))["val"][0]

        def filter_mode(sig, mode):
            if mode == 0:
                return apply_filter(sig)                            # Butterworth filt -1 to 1
            elif mode == 1:
                return apply_filter_0_1(sig)                        # Butterworth filt 0 to 1
            elif mode == 2:
                return new_filt(sig, 45, 20, 300, norm_mode=1)      # FIR filt -1 to 1
            elif mode == 3:
                return new_filt(sig, 45, 20, 300, norm_mode=0)      # FIR filt 0 to 1
            else:
                return sig

        # Apply filter
        filtered = filter_mode(sig, filt_type)
        # Normalise signal length
        filtered = normalise_length_zero_pad(filtered, length_norm)

        # Expand dims to fit into CNN
        filtered = np.expand_dims(filtered, 1)
        # Store image in a file
        np.save(image_dir + ID + ".npy", filtered)


# Filtering process for Cambell's data
def create_filtered_signals_cambell(ID_list, data_dir='Cambell_data/', image_dir='Cambell_data_filtered/', filt_type=0, length_norm=9000):
    for i, ID in enumerate(ID_list):
        # Load file
        sig = np.loadtxt(data_dir + ID + '/' + ID + '.txt')

        def filter_mode(sig, mode):
            if mode == 0:
                return apply_filter(sig)                            # Butterworth filt -1 to 1
            elif mode == 1:
                return apply_filter_0_1(sig)                        # Butterworth filt 0 to 1
            elif mode == 2:
                return new_filt(sig, 45, 20, 250, norm_mode=1)      # FIR filt -1 to 1
            elif mode == 3:
                return new_filt(sig, 45, 20, 250, norm_mode=0)      # FIR filt 0 to 1
            else:
                return sig

        # Apply filter
        filtered = filter_mode(sig, filt_type)
        # Resample
        filtered = resample_timescale(filtered, 250, 300)
        # Normalise signal length
        filtered = normalise_length_zero_pad(filtered, length_norm)

        # Expand dims to fit into CNN
        filtered = np.expand_dims(filtered, 1)
        # Store image in a file
        np.save(image_dir + ID + ".npy", filtered)


# Downsample signal resolution
def downsample_signal(ID_list, image_dir, save_dir, target_length):
    for i, ID in enumerate(ID_list):
        # Load file
        sig = np.load(image_dir + ID + '.npy')
        # Normalise signal length
        sig = resample(sig, target_length)
        # Store image in a file
        np.save(save_dir + ID + ".npy", sig)


# Apply cwt to a single signal
def apply_cwt(filtered_sig):
    cwt, freqs = pywt.cwt(filtered_sig, widths, "fbsp2-1.9-1.0")
    cwt_small = block_reduce(np.abs(cwt), block_size=(1, 20), func=np.mean)
    # return np.transpose(cwt_small)
    return cwt_small


# Apply cwt, matlab scalogram version
def apply_cwt_mat(filtered_sig):
    cwt, freqs = pywt.cwt(filtered_sig, widths2, "cmor1.5-2.0")
    cwt_small = block_reduce(np.abs(cwt), block_size=(1, 15), func=np.mean)
    return cwt_small[:, 0:598]


# Create cwt images
def create_cwt_images(ID_list, data_dir, image_dir):
    for i, ID in enumerate(ID_list):
        # Load file
        sig = (loadmat(data_dir + ID + '.mat'))["val"]
        # Normalise signal length
        sig = normalise_length(sig, 9000)
        # Apply filter
        filtered = apply_filter(sig)
        # Apply cwt
        cwt = apply_cwt(filtered)
        # Convert to rgb image
        cwt = cwt * (255 / np.amax(cwt))
        image_file = Image.fromarray(cwt).convert("RGB")
        # Store image in a file
        np.save(image_dir + ID + ".npy", image_file)


# Create cwt images, matlab version
def create_cwt_images_mat(ID_list, data_dir, image_dir):
    for i, ID in enumerate(ID_list):
        # Load file
        sig = (loadmat(data_dir + ID + '.mat'))["val"]
        # Normalise signal length
        sig = normalise_length(sig, 9000)
        # Apply filter
        filtered = apply_filter(sig)
        # Apply cwt
        cwt = apply_cwt_mat(filtered)
        # Convert to rgb image
        cwt = cwt * (255 / np.amax(cwt))
        image_file = Image.fromarray(cwt).convert("RGB")
        # Store image in a file
        np.save(image_dir + ID + ".npy", image_file)


# Create 3 channel rgb image from 1 channel np array, extra function for previous non rgb cwt images
def create_rgb_images(ID_list, array_dir, image_dir):
    for i, ID in enumerate(ID_list):
        # Load file
        image_array = np.load(array_dir + ID + '.npy')
        image_array = image_array[:, :, 0]
        image_array = image_array * (255 / np.amax(image_array))
        # Convert to rgb image
        image_file = Image.fromarray(image_array).convert("RGB")
        # Store image in a file
        np.save(image_dir + ID + ".npy", image_file)


# Create downsample rgb array and reduce resolution
def downsample_images(ID_list, array_dir, image_dir):
    for i, ID in enumerate(ID_list):
        # Load file
        image_array = np.load(array_dir + ID + '.npy')
        small_image = block_reduce(image_array, block_size=(1, 2, 1), func=np.mean)
        small_image = small_image[:, :, 0]
        small_image = np.transpose(small_image)
        small_image = small_image * (255 / np.amax(small_image))
        # Convert to rgb image
        image_file = Image.fromarray(small_image).convert("RGB")
        # Store image in a file
        np.save(image_dir + ID + ".npy", image_file)

