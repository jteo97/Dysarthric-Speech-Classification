# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:09:05 2021

@author: Justin
"""

import numpy as np

import librosa, librosa.display
import glob, os
import time
import tables
import wave
import contextlib
import matplotlib.pyplot as plt
import numpy as np

base_path_to_audio = 'UASPEECH/audio/Dysarthric/'
path_to_B1_samples = 'dataset&labels/B1/'
path_to_B2_samples = 'dataset&labels/B2/'
path_to_B3_samples = 'dataset&labels/B3/'
mapping_to_inteligibility = {
    0: "Very low",
    1: "Low",
    2: "Mid",
    3: "High"
}

samples = {
  "F02": 1,
  "F03": 0,
  "F04": 2,
  "F05": 3,
  "M01": 0,
  "M04": 0,
  "M05": 2,
  "M06": 1,
  "M07": 1,
  "M09": 3
}

mapping_to_set = {
    "B1": 0,
    "B2": 1,
    "B3": 2
    }

mapping_to_gender = {
    "F": 0,
    "M": 1,
    }

sample_rate = 22050 #22050
cutoff_length = 8
cutoff_signal_length = cutoff_length * sample_rate

def get_mapping(label):
    return mapping_to_inteligibility.get(label)

def get_audio_files(id, signal_count, signal_data, duration_of_clips):
    path_to_audio_samples = base_path_to_audio + id

    i = 1
    for filename in glob.glob(os.path.join(path_to_audio_samples, '*.wav')):
        signal = librosa.load(filename, sr=sample_rate)

        with contextlib.closing(wave.open(filename,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            duration_of_clips.append(duration)

        #For cutting off long length audio clips
        if len(signal[0]) < cutoff_signal_length:

            filename = filename[filename.rfind("\\") + 1: filename.rfind(".")]
            temp_signal = signal[0]
            labeling = samples.get(id)
            add_to_list(signal_data, filename, labeling, temp_signal)
            signal_count += 1
            # if i == 1: #Number of samples per person
            #     break
            # i = i + 1

        if signal_count % 100 == 0:
            print(signal_count, "at file: " + filename[filename.rfind("\\") + 1: filename.rfind(".")])

    return signal_count, duration_of_clips

def pad_signals_to_equal_length_repeating(signal_dic, padded_signal_data):
    max_length = 0
    for item in signal_dic:
        length_of_signal = len(item[4])
        if length_of_signal > max_length:
            max_length = length_of_signal

    for item in signal_dic:
        base_length_of_clip = len(item[4])
        gender = item[0]
        ident = item[1]
        set_id = item[2]
        labeling = item[3]
        signal = item[4]
        dup_signal = []
        dup_signal = np.array(dup_signal)

        while ((len(dup_signal) + base_length_of_clip) <= max_length):
            dup_signal = np.concatenate((dup_signal, signal))

        dup_signal = pad(dup_signal, max_length)

        temp_arr = [gender, ident, set_id, labeling, dup_signal]
        padded_signal_data.append(temp_arr)

def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr

def add_to_list(list_data, filename, labeling, signal):
    gender = mapping_to_gender.get(filename[0])
    ident = int(filename[1:3])
    set_id = mapping_to_set.get(filename[4:6])
    temp_arr = [gender, ident, set_id, labeling, signal]
    list_data.append(temp_arr)


def get_mfccs(signal):

    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=4096, hop_length=1024)

    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)

    mfcc = np.reshape(mfcc, mfcc.shape[0]*mfcc.shape[1])
    mfcc_delta = np.reshape(mfcc_delta, mfcc_delta.shape[0]*mfcc_delta.shape[1])
    mfcc_delta2 = np.reshape(mfcc_delta2, mfcc_delta2.shape[0]*mfcc_delta2.shape[1])

    total = np.concatenate((mfcc, mfcc_delta, mfcc_delta2))

    return total
    # return mfcc

#%%
# import audio clips
t0 = time.time()
signal_count = 0
signal_data = list()
padded_signal_data = list()
duration_of_clips = list()
print('Starting processing of audio clips into signals...')

for key in samples.keys():

    signal_count, duration_of_clips = get_audio_files(key, signal_count, signal_data, duration_of_clips)

t1 = time.time()
print('Finished processing... time taken: ', t1-t0)
print('Total audio clips processed: ', signal_count)

#%%

#Plot distribution of audio clip times
plt.hist(duration_of_clips, density=False, bins=30)
plt.title('Distribution of audio clip times')
plt.ylabel('Count')
plt.xlabel('Seconds');

N = 60
plt.gca().margins(x=0)
plt.gcf().canvas.draw()
tl = plt.gca().get_xticklabels()
maxsize = max([t.get_window_extent().width for t in tl])
m = 0.2 # inch margin
s = maxsize/plt.gcf().dpi*N+2*m
margin = m/plt.gcf().get_size_inches()[0]
plt.gcf().subplots_adjust(left=margin, right=1.-margin)
plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

plt.xticks(np.arange(1, 60, 1.0))
plt.savefig("distribution.png")
plt.show()

#%%
print('Starting padding of signals to same length...')
pad_signals_to_equal_length_repeating(signal_data, padded_signal_data)

t2 = time.time()
print('Finished padding of signals to same length... time taken: ', t2-t1)

#%%
def generate_mfcc(item):

    gender = item[0]
    ident = item[1]
    set_id = item[2]
    labeling = item[3]
    signal = item[4]

    new_arr = np.concatenate(([gender], [ident], [set_id], [labeling], get_mfccs(signal)))
    new_arr = np.reshape(new_arr, (1, new_arr.shape[0]))

    return new_arr

filename = 'output_dataset39mfcc.h5'
i = 0
for item in padded_signal_data:
    if i == 0:
        new_arr = generate_mfcc(item)

        ROW_SIZE = new_arr.shape[1]
        NUM_COLUMNS = len(padded_signal_data)

        f = tables.open_file(filename, mode='w')
        atom = tables.Float64Atom()

        array_c = f.create_earray(f.root, 'data', atom, (0, ROW_SIZE))
        array_c.append(new_arr)
        i = i + 1
        f.close()
    else:
        new_arr = generate_mfcc(item)
        f = tables.open_file(filename, mode='a')
        f.root.data.append(new_arr)
        f.close()
