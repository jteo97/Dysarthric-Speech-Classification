# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:09:05 2021

@author: Justin
"""

import numpy as np

import librosa, librosa.display
import glob, os
import time

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

sample_rate = 22050 #22050
cutoff_length = 8
cutoff_signal_length = cutoff_length * sample_rate

def get_mapping(label):
    return mapping_to_inteligibility.get(label)

def get_audio_files(id, signal_count):
    path_to_audio_samples = base_path_to_audio + id

    i = 1
    for filename in glob.glob(os.path.join(path_to_audio_samples, '*.wav')):
        signal = librosa.load(filename, sr=sample_rate)

        #For cutting off long length audio clips
        if len(signal[0]) < cutoff_signal_length:

            filename = filename[filename.rfind("\\") + 1: filename.rfind(".")]
            temp_signal = signal[0]
            labeling = samples.get(id)
            add_to_dict(filename, temp_signal, labeling)
            signal_count += 1
            if i == 2: #Number of samples per person
                break
            i = i + 1

        if signal_count % 100 == 0:
            print(signal_count, "at file: " + filename[filename.rfind("\\") + 1: filename.rfind(".")])

    return signal_count

# def pad_signals_to_equal_length_with_zeros(all_signals):
#     max_length = 0
#     for person in all_signals:
#         for clip in person:
#             if len(clip) > max_length:
#                 max_length = len(clip)

#     list_of_all_padded_signals = list()
#     for person in all_signals:
#         list_of_padded_signals = list()
#         for clip in person:
#             padded_signal = pad(clip, max_length)
#             list_of_padded_signals.append(padded_signal)
#         list_of_all_padded_signals.append(list_of_padded_signals)

#     return list_of_all_padded_signals

# def pad_signals_to_equal_length_repeating(all_signals):
#     max_length = 0
#     for person in all_signals:
#         for clip in person:
#             if len(clip) > max_length:
#                 max_length = len(clip)

#     list_of_all_padded_signals = list()
#     for person in all_signals:
#         list_of_padded_signals = list()
#         for clip in person:
#             base_length_of_clip  = len(clip)
#             dup_signal = []
#             dup_signal = np.array(dup_signal)

#             while ((len(dup_signal) + base_length_of_clip) <= max_length):
#                 dup_signal = np.concatenate((dup_signal, clip))

#             dup_signal = pad(dup_signal, max_length)
#             list_of_padded_signals.append(dup_signal)
#         list_of_all_padded_signals.append(list_of_padded_signals)

#     return list_of_all_padded_signals

def pad_signals_to_equal_length_repeating(signal_dic):
    max_length = 0
    for key in signal_dic.keys():
        info = signal_dic.get(key)
        length_of_signal = len(info[0])
        if length_of_signal > max_length:
                max_length = length_of_signal

    for key in signal_dic.keys():
        info = signal_dic.get(key)
        base_length_of_clip = len(info[0])
        signal = info[0]
        labeling = info[1]
        dup_signal = []
        dup_signal = np.array(dup_signal)

        while ((len(dup_signal) + base_length_of_clip) <= max_length):
            dup_signal = np.concatenate((dup_signal, signal))

        dup_signal = pad(dup_signal, max_length)

        signal_dic.update({key: [dup_signal, labeling]})

    return signal_dic

def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr

def add_to_dict(filename, signal, labeling):
    signal_data[filename] = [signal, labeling]

def split_samples_to_sets(signal_dic, sets):
    B1, B2, B3 = [], [], []
    for key in signal_dic.keys():
        info = signal_dic.get(key)
        signal = info[0]
        labeling = info[1]
        if sets[0] in key:
            new_arr = np.concatenate(([labeling], signal))
            signal = np.reshape(new_arr, (1, new_arr.shape[0]))
            B1.append(signal)
        elif sets[1] in key:
            new_arr = np.concatenate(([labeling], signal))
            signal = np.reshape(new_arr, (1, new_arr.shape[0]))
            B2.append(signal)
        elif sets[2] in key:
            new_arr = np.concatenate(([labeling], signal))
            signal = np.reshape(new_arr, (1, new_arr.shape[0]))
            B3.append(signal)

    return B1, B2, B3

def get_mfccs(signal_list):
    mfccs_list = list()
    mfcc_delta_list = list()
    mfcc_delta2_list = list()
    combined_mfccs = list()

    for data in signal_list:

        signal = data[0][1:]
        labeling = data[0][:1]
        labeling = labeling[0]
        mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13)

        mfcc_delta = librosa.feature.delta(mfcc)

        mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)

        total = np.concatenate((mfcc, mfcc_delta, mfcc_delta2))

        mfcc = np.reshape(mfcc, 13*mfcc.shape[1])
        mfcc = np.concatenate(([labeling], mfcc))
        mfcc = np.reshape(mfcc, (1, mfcc.shape[0]))

        total = np.reshape(total, 39*total.shape[1])
        total = np.concatenate(([labeling], total))
        total = np.reshape(total, (1, total.shape[0]))

        mfccs_list.append(mfcc)
        mfcc_delta_list.append(mfcc_delta)
        mfcc_delta2_list.append(mfcc_delta2)
        combined_mfccs.append(total)

    return mfccs_list, mfcc_delta_list, mfcc_delta2_list, combined_mfccs

#%%
# import audio clips
t0 = time.time()
signal_count = 0
signal_data = dict()
print('Starting processing of audio clips into signals...')

for key in samples.keys():

    signal_count = get_audio_files(key, signal_count)

t1 = time.time()
print('Finished processing... time taken: ', t1-t0)
print('Total audio clips processed: ', signal_count)

#%%
print('Starting padding of signals to same length...')
signal_data = pad_signals_to_equal_length_repeating(signal_data)

t2 = time.time()
print('Finished padding of signals to same length... time taken: ', t2-t1)

#signal = librosa.load(base_path_to_audio + 'F02/'+ 'F02_B1_C10_M3.wav', sr=100)
#%%
# split samples to B1, B2, B3
name = ["B1", "B2", "B3"]
B1_signal, B2_signal, B3_signal = split_samples_to_sets(signal_data, name)

#%%
# save signal arrays
# B1 = np.concatenate(B1_signal, axis=0)
# B2 = np.concatenate(B2, axis=0)
# B3 = np.concatenate(B3, axis=0)

# np.save(path_to_B1_samples + "datasetB1.npy", B1)
# np.save(path_to_B2_samples + "datasetB2.npy", B2)
# np.save(path_to_B3_samples + "datasetB3.npy", B3)

# data1 = np.load(path_to_B1_samples + "datasetB1.npy")
# data2 = np.load(path_to_B2_samples + "datasetB2.npy")
# data3 = np.load(path_to_B3_samples + "datasetB3.npy")

#%%
# convert signals to MFCCs and their deltas
print('Starting generation of MFCCs, delta MFCCs and delta2 MFCCs...')

B1_mfccs, B1_mfccs_delta, B1_mfccs_delta2, B1_combined_mfccs = get_mfccs(B1_signal)
# B2_mfccs, B2_mfccs_delta, B2_mfccs_delta2, B2_combined_mfccs = get_mfccs(B2_signal)
# B3_mfccs, B3_mfccs_delta, B3_mfccs_delta2, B3_combined_mfccs = get_mfccs(B3_signal)

B1_mfccs = np.concatenate(B1_mfccs, axis=0)
B1_combined_mfccs = np.concatenate(B1_combined_mfccs, axis=0)
# B2_mfccs = np.concatenate(B2_mfccs, axis=0)
# B2_combined_mfccs = np.concatenate(B2_combined_mfccs, axis=0)
# B3_mfccs = np.concatenate(B3_mfccs, axis=0)
# B3_combined_mfccs = np.concatenate(B3_combined_mfccs, axis=0)

t3 = time.time()
print('Finished generation of all MFCCs... time taken: ', t3-t2)
#%%
# save the mfccs to npy files
np.save(path_to_B1_samples + "dataset13mfccB1.npy", B1_mfccs)
np.save(path_to_B1_samples + "dataset39mfccB1.npy", B1_combined_mfccs)

# np.save(path_to_B2_samples + "dataset13mfccB2.npy", B2_mfccs)
# np.save(path_to_B2_samples + "dataset39mfccB2.npy", B2_combined_mfccs)

# np.save(path_to_B3_samples + "dataset13mfccB3.npy", B3_mfccs)
# np.save(path_to_B3_samples + "dataset39mfccB3.npy", B3_combined_mfccs)

#%%
# data1 = np.load(path_to_B1_samples + "dataset13mfccB1.npy")
# data2 = np.load(path_to_B1_samples + "dataset39mfccB1.npy")
# #%%
# #Flatten the lists into numpy arrays

# y = [item for sublist in y_label for item in sublist]
# X = [item for sublist in total_mfccs for item in sublist]

# y = np.array(y)
# X = np.array(X)
# X_unflatten = X

# # To flatten X down to 2D array
# X_temp = list()
# # count = 0
# for sample in X:
#     temp = sample.reshape(-1)
#     # print(temp)
#     # x2324 = temp
#     # print(count)
#     # count = count + 1
#     X_temp.append(temp)

# X = np.array(X_temp)

# #%%
# # np.savetxt("dataset&labels/dataset234.csv", X, delimiter=",")
# # np.savetxt("dataset&labels/labels234.csv", y, delimiter=",")

# # t4 = time.time()
# # print('Time taken to generate dataset: ', t4-t0)

# # #%%

# # x3 = list()
# # x5 = list()
# # for q in padded_signals_same_length:
# #     x4 = list()
# #     for w in q:
# #         x2 = np.reshape(w, (1, w.shape[0]))

# #         x4.append(x2)

# #     x3.append(x4)


# # X1 = [item for sublist in x3 for item in sublist]

# # f_0_1 = np.concatenate(X1)

# # #%%
# # np.savetxt("dataset&labels/signals.csv", f_0_1, delimiter=",")


# #%%
# filename = 'M07_B2_CW27_M8'
# signal = all_signals[0][0]
# labeling = 1
# add_to_dict(filename, signal, labeling)

# #%%
# for key in signal_data.keys():
#     x = signal_data.get(key)
#     print(key, x[0], x[1])

# #%%
# dfs = np.array([3,3,3])
# print(dfs.shape)