# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:23:14 2021

@author: Justin
"""
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import regularizers
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import seaborn as sn
import tables
import statistics

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
    0: "B1",
    1: "B2",
    2: "B3"
    }

mapping_to_gender = {
    0: "F",
    1: "M",
    }

dataset_headers = ["gender", "ident", "set_id", "label", "mfccs"]

def get_model(input_shape):

    #Define model
    model = models.Sequential()

    model.add(layers.Dense(input_shape, activation='relu', input_shape=(input_shape,))) #for already flattened input shape to 1D
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(4, activation='softmax'))
    print(model.summary())

    return model

def perform_fold(X_train, y_train, X_test, y_test):
    input_shape = X_train.shape[1]
    print(input_shape)
    model = get_model(input_shape)

    #Compile the model
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    #Train the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split= 0.15)

    results = model.evaluate(X_test, y_test)
    print(results)
    loss, accuracy = results[0], results[1]

    return history, model, accuracy, loss

def plot_acc_loss(history):
    print(history.history.keys())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def find_predicted(predictions):
    #Turn highest probability to 1 and everything else zero, also turn to 1D array
    y_pred = list()

    for i in range(0, len(predictions)):
        pred_sample = predictions[i]
        pred_index = np.argmax(pred_sample)
        y_pred.append(pred_index)

    y_pred = np.array(y_pred)
    return y_pred

def plot_std_matrix(std_matrix, title):

    df_cm = pd.DataFrame(std_matrix, range(std_matrix.shape[0]), range(std_matrix.shape[1]))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    cm.set_title(title)
    plt.show()

def print_stats(loss, accuracy):
    for i in range(0, len(loss)):
        print('Fold ' + str(i) + ': Loss: %(loss)f, Accuracy: %(accuracy)f' % {'loss': loss[i], 'accuracy': accuracy[i]})

    print('Average loss: %(avg_loss)f, Average accuracy: %(avg_accuracy)f' % {'avg_loss': np.mean(loss), 'avg_accuracy': np.mean(accuracy)})

def plot_average_confusion_matrix(all_cm):
    shape = all_cm[0].shape
    avg_cm = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            total = 0
            for k in range(0, len(all_cm)):
                current_cm = all_cm[k]
                total += current_cm[i][j]

            avg = total / len(all_cm)

            avg_cm[i][j] = avg

    return avg_cm

#%%
#Import dataset
filename = 'output_dataset13mfcc.h5'
f = tables.open_file(filename)
full_dataset = f.root.data[:,:] #Column names Gender, Identification, Set_id, Label, MFCCs

#%%
#Split the data into training, validation, test sets

B1, B2, B3 = full_dataset[full_dataset[:,2] == 0,:], full_dataset[full_dataset[:,2] == 1,:], full_dataset[full_dataset[:,2] == 2,:] #Split dataset according to set_id
B1, B2, B3 = np.delete(B1, 2, 1), np.delete(B2, 2, 1), np.delete(B3, 2, 1) #Remove the set_id column
X_b1, X_b2, X_b3 = B1[:,3:], B2[:,3:], B3[:,3:] #Take MFCCs
y_b1, y_b2, y_b3 = B1[:,2], B2[:,2], B3[:,2] #Take label
y_b1_true, y_b2_true, y_b3_true = B1[:,:3], B2[:,:3], B3[:,:3] #Extract all information of samples

#%%
#Define folds
folds = [[X_b1, y_b1, X_b2, y_b2, X_b3, y_b3, y_b3_true],
         [X_b2, y_b2, X_b3, y_b3, X_b1, y_b1, y_b1_true],
         [X_b1, y_b1, X_b3, y_b3, X_b2, y_b2, y_b2_true]]

all_cm_for_class = list()
all_cm_for_gender = list()
all_cm_for_individual = list()
all_accuracy = list()
all_loss = list()

for fold in folds:
    X_train = np.concatenate((fold[0], fold[2]))
    y_train = np.concatenate((fold[1], fold[3]))
    X_test = fold[4]
    y_test = fold[5]
    y_test_with_name = fold[6]

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test, y_test_with_name = shuffle(X_test, y_test, y_test_with_name, random_state=0)

    # Scale the inputs
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    one_hot_train_labels = to_categorical(y_train)
    one_hot_test_labels = to_categorical(y_test)

    #Perform the folds
    history, model, accuracy, loss = perform_fold(X_train, one_hot_train_labels, X_test, one_hot_test_labels)

    all_accuracy.append(accuracy)
    all_loss.append(loss)

    plot_acc_loss(history) #plot graphs

    predictions = model.predict(X_test)
    y_pred = find_predicted(predictions)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    all_cm_for_class.append(cm)
    df_cm = pd.DataFrame(cm, range(4), range(4))
    sn.set(font_scale=1.4) # for label size
    cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    cm.set_title('Confusion Matrix for classes')
    cm.set_ylabel('True')
    cm.set_xlabel('Pred')
    plt.show()


    cm_for_gender = np.zeros((2, 2))
    for index, value in enumerate(y_test_with_name):
        if "F" in mapping_to_gender.get(value[0]) and float(value[2]) == y_pred[index]:
            cm_for_gender[0][0] += 1
        elif "F" in mapping_to_gender.get(value[0]) and float(value[2]) != y_pred[index]:
            cm_for_gender[0][1] += 1
        elif "M" in mapping_to_gender.get(value[0]) and float(value[2]) == y_pred[index]:
            cm_for_gender[1][0] += 1
        elif "M" in mapping_to_gender.get(value[0]) and float(value[2]) != y_pred[index]:
            cm_for_gender[1][1] += 1

    cm_for_gender = normalize(cm_for_gender, axis=1, norm='l1')
    all_cm_for_gender.append(cm_for_gender)
    df_cm = pd.DataFrame(cm_for_gender, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, yticklabels=["Female", "Male"], xticklabels=["Correct", "Wrong"]) # font size
    cm.set_title('Confusion Matrix for gender')
    cm.set_ylabel('Gender')
    cm.set_xlabel('Pred')
    plt.show()


    cm_for_individual = np.zeros((10, 2))
    for index, value in enumerate(y_test_with_name):
        i = 0
        individual_id = mapping_to_gender.get(value[0]) + "0" + str(int(value[1]))
        for key in samples.keys():
            if key in individual_id and float(value[2]) == y_pred[index]:
                cm_for_individual[i][0] += 1

            elif key in individual_id and float(value[2]) != y_pred[index]:
                cm_for_individual[i][1] += 1

            i += 1

    cm_for_individual = normalize(cm_for_individual, axis=1, norm='l1')
    all_cm_for_individual.append(cm_for_individual)
    df_cm = pd.DataFrame(cm_for_individual, range(10), range(2))
    sn.set(font_scale=1.4) # for label size
    y_axis = samples.keys()
    cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, yticklabels=y_axis, xticklabels=["Correct", "Wrong"]) # font size
    cm.set_title('Confusion Matrix for individual')
    cm.set_ylabel('Individual')
    cm.set_xlabel('Pred')
    plt.show()


# model.save('DysarthricModel.h5')


#%%
#Print averages
print_stats(all_loss, all_accuracy)

#%%
#Print averages
#Accuracy of classes
avg_cm_for_class = plot_average_confusion_matrix(all_cm_for_class)
df_cm = pd.DataFrame(avg_cm_for_class, range(4), range(4))
sn.set(font_scale=1.4) # for label size
cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
cm.set_title('Average Confusion Matrix for classes')
cm.set_ylabel('True')
cm.set_xlabel('Pred')
plt.show()

# #Accuracy of gender
avg_cm_for_gender = plot_average_confusion_matrix(all_cm_for_gender)
df_cm = pd.DataFrame(avg_cm_for_gender, range(2), range(2))
sn.set(font_scale=1.4) # for label size
cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, yticklabels=["Female", "Male"], xticklabels=["Correct", "Wrong"]) # font size
cm.set_title('Average Confusion Matrix for gender')
cm.set_ylabel('Gender')
cm.set_xlabel('Pred')
plt.show()

#Accuracy of Individuals
avg_cm_for_individual = plot_average_confusion_matrix(all_cm_for_individual)
df_cm = pd.DataFrame(avg_cm_for_individual, range(10), range(2))
sn.set(font_scale=1.4) # for label size
y_axis = samples.keys()
cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, yticklabels=y_axis, xticklabels=["Correct", "Wrong"]) # font size
cm.set_title('Average Confusion Matrix for individual')
cm.set_ylabel('Individual')
cm.set_xlabel('Pred')
plt.show()

#%%
#Calculate Standard Deviation
std_accuracy = np.std(all_accuracy)
std_loss = np.std(all_loss)


std_class = np.zeros((4,4))
for i in range(0, all_cm_for_class[0].shape[0]):
    for j in range(0, all_cm_for_class[0].shape[1]):
        temp_arr = []
        for k in range(0, len(all_cm_for_class)):
            temp_arr.append(all_cm_for_class[k][i][j])

        std_class[i][j] = np.std(temp_arr)
plot_std_matrix(std_class, "Std for class")

std_gender = np.zeros((2,2))
for i in range(0, all_cm_for_gender[0].shape[0]):
    for j in range(0, all_cm_for_gender[0].shape[1]):
        temp_arr = []
        for k in range(0, len(all_cm_for_gender)):
            temp_arr.append(all_cm_for_gender[k][i][j])

        std_gender[i][j] = np.std(temp_arr)
plot_std_matrix(std_gender, "Std for gender")

std_individual = np.zeros((10,2))
for i in range(0, all_cm_for_individual[0].shape[0]):
    for j in range(0, all_cm_for_individual[0].shape[1]):
        temp_arr = []
        for k in range(0, len(all_cm_for_individual)):
            temp_arr.append(all_cm_for_individual[k][i][j])

        std_individual[i][j] = np.std(temp_arr)
plot_std_matrix(std_individual, "Std for individual")

#%%
#Train model on all data

X_train = np.concatenate((X_b1, X_b2, X_b3))
y_train = np.concatenate((y_b1, y_b2, y_b3))

X_train, y_train = shuffle(X_train, y_train, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

one_hot_train_labels = to_categorical(y_train)
print(X_train.shape[1])
model = get_model(X_train.shape[1])

opt = optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, one_hot_train_labels, batch_size=32, epochs=20)
#%%
print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()