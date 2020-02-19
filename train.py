#!/usr/bin/env python

import os

import numpy as np
from keras import backend as K
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight

from custom_data_generator import CustomDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# OPTIONS #
IMAGE_SIZE = 224
CLASSIFY = 0
REGRESS = 1


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def process_data(t_type, paths, labels):
    labels_out = []
    paths_out = []
    count = 0
    for i, (emotion, valence, arousal) in enumerate(labels):
        if t_type == CLASSIFY:
            if emotion > 7:
                continue
            labels_out.append(emotion)
            paths_out.append(paths[i])
        else:
            if arousal == -2 or valence == -2:
                continue
            labels_out.append([valence, arousal])
            paths_out.append(paths[i])
        count += 1
        print('Processed:', count, end='\r')
    if t_type == CLASSIFY:
        weights = class_weight.compute_class_weight('balanced', np.unique(labels_out), labels_out)
        weights = dict(enumerate(weights))
        labels_out = to_categorical(labels_out, num_classes=8)
    else:
        weights = None
    print('Processed:', count)
    return paths_out, labels_out, weights


def mobilenet_v2_model(t_type, dropout=0.5):
    base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)
    if t_type == CLASSIFY:
        predictions = Dense(8, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
    else:
        predictions = Dense(2, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003), metrics=[root_mean_squared_error])

    return model


def regressor_from_classifier(c_model, dropout=False):
    x = c_model.output
    if dropout:
        x = Dropout(0.5)(x)
    predictions = Dense(2, activation='linear', name='regression_output')(x)
    model = Model(inputs=c_model.input, outputs=predictions)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003))

    return model


def train(t_type, model, output_path, epochs, batch_size):
    print('** LOADING DATA **')
    t_paths = np.load('training_paths.npy')
    t_labels = np.load('training_labels.npy')
    t_paths, t_labels, t_weights = process_data(t_type, t_paths, t_labels)
    v_paths = np.load('validation_paths.npy')
    v_labels = np.load('validation_labels.npy')
    v_paths, v_labels, v_weights = process_data(t_type, v_paths, v_labels)

    train_generator = CustomDataGenerator(t_paths, t_labels, batch_size, shuffle=True, augment=True)
    val_generator = CustomDataGenerator(v_paths, v_labels, batch_size)

    t_steps = len(t_labels) // batch_size
    v_steps = len(v_labels) // batch_size

    print('** TRAINING MODEL **')
    if t_type == CLASSIFY:
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=t_steps,
            class_weight=t_weights,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=v_steps,
            callbacks=[
                ModelCheckpoint(output_path + '{epoch:02d}_{val_loss:.3f}_T.h5', monitor='val_acc',
                                save_best_only=False,
                                save_weights_only=True),
                CSVLogger('log_classification.csv', append=True, separator=';')],
            workers=8,
            use_multiprocessing=True)
    else:
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=t_steps,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=v_steps,
            callbacks=[
                ModelCheckpoint(output_path + '{epoch:02d}_{val_loss:.4f}_T.h5', save_best_only=False,
                                save_weights_only=True),
                CSVLogger('log_regression.csv', append=True, separator=';')],
            workers=8,
            use_multiprocessing=True)

    print('** EXPORTING MODEL **')
    np.save(output_path + '_HIST', history.history)
    for layer in model.layers:
        if t_type(layer) is Dropout:
            model.layers.remove(layer)
    model.save_weights(output_path + '_weights.h5')
    model.save(output_path + '_full_model.h5')


if __name__ == '__main__':
    '''
                    CLASSIFICATION METRICS
    ------------------------------------------------------
    ACC                  0.5992
    F1                   0.6
    KAPPA                0.542
    ALPHA                0.5419
    AUCPR                0.6529
    AUC                  0.9127
            
    Confusion Matrix:
    
         N   H   Sa  Su  Af  D   An  C
    N  [280  11  50  40  16  15  41  47]
    H  [ 19 371   6  24   3  11   3  63]
    Sa [ 65   6 314  15  27  21  47   5]
    Su [ 45  33  24 293  69  16  11   9]
    Af [ 22   8  36  79 312  24  19   0]
    D  [ 28  17  47  14  15 269  91  19]
    An [ 65   4  28  21  25  48 293  16]
    C  [ 82  66  16  14   4  25  28 265]    

    F-score:
    
                  precision    recall  f1-score   support
               0       0.46      0.56      0.51       500
               1       0.72      0.74      0.73       500
               2       0.60      0.63      0.62       500
               3       0.59      0.59      0.59       500
               4       0.66      0.62      0.64       500
               5       0.63      0.54      0.58       500
               6       0.55      0.59      0.57       500
               7       0.62      0.53      0.57       500
        accuracy                           0.60      4000
       macro avg       0.60      0.60      0.60      4000
    weighted avg       0.60      0.60      0.60      4000
    
    
    
                      REGRESSION METRICS
    ------------------------------------------------------
                         VALENCE              AROUSAL
    RMSE                 0.3949               0.3755
    CORR                 0.6162               0.5465
    SAGR                 0.7562               0.7458
    CCC                  0.5822               0.463
    '''
