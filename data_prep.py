#!/usr/bin/env python

import csv
import os

import numpy as np


def process_labels(file_path):
    labels = []
    paths = []
    with open(file_path, 'r') as csv_file:
        r = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in r:
            if 'sub' in row[0]:
                continue
            valence = float(row[-2])
            arousal = float(row[-1])
            emotion = int(row[-3])
            path = 'data/' + row[0]
            if emotion > 7 and (arousal == -2 or valence == -2):
                continue
            if not os.path.exists(path):
                print('error: no image')
                continue
            labels.append((emotion, valence, arousal))
            paths.append(path)
            count += 1
            print('Loaded:', count, end='\r')
    print('Loaded:', count)
    return paths, labels


if __name__ == '__main__':
    t_paths, t_labels = process_labels('training.csv')
    np.save('data/training_paths', t_paths)
    np.save('data/training_labels', t_labels)
    v_paths, v_labels = process_labels('validation.csv')
    np.save('data/validation_paths', v_paths)
    np.save('data/validation_labels', v_labels)
