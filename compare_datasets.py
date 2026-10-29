#!/usr/bin/env python3
import numpy as np

data1 = np.load('dataset_ml_v2.npz')
print('Unfiltered dataset (dataset_ml_v2.npz):')
print('  Train WR:', f"{data1['y_train'].mean():.1%}")
print('  Test WR:', f"{data1['y_test'].mean():.1%}")
print('  Total:', len(data1['y_train']) + len(data1['y_test']), 'sequences')
print()

data2 = np.load('dataset_ml_filtered.npz')
print('Filtered dataset (dataset_ml_filtered.npz):')
print('  Train WR:', f"{data2['y_train'].mean():.1%}")
print('  Test WR:', f"{data2['y_test'].mean():.1%}")
print('  Total:', len(data2['y_train']) + len(data2['y_test']), 'sequences')
