import csv
import numpy as np

train_X = []
train_Y = []
test_X = []
test_Y = []
with open('data/fer2013.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        pixels = row[1]
        pix_array = np.array(pixels.split())
        pix_array = pix_array.astype('int32')
        pix_array = np.reshape(pix_array, (48, 48, 1))
        if row[2] == 'Training':
            train_X.append(pix_array)
            train_Y.append(int(row[0]))
        else:
            test_X.append(pix_array)
            test_Y.append(int(row[0]))

train_X = np.array(train_X)
test_X = np.array(test_X)
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)
np.save('train_X.npy', train_X)
np.save('test_X.npy', test_X)
np.save('train_Y.npy', train_Y)
np.save('test_Y.npy', test_Y)


