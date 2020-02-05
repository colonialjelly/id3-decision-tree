from id3 import ID3Classifier
from minipanda import Data
import numpy as np

if __name__ == '__main__':
    data_train = Data(fpath='data/house_train.csv')
    data_test = Data(fpath='data/house_test.csv', split_label=True)
    id3 = ID3Classifier(max_depth=3)
    id3.fit(data_train)
    y_pred = id3.predict(data_test)
    print("Test Accuracy: ", np.mean(y_pred == data_test.label))
