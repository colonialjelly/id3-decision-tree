# ID3 Decision Tree

ID3 (Iterative Dichotomiser 3) is a decision tree learning algorithm. It works by recursively growing the tree top down. At each
step it decides which attribute (feature) is the most suitable to split on by measuring how much information it provides. Information gain for an attribute A is calculated
as the difference between the entropy of the data and the entropy of the data after taking the split A.

## Usage

The data reader accepts either a numpy array or a path to a csv file. It expects that the header is included. The label column should either be named "label" or a custom name can be
provided during initialization of the Data class. 

The classifier requires that the data be provided as a Data object. An optional max_depth parameter can be provided for pruning. 
The method .fit() can be called to learn the tree and .predict() to generate predictions.

```python
from id3 import ID3Classifier
from minipanda import Data
import numpy as np

data_train = Data(fpath='data/house_train.csv')
data_test = Data(fpath='data/house_test.csv', split_label=True)
id3 = ID3Classifier(max_depth=3)
id3.fit(data_train)
y_pred = id3.predict(data_test)
print("Test Accuracy: ", np.mean(y_pred == data_test.label))
```

## Requirements

- `numpy>=1.17.2`

## References

- Original Paper by Ross Quinlan [Induction of Decision Trees](https://hunch.net/~coms-4771/quinlan.pdf)