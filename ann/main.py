import network
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/data_banknote_authentication.txt", delimiter=",", header=None)
X = df.iloc[:, :4].to_numpy(dtype = np.longdouble)
y = df.iloc[:, 4:5].to_numpy(dtype = np.longdouble)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = network.Network(inputDim=4, intializationScheme='glorot-uniform')
model.addLayer(dim = 3, activation='sigmoid')
model.addLayer(dim = 1, activation='sigmoid')
model.compile(optimizer='adam', batch_type="mbgd", batch_size=32)
model.train(X_train.T, y_train.T, alpha=0.1, epochs = 100)

pred, cf, accuracy = model.predict(X_test.T, y_test.T)
print("confusion matrix: \n", cf)
print("accuracy: ", accuracy)