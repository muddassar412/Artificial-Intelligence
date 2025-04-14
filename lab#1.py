import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
x = np.array([[5],[15],[25],[35],[45],[55]])#.reshape((-1,1))
y = np.array([5,20,14,32,22,38])
modle = linear_model.LinearRegression()
modle.fit(x,y)
x_new = np.array([150]).reshape((-1,1))
y-new = modle.predict(x_new)
print(y-new)
