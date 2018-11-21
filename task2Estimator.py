from sklearn import datasets
from sklearn import linear_model
import numpy as np
import pandas as pd


#data = datasets.load_boston()
data = pd.read_csv("SnakeRangeEastSagebrush_1101_to_1102copy.csv")
targetdata = pd.read_csv("SnakeRangeEastSagebrush_1102_to_1103copy.csv")

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(targetdata)

#print(data)
#   NOTE: To access individual columns, do dataframeVar.["column name"]
#print(target.["Speed"])


x = df
y = target["Speed"]

lm = linear_model.LinearRegression()
model = lm.fit(x,y)

predictions = lm.predict(x)
#print(predictions[0:5])
print(predictions)







