import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression





df = pd.read_csv("homeprices.csv")
plt.xlabel('area (sq. ft.)')
plt.ylabel('prices (USD)')
plt.scatter(df.area,df.prices,color='red',marker='+') 
plt.show()

X=df.iloc[:,0].values
Y=df.iloc[:,1].values

reg = LinearRegression()
reg.fit(X.reshape(-1,1),Y)

reg.predict([[3300]])





