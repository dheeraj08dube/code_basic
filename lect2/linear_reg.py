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

reg.coef_
reg.intercept_

135.78767123*3300+180616.43835616432
135.78767123*5000+180616.43835616432

d = pd.read_csv("areas.csv")

d.head(3)

reg.predict(d)
p = reg.predict(d)
d['prices'] = p

d.to_csv("predictions.csv",index=False)

df = pd.read_csv("homeprices.csv")
plt.xlabel('area (sq. ft.)')
plt.ylabel('prices (USD)')
plt.scatter(df.area,df.prices,color='red',marker='+') 
plt.plot(df.area,reg.predict(df[['area']]),color='blue') 
plt.show()


