
import pandas as pd

import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# загружаем файл с данными
df = pd.read_csv("D:/profile/Downloads/echo_salmon_data.csv")
plt.scatter(df.Lat,df.T_mean_10km)
skm = lm.LinearRegression()
y = SimpleImputer().fit_transform(pd.DataFrame(df.T_mean_10km))
x = pd.DataFrame(df.Lat)
skm.fit(y,x)
print(skm.coef_)
plt.xlabel('Lat')
plt.ylabel('Temperature')
plt.plot(skm.predict(y),y,color='r')
plt.show()
