
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pygam import LinearGAM
from sklearn.impute import SimpleImputer

# загружаем файл с данными
df = pd.read_csv("D:/profile/Downloads/echo_salmon_data.csv")

x = SimpleImputer().fit_transform(df[['T_mean_10km']].values)

y = df['All_0_100'].values

o
lams=np.logspace(x.min(), x.max(), 100)
gam = LinearGAM(n_splines=50).gridsearch(x, y,lam=lams)
xx = np.linspace(x.min(), x.max(), 100)
yy = gam.predict(xx)

plt.plot(xx, yy, color='r')
plt.xlabel('Temperature')
plt.ylabel('Fish Count')
plt.plot(xx, gam.prediction_intervals(xx, width=.1), color='b', ls='--')
plt.show()
