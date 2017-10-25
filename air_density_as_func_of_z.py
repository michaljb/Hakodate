
# coding: utf-8

# In[1]:

import get_decreasing_data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as cnc

get_ipython().magic(u'matplotlib inline')


# In[2]:

air_pressure = pd.read_html("https://en.wikipedia.org/wiki/Barometric_formula")[2]
air_pressure


# In[3]:

x = np.array([int(i.replace(" ", "")) for i in air_pressure[1].tolist()[2:]])
y = np.array([float(i) for i in air_pressure[2].tolist()[2:]])


# In[4]:

a, b = np.polyfit(x, y, deg=1)
y_fit = a*x + b
yy = np.array([y, y_fit]).T


# In[6]:

plt.plot(x,yy)


# In[7]:

a, b


# In[ ]:



