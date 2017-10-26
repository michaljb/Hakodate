
# coding: utf-8

# In[49]:

import get_decreasing_data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as cnc

get_ipython().magic(u'matplotlib inline')


# In[74]:

def plot_series_as_func_of_time(col):
    y = np.array(col)
    x = x = np.arange(0,y.shape[0])/2
    plt.plot(x,y)
    return x, y


# In[75]:

df = get_decreasing_data.get_decreasing_df()


# In[76]:

df["posZ"]


# In[79]:

_ = plot_series_as_func_of_time(df["posZ"].iloc[0])


# In[ ]:

x, y = plot_series_as_func_of_time(df["velZ"].iloc[0])


# In[ ]:



