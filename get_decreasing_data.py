
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

# data = pd.read_csv("rafael/train_sample.csv")


# In[28]:

def read_train_data(sample=True):
    if sample:
        return pd.read_csv("rafael/train_sample.csv")
    else:
        return pd.read_csv("rafael/train.csv")


# In[3]:

# cols = data.columns
# cols


# In[38]:



# posX = [col for col in cols if col.startswith("posX")]
# posY = [col for col in cols if col.startswith("posY")]
# posZ = [col for col in cols if col.startswith("posZ")]

# velX = [col for col in cols if col.startswith("velX")]
# velY = [col for col in cols if col.startswith("velY")]
# velZ = [col for col in cols if col.startswith("velZ")]


# In[33]:

# data[posZ]


# In[34]:

# data[posZ].apply(lambda x: x.max()>x[x.notnull()[-1]], axis=1).sum()


# In[66]:

def get_decreaase_indices(data):
    columns = get_columns("posZ", data.columns)
    return data[columns].apply(lambda x: x.max()>x[x.notnull()][-1], axis=1)


# In[40]:




# In[69]:

def read_train_data(sample=True):
    if sample:
        return pd.read_csv("rafael/train_sample.csv")
    else:
        return pd.read_csv("rafael/train.csv")

all_data = read_train_data()

def get_columns(prefix, data_columns):
    return [col for col in data_columns if col.startswith(prefix)]

posX, posY, posZ, velX, velY, velZ = [get_columns(column, all_data.columns) for column in 
                                      ("posX", "posY", "posZ", "velX", "velY", "velZ")]


def get_decrease_indices(data):
    return data[posZ].apply(lambda x: x.max()>x[x.notnull()][-1], axis=1)


def max_z_column_index(data):
    col_name_z_max = data[posZ].apply(lambda x: x.argmax(), axis=1)
    max_column_index = col_name_z_max.apply(lambda x: int(x.split("_")[-1]))
    return max_column_index


def indices_of_decreaing_z_rows(data):
    decrease_indices = data[posZ].apply(lambda x: x.max()>x[x.notnull()][-1], axis=1)
    return decrease_indices


def get_last_nonnull_index(data):
    return data.apply(lambda x: x.notnull().sum(), axis=1)


def build_dataframe(data, first_index, last_index):
    """
    arg data: might be all the data or subset of the data.
    """
    data["start_decrease"] = first_index
    data["end_data"] = last_index

    data_frame = pd.concat(
              [data.apply(lambda x: x.loc[posX][x.start_decrease:x.end_data].tolist(), axis=1), 
               data.apply(lambda x: x.loc[posY][x.start_decrease:x.end_data].tolist(), axis=1),
               data.apply(lambda x: x.loc[posZ][x.start_decrease:x.end_data].tolist(), axis=1),
               data.apply(lambda x: x.loc[velX][x.start_decrease:x.end_data].tolist(), axis=1),
               data.apply(lambda x: x.loc[velY][x.start_decrease:x.end_data].tolist(), axis=1),
               data.apply(lambda x: x.loc[velZ][x.start_decrease:x.end_data].tolist(), axis=1)], 
               axis=1)
    data_frame.columns = ["posX", "posY", "posZ", "velX", "velY", "velZ"]
    return data_frame


# In[70]:

def get_decreasing_df(sample=True):
    decrease_indices =  get_decrease_indices(read_train_data(sample))
    data_z_decreasing = all_data[decrease_indices]
    last_sample_index = get_last_nonnull_index(all_data[posZ])
    first_decreasing_z_index = max_z_column_index(all_data)
    data_frame = build_dataframe(data_z_decreasing, first_decreasing_z_index, last_sample_index)
    return data_frame


# In[ ]:



