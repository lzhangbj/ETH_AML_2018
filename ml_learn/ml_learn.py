import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
# print(check_output(["ls", "input"]).decode("utf8"))

data = pd.read_csv('input/column_2C_weka.csv')
# print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')
# print(data.head())
# print(data.info())
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()

