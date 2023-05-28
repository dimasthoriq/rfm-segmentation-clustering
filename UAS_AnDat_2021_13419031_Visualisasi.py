#!/usr/bin/env python
# coding: utf-8

# # Statistika Deskriptif dan Visualisasi

# In[2]:


#import packages libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from pandas_profiling import ProfileReport


# In[3]:


#import dataset after cleaning & preprocessing
dataset = pd.read_excel('dataset visualisasi.xlsx')


# In[5]:


dataset.head(10)


# In[7]:


print(dataset.describe())


# In[8]:


dataset["Kategori Produk"].unique()


# In[62]:


#The most sold product categories
df = dataset[['Kategori Produk','Qty']].groupby(['Kategori Produk'])['Qty'].sum().reset_index(name='sum').sort_values(['sum'], ascending=False).head(10)
print(df)


# In[63]:


#convert to dataframe
data = pd.DataFrame(df)
print(data)


# In[65]:


#visualize with bar plot
names = data["Kategori Produk"]
values = data["sum"]

plt.figure(figsize=(50, 6))
plt.title("Penjualan Terbanyak", fontsize = 16)
plt.xlabel ('Kategori Produk')
plt.ylabel('Banyak Penjualan')

plt.subplot(131)
plt.bar(names, values)
plt.show()


# In[6]:


dataset.nunique()


# In[20]:


#Aggregate spendings from every customer category
dataset.groupby(['Kategori Pelanggan'])['Harga Bruto'].sum()


# In[21]:


dataset['Kategori Pelanggan'].unique()


# In[126]:


#pie chart to see the proportion of income from each customer category
labels = 'Grosir', 'Hotel&Rest', 'Langganan', 'Umum'
penjualan = [31416178119, 693088538, 4895343244, 7626510796]
explode = (0, 0, 0, 0) 

fig1, ax1 = plt.subplots()
ax1.pie(penjualan, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = ['tomato', 'gold', 'skyblue','violet','lime'])
ax1.set_title('Penjualan Berdasarkan Kategori Pelanggan', fontsize=14)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[158]:


#dataframe for GGS branch
ggs_df = dataset[dataset['Gudang'] == 'GGS']
ggs = pd.DataFrame(ggs_df)
ggs.head(5)


# In[159]:


print(ggs.describe())


# In[92]:


#most sold product categories in GGS branch
df_for_ggs = ggs[['Kategori Produk','Qty']].groupby(['Kategori Produk'])['Qty'].sum().reset_index(name='sum').sort_values(['sum'], ascending=False).head(10)
print(df_for_ggs)


# In[93]:


#convert the sorted data into dataframe
dataggs = pd.DataFrame(df_for_ggs)
print(dataggs)


# In[148]:


#Visualize with bar chart
names = dataggs["Kategori Produk"]
values = dataggs["sum"]

plt.figure(figsize=(20, 5))
plt.title("Penjualan Terbanyak Toko GGS", fontsize = 20)
plt.xlabel ("Kategori Produk", fontsize = 10)
plt.ylabel("Banyak Penjualan", fontsize = 10)

plt.bar(names, values, color = (1, 0, 0.5, 1))
plt.show()


# In[156]:


#dataframe for GHL branch
ghl_df = dataset[dataset['Gudang'] == 'GHL']
ghl = pd.DataFrame(ghl_df)
ghl.head(5)


# In[157]:


print(ghl.describe())


# In[96]:


#most sold product categories in GHL branch
df_for_ghl = ghl[['Kategori Produk','Qty']].groupby(['Kategori Produk'])['Qty'].sum().reset_index(name='sum').sort_values(['sum'], ascending=False).head(10)
print(df_for_ghl)


# In[97]:


#convert the sorted data into dataframe
dataghl = pd.DataFrame(df_for_ghl)
print(dataghl)


# In[149]:


#Visualize with bar chart
names = dataghl["Kategori Produk"]
values = dataghl["sum"]

plt.figure(figsize=(20, 5))
plt.title("Penjualan Terbanyak Toko GHL", fontsize = 20)
plt.xlabel ("Kategori Produk", fontsize = 10)
plt.ylabel("Banyak Penjualan", fontsize = 10)

plt.bar(names, values, color = (0, 0.75, 0.5, 1))
plt.show()


# In[163]:


#dataframe for GHI branch
ghi_df = dataset[dataset['Gudang'] == 'GHI']
ghi = pd.DataFrame(ghi_df)
ghi.head(5)


# In[169]:


print(ghi.describe())


# In[165]:


#most sold product categories in GHI branch
df_for_ghi = ghi[['Kategori Produk','Qty']].groupby(['Kategori Produk'])['Qty'].sum().reset_index(name='sum').sort_values(['sum'], ascending=False).head(10)
print(df_for_ghi)


# In[166]:


#convert the sorted data into dataframe
dataghi = pd.DataFrame(df_for_ghi)
print(dataghi)


# In[168]:


#Visualize with bar chart
names = dataghi["Kategori Produk"]
values = dataghi["sum"]

plt.figure(figsize=(20, 5))
plt.title("Penjualan Terbanyak Toko GHI", fontsize = 20)
plt.xlabel ("Kategori Produk", fontsize = 10)
plt.ylabel("Banyak Penjualan", fontsize = 10)

plt.bar(names, values, color = (0, 0.25, 0.85, 1))
plt.show()


# In[180]:


#import RFM dataset
df2 = pd.read_excel('final.xlsx')
df2.head()
df2 = df2.sample(n=150, random_state=15)


# In[181]:


plt.figure(figsize = (10, 6))
sns.heatmap(df2.corr(),vmin=-1, vmax=1, center=0, annot = True, cmap="mako")
plt.title("Correlation Plot between Variables")
plt.show()


# In[4]:


#Additional exploration
profile = ProfileReport(dataset, title="Pandas Profiling Report 2", explorative=True)

profile.to_file("Report Profiling 2.html")

