#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[1]:


#import packages libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
from pandas_profiling import ProfileReport
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[2]:


#import datasets
ggs0918 = pd.read_excel("GGS-RINCIANPENJUALAN-201809.xlsx")
ggs1018 = pd.read_excel("GGS-RINCIANPENJUALAN-201810.xlsx")
ggs1118 = pd.read_excel("GGS-RINCIANPENJUALAN-201811.xlsx")
ggs1218 = pd.read_excel("GGS-RINCIANPENJUALAN-201812.xlsx")
ggs0119 = pd.read_excel("GGS-RINCIANPENJUALAN-201901.xlsx")
ggs0219 = pd.read_excel("GGS-RINCIANPENJUALAN-201902.xlsx")
ggs0319 = pd.read_excel("GGS-RINCIANPENJUALAN-201903.xlsx")
ggs0419 = pd.read_excel("GGS-RINCIANPENJUALAN-201904.xlsx")
ggs0519 = pd.read_excel("GGS-RINCIANPENJUALAN-201905.xlsx")
ggs0619 = pd.read_excel("GGS-RINCIANPENJUALAN-201906.xlsx")
ggs0719 = pd.read_excel("GGS-RINCIANPENJUALAN-201907.xlsx")
ggs0819 = pd.read_excel("GGS-RINCIANPENJUALAN-201908.xlsx")
ggs0919 = pd.read_excel("GGS-RINCIANPENJUALAN-201909.xlsx")

ghi0918 = pd.read_excel("GHI RINCIAN PENJUALAN 201809.txt.xls")
ghi1018 = pd.read_excel("GHI RINCIAN PENJUALAN 201810.txt.xls")
ghi1118 = pd.read_excel("GHI RINCIAN PENJUALAN 201811.txt.xls")
ghi1218 = pd.read_excel("GHI RINCIAN PENJUALAN 201812.txt.xls")
ghi0119 = pd.read_excel("GHI RINCIAN PENJUALAN 201901.txt.xls")
ghi0219 = pd.read_excel("GHI RINCIAN PENJUALAN 201902.txt.xls")
ghi0319 = pd.read_excel("GHI RINCIAN PENJUALAN 201903.txt.xls")
ghi0419 = pd.read_excel("GHI RINCIAN PENJUALAN 201904.txt.xls")
ghi0519 = pd.read_excel("GHI RINCIAN PENJUALAN 201905.txt.xls")
ghi0619 = pd.read_excel("GHI RINCIAN PENJUALAN 201906.txt.xls")
ghi0719 = pd.read_excel("GHI RINCIAN PENJUALAN 201907.txt.xls")
ghi0819 = pd.read_excel("GHI RINCIAN PENJUALAN 201908.txt.xls")
ghi0919 = pd.read_excel("GHI RINCIAN PENJUALAN 201909.xls")

ghl0918 = pd.read_excel("GHL-RINCIAN PENJUALAN-201809.xls")
ghl1018 = pd.read_excel("GHL-RINCIAN PENJUALAN-201810.xls")
ghl1118 = pd.read_excel("GHL-RINCIAN PENJUALAN-201811.xls")
ghl1218 = pd.read_excel("GHL-RINCIAN PENJUALAN-201812.xls")
ghl0119 = pd.read_excel("GHL-RINCIAN PENJUALAN-201901.xls")
ghl0219 = pd.read_excel("GHL-RINCIAN PENJUALAN-201902.xls")
ghl0319 = pd.read_excel("GHL-RINCIAN PENJUALAN-201903.xls")
ghl0419 = pd.read_excel("GHL-RINCIAN PENJUALAN-201904.xls")
ghl0519 = pd.read_excel("GHL-RINCIAN PENJUALAN-201905.xls")
ghl0619 = pd.read_excel("GHL-RINCIAN PENJUALAN-201906.xls")
ghl0719 = pd.read_excel("GHL-RINCIAN PENJUALAN-201907.xls")
ghl0819 = pd.read_excel("GHL-RINCIAN PENJUALAN-201908.xls")
ghl0919 = pd.read_excel("GHL-RINCIAN PENJUALAN-201909.xls")


# In[3]:


#Combine all datasets into one dataframe
ggs = [ggs0918,ggs1018,ggs1118,ggs1218,ggs0119,ggs0219,ggs0319,ggs0419,ggs0519,ggs0619,ggs0719,ggs0819,ggs0919]
ghi = [ghi0918,ghi1018,ghi1118,ghi1218,ghi0119,ghi0219,ghi0319,ghi0419,ghi0519,ghi0619,ghi0719,ghi0819,ghi0919]
ghl = [ghl0918,ghl1018,ghl1118,ghl1218,ghl0119,ghl0219,ghl0319,ghl0419,ghl0519,ghl0619,ghl0719,ghl0819,ghl0919]
sign = ["Gudang","Kode Pelanggan", "Kategori", "Kode Produk", "Kategori.1", "Tanggal","No.Transaksi", "Qty", "Harga Bruto"]
numcol=["Qty", "Harga Bruto"]

df_ggs=pd.DataFrame()
df_ghi=pd.DataFrame()
df_ghl=pd.DataFrame()
for i in ggs:
    x=i[sign]
    df_ggs = df_ggs.append(x)
for i in ghi:
    y=i[sign]
    df_ghi = df_ghi.append(y)
for i in ghl:
    z=i[sign]
    df_ghl = df_ghl.append(z)
    
df = pd.concat([df_ggs,df_ghi,df_ghl],ignore_index=True)
df.rename(columns={"Kategori": "Kategori Pelanggan", "Kategori.1": "Kategori Produk"}, inplace=True)


# In[4]:


#Create function to check missing and unique value percentage
def df_summ(df):
    result = pd.DataFrame()
    
    result['Kolom'] = df.columns
    result['Tipe'] = df.dtypes.values
    result['Missing'] = df.isna().sum().values
    result['Missing (%)'] = result['Missing']*100/len(df)
    result['Unik'] = df.nunique().values
    result['Unik (%)'] = result['Unik']*100/len(df)
    
    return result


# In[5]:


df.head()


# In[6]:


df_summ(df)


# # Data Cleaning

# In[7]:


#Check duplicated rows
df.duplicated().sum()


# In[8]:


#Drop duplicated rows
df = df.drop_duplicates()

#Rechecking..
print("Data yang terduplikasi ada sebanyak: ", df.duplicated().sum(), " baris")


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


# Outlier Boxplots
fig, axes = plt.subplots(1,2, figsize=(24,8))
j = 0
for col in numcol:
    sns.boxplot(x=col, ax=axes[j], data=df)
    sns.despine()
    j += 1
        
fig.suptitle("Outlier Boxplots IQR", fontsize=24)

# Show boxplot
plt.savefig("Outlier Boxplots IQR.jpg")
plt.show()


# In[11]:


df1 = df.copy()


# In[12]:


#Checking skewness score of numeric variable to identify suitable transformation method
#closer to 0 the better

def skor_skew(df,x):
    print("Skor Skewness untuk data", x)
    print("Nilai untuk data asli: ", df[x].skew().round(2))
    print("Nilai setelah di transformasi dengan log: ", np.log(df[x]).skew().round(2))
    print("Nilai setelah di transformasi dengan square root: ", np.sqrt(df[x]).skew().round(2))
    print("Nilai setelah di transformasi dengan box cox: ", pd.Series(stats.boxcox(df[x])[0]).skew().round(2))
    print("Nilai setelah di transformasi dengan cubic root: ", np.cbrt(df[x]).skew().round(2))
    print("")

for i in numcol:
    skor_skew(df1, i)

#Error message means one of the transformation method is not compatible, just ignore it


# In[13]:


#transform the skewed numeric variablewith log transformation for Qty and sqrt transformation for Harga Bruto
df1["Qty Trans"] = np.log(df1["Qty"])
df1["Harga Bruto Trans"] = np.sqrt(df1["Harga Bruto"])
trans = ["Qty Trans", "Harga Bruto Trans"]


# In[14]:


# Outlier Boxplots after transformation
fig, axes = plt.subplots(1,2, figsize=(24,8))
j = 0
for col in trans:
    sns.boxplot(x=col, ax=axes[j], data=df1)
    sns.despine()
    j += 1
        
fig.suptitle("Outlier Boxplots Transformed", fontsize=24)

# Show boxplot
plt.savefig("Outlier Boxplots Trans.jpg")
plt.show()


# In[15]:


#outlier handling
def remove_outlier(df,outcol):
    q1=np.nanquantile(df[outcol],0.25)
    q3=np.nanquantile(df[outcol],0.75)
    iqr=q3-q1
    ll=q1-1.5*iqr
    ul=q3+1.5*iqr
    out=df[(df[outcol]<ll)|(df[outcol]>ul)].index
    return df.drop(index=out)

dfx=remove_outlier(df, "Qty")
dfx=remove_outlier(dfx, "Harga Bruto")
df1=remove_outlier(df1, "Qty Trans")
df1=remove_outlier(df1, "Harga Bruto Trans")

#Outlier handling
print("TANPA TRANSFORMASI")
print("Jumlah baris sebelum menghapus outlier: ", df.shape[0])
print("Jumlah baris setelah menghapus outlier: ", dfx.shape[0])
print("Persentase baris yang dibuang: ", (df.shape[0]-dfx.shape[0])/df.shape[0]*100, "%")
print("")

print("DENGAN TRANSFORMASI")
print("Jumlah baris sebelum menghapus outlier: ", df.shape[0])
print("Jumlah baris setelah menghapus outlier: ", df1.shape[0])
print("Persentase baris yang dibuang: ", (df.shape[0]-df1.shape[0])/df.shape[0]*100, "%")


# In[16]:


#handling missing value
df1=df1.dropna()


# In[17]:


df1.shape


# In[18]:


#recheck...
df_summ(df1)


# In[19]:


df1["Kategori Pelanggan"].unique()


# In[20]:


#replacing typo from Kategori Pelanggan LGM --> LGGN
df1["Kategori Pelanggan"].loc[(df1["Kategori Pelanggan"] == "LGM")] = "LGGN"


# In[21]:


df1["Kategori Pelanggan"].unique()


# In[22]:


df1["Gudang"].unique()


# In[23]:


#replace typo Gudang GS --> GGS
df1["Gudang"].loc[(df1["Gudang"] == "GS")] = "GGS"


# In[24]:


df1["Gudang"].unique()


# In[25]:


df_summ(df1)


# # Data Preprocessing

# In[56]:


df2 = df1.copy()


# In[57]:


#Prepare the RFM table

#total sum
df2["TotalSum"] = df2["Qty"] * df2["Harga Bruto"]

#recency variable
from datetime import timedelta
skrg = max(df2.Tanggal) + timedelta(days=1)

#aggregate the dataset by Kode Pelanggan (The RFM Table)
df2 = df2.groupby(["Kode Pelanggan"]).agg({
    "Tanggal": lambda x: (skrg - x.max()).days,
    "No.Transaksi": "count",
    "TotalSum": "sum"})

# Rename columns
df2.rename(columns = {"Tanggal": "Recency","No.Transaksi": "Frequency","TotalSum": "Monetary Value"}, inplace=True)


# In[58]:


df2.describe()


# In[59]:


df2.sort_values("Frequency", ascending=False).head(7)


# In[60]:


df2.sort_values("Monetary Value", ascending=False).head(7)


# In[61]:


#because customer with Kode Pelanggan = UMUM is the accumulated of non-coded customers (general customers),
#the frequency and Monetary Value gets too big and becoming irrelevant (accumulation of different common customers)
#thus we will drop it
df2 = df2.drop(index="UMUM")


# In[62]:


#check data distribution plot
fig, axes = plt.subplots(1,3, figsize=(24,8))
j = 0
for col in df2.columns:
    sns.distplot(x=df2[col], ax=axes[j])
    sns.despine()
    axes[j].set_xlabel(col)
    j += 1
        
fig.suptitle("Distribution Plot RFM", fontsize=24)

# Show dist plot
plt.savefig("Distplot RFM awal.jpg")
plt.show()


# In[63]:


for i in df2.columns:
    skor_skew(df2, i)


# In[64]:


df3 = df2.copy()

#transformasi variabel Recency dengan metode sqrt
df3["Recency"] = np.sqrt(df3["Recency"])

#transformasi variabel Frequency dengan metode box cox
df3["Frequency"] = stats.boxcox(df3["Frequency"])[0]

#transformasi variabel Monetary Value dengan metode box cox
df3["Monetary Value"] = stats.boxcox(df3["Monetary Value"])[0]

df3.describe()


# In[65]:


#recheck distribution after transform
fig, axes = plt.subplots(1,3, figsize=(24,8))
j = 0
for col in df3.columns:
    sns.distplot(x=df3[col], ax=axes[j])
    sns.despine()
    axes[j].set_xlabel(col)
    j += 1
        
fig.suptitle("Distribution Plot RFM Transformed", fontsize=24)

# Show dist plot
plt.savefig("Distplot RFM Transformed.jpg")
plt.show()


# In[66]:


#Normalize data with standardization
scaler = StandardScaler()
fix = scaler.fit_transform(df3)

#check that mean=0 & std=1, thus it is standard normal distribution
df_fix = pd.DataFrame(fix, columns=["Recency","Frequency","Monetary Value"])
df_fix = df_fix.set_index(df3.index)
df_fix.describe()


# In[67]:


df_fix.head()


# In[68]:


#export dataframe final to excel for visualization and descriptive statistics input
df_fix.to_excel("final.xlsx")


# In[78]:


#export dataframe final to excel for visualization and descriptive statistics input
df2.to_excel("dataset visualisasi.xlsx")


# # Data Modelling

# In[70]:


#determining the cluster number with elbow method
fig, ax = plt.subplots(figsize=(12,8))

#WCSS is the sum of squared distance between each point and the centroid in a cluster
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state = 42)
    kmeans.fit(fix)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

# Visualize elbow method
plt.savefig("Elbow Method.jpg")
plt.show()


# In[71]:


#determining the cluster number with Silhouette method
for i in range(2,11):
    model = KMeans(n_clusters=i, init="k-means++", random_state = 42)
    preds = model.fit_predict(fix)
    centers = model.cluster_centers_

    score = silhouette_score(fix, preds).round(3)
    print("Untuk k={}, silhouette score= {}".format(i, score))


# In[72]:


#first we choose k=3
model = KMeans(n_clusters=3, random_state=42).fit(fix)
labels = model.labels_
df2["Cluster"]=labels
df_fix["Cluster"]=labels


# In[73]:


#show the statistic profile (mean) from each cluster
df2.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary Value':['mean', 'count']}).round(2)


# In[74]:


#Take aways:
#Cluster 0: Highest freq and spending, with the lowest recency (most recent last transactions)
#Cluster 1: Lowest freq and spending, highest recency (oldest last transactions)
#Cluster 2: Intermediate (just between the 0 and 1 cluster)

#Business implications:
#Cluster 0 = Loyal Customer --> give upmost priority and special treatment, avoid churn at every cost
#Cluster 1 = Lost Customer / Bypasser --> They only transact every once in a while, no need to spend resource for this cluster
#Cluster 2 = Opportunity --> Design marketing strategies to increase their value to the supermarket


# In[79]:


#export dataframe to excel for implementation
df2.to_excel("Hasil Clustering.xlsx")


# # Visualisasi Hasil Clustering

# In[75]:


fig = plt.figure(figsize = (16,12))
ax = fig.add_subplot(111, projection='3d')

x=fix
y_clusters=labels

ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 30 , color = 'green', label = "cluster 0")
ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 30 , color = 'red', label = "cluster 1")
ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 30 , color = 'blue', label = "cluster 2")

center = model.cluster_centers_
ax.scatter(center[:,0], center[:,1],center[:,2],marker="o", c="black", s=300)

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary Value')
ax.legend()

plt.savefig("Hasil Clustering.jpg")
plt.show()

