#Importing necessary packages 
import pandas as pd 
import numpy as np


from sklearn.decomposition import PCA   #For performing PCA
import matplotlib.pyplot as plt   
from sklearn.preprocessing import scale 

from scipy.cluster.hierarchy import linkage #TO GET A DENDROGRAM WE USE THESE FUNCTIONS
import scipy.cluster.hierarchy as sch 
###Question -1
wi = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/PCA/Datasets_PCA/wine.csv")#read data
wi.describe() #Summary of data


winorm = scale(wi)#Normalizing data
winorm

pcwi = PCA(n_components = 14) #Number of pc's
pcwi_values = pcwi.fit_transform(winorm)

var = pcwi.explained_variance_ratio_#Variance in each PCA
var


pcwi.components_ #Weights
pcwi.components_[0:3] #Chossing first 3 pc's

var1 = np.cumsum(np.round(var, decimals = 4) * 100)#Cumulative variance
var1

plt.plot(var1, color = "red") #Variance plot for pc's

# PCA scores
pcwi_values

pca_data = pd.DataFrame(pcwi_values) #converting into data frame
pca_data.columns = "comp1", "comp2", "comp3", "comp4", "comp5", "comp6","comp7","comp8","comp9","comp10","comp11","comp12","comp13","comp14"#Assigning column names
f = pd.concat([wi.Type, pca_data.iloc[:, 0:3]], axis = 1) #combine type column with three pc's

# for creating dendrogram 


z = linkage(f, method = "complete")  

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Balance');plt.ylabel('Bonus_miles')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()  #Plot is show since it consists of 4000 data points it takes time to get the output of dendrogram

# Now applying AgglomerativeClustering 
from sklearn.cluster import AgglomerativeClustering

#choosing 8 clusters from the above dendrogram

complete = AgglomerativeClustering(n_clusters = 8, linkage = 'complete', affinity = "euclidean").fit(f) 
complete.labels_

wi_label = pd.Series(complete.labels_) #Using labels

wi['cluster'] = wi_label

wi1 = wi.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]] #Rearragnging the columns so that cluster comes first
wi1.head()

#K means
from sklearn.cluster import	KMeans 
wn = []  #creating empty variable

e = list(range(2, 5))

for i in e:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(f)
    wn.append(kmeans.inertia_)
    
wn

# Scree plot 

plt.plot(e, wn, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model = KMeans(n_clusters = 3) #Choosing a point where an elbow is formed in the plot to form clusters

model.fit(f)

model.labels_ # getting the labels of clusters assigned to each row 

wine = pd.Series(model.labels_)  # converting array into pandas series object 

wi['clust'] = wine 

wi.head()

f.head()

wi.iloc[:, 1:14].groupby(wi.clust).mean()   #Using groupby function


####Question -2
ha = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/PCA/Datasets_PCA/heart disease.csv") #read data
ha.describe() #Summary of data

ha_norm = scale(ha) #Normalizing data
ha_norm

pcha = PCA(n_components = 14) #Choosing number of components
pcha_values = pcha.fit_transform(ha_norm)

vari = pcha.explained_variance_ratio_ #Variance in each pc
vari

pcha.components_ #Weights

pcha.components_[0:3]#Chossing first 3 pc's

vari1 = np.cumsum(np.round(vari, decimals = 4) * 100)#Cumulative variance
vari1

plt.plot(vari1, color = "blue") #Variance plot for pc's

pca_dataha = pd.DataFrame(pcha_values)#Converting into dataframes
pca_dataha.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7","comp8","comp9","comp10","comp11","comp12","comp13"#Assigning column names
f2 = pd.concat([ha.age, pca_dataha.iloc[:, 0:3]], axis = 1)#combine age column with first three pc's

# for creating dendrogram 
w = linkage(f2, method = "complete")  

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Balance');plt.ylabel('Bonus_miles')
sch.dendrogram(w, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()  #Plot

# Now applying AgglomerativeClustering 
from sklearn.cluster import AgglomerativeClustering

complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(ha_norm) 
complete.labels_

ha_label = pd.Series(complete.labels_) #Using labels

ha['cluster'] = ha_label

ha1 = ha.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]] #Rearragnging the columns so that cluster comes first
ha1.head()

#K means
from sklearn.cluster import	KMeans 
ht = []  #creating empty variable

h = list(range(2, 5))

for i in h:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(f2)
    ht.append(kmeans.inertia_)
    
ht

# Scree plot 

plt.plot(h, ht, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model = KMeans(n_clusters = 3)

model.fit(f2)

model.labels_ # getting the labels of clusters assigned to each row 

heart = pd.Series(model.labels_)  # converting array into pandas series object 

ha['clust'] = heart 
ha.head()

f2.head()

ha.iloc[:, 0:14].groupby(ha.clust).mean()   #Using groupby function
