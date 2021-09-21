#Question-1
wi <- read.csv("/Users/jobinsamuel/Desktop/Assignments/PCA/Datasets_PCA/wine.csv") #read data
attach(wi)#attach data
summary(wi)
nrd<- scale(wi)
summary(nrd)
pw <- princomp(nrd, cor = TRUE, scores = TRUE, covmat = NULL)#princomp  is  used for principle component analysis where only numeric data is passed and here we are only calculating correlation and scores.
str(pw) #here the loadings which we see is nothing but weights
summary(pw)

pw$loadings

pw$scores

plot(pw) #It shows the plot of pc's
#biplot(pw) shows a graph with respect to pc1 and pc2

plot(cumsum(pw$sdev * pw$sdev) * 100 / (sum(pw$sdev * pw$sdev)), type = "b") #Plot showing how the variance is increasing

pw$scores[, 1:3] #considering only first three pc's
fpw <- cbind(wi[,1],pw$scores[,1:3])#Columnbinding the three pc's with type column of wi
fpw

# Distance matrix
d <- dist(fpw, method = "euclidean") 
hclst <- hclust(d, method = "complete")

#Plotting dendrogram
plot(hclst,hang = -1)

treecut <- cutree(hclst, k = 4) # Cut tree into 4 clusters

rect.hclust(hclst, k = 4, border = "red")

wine <- NULL  #Creating an empty variable 

for (i in 2:12) {                #Using for loop 
  wine <- c(wine, kmeans(wi, centers = i)$tot.withinss) #The totalwithiness should be stored in the variable
}
wine
#Scree Plot
plot(2:12, wine, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
#In this plot should check for an elbow shape

west <- kmeans(wi, 4) #The number at which the elbow is present and also the clusters which has minimum total withiness and maximum betweenness
str(west) #Here we look at the total withiness and betweeness
lst <- data.frame(west$cluster, wi) #Cluster is appended to the data frame

aggregate(wi[, 2:12], by = list(west$cluster), FUN = mean) #Finally aggregating


#Question-2
ha <- read.csv("/Users/jobinsamuel/Desktop/Assignments/PCA/Datasets_PCA/heart disease.csv")#read data
attach(ha)#attach data
nrmd <- scale(ha)
summary(nrmd)

phc <- princomp(nrmd, cor = TRUE, scores = TRUE, covmat = NULL)#princomp  is  used for principle component analysis where only numeric data is passed and here we are only calculating correlation and scores.
str(phc) #here the loadings which we see is nothing but weights
summary(phc)

loadings(phc)

phc$scores
plot(phc)#It shows the plot of pc's

plot(cumsum(phc$sdev * phc$sdev) * 100 / (sum(phc$sdev * phc$sdev)), type = "b")#Plot showing how the variance is increasing
phc$scores
#considering only first three pc's
phc$scores[, 1:3]
fphc <- cbind(ha[,1],phc$scores[,1:3])#Columnbinding the three pc's with type column of wi
fphc
summary(fphc)


# Distance matrix
dismtr <- dist(fphc, method = "euclidean") 
hahclst <- hclust(dismtr, method = "complete")

#Plotting dendrogram
plot(hahclst,hang = -1)

treecut <- cutree(hahclst, k = 9) # Cut tree into 9 clusters

rect.hclust(hahclst, k = 9, border = "purple")


heart <- NULL  #Creating an empty variable 

for (i in 2:10) {                #Using for loop 
  heart <- c(heart, kmeans(ha, centers = i)$tot.withinss) #The totalwithiness should be stored in the variable
}
heart
#Scree Plot
plot(2:10, heart, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
#In this plot should check for an elbow shape

hdis <- kmeans(ha, 3) #The number at which the elbow is present and also the clusters which has minimum total withiness and maximum betweenness
str(hdis) #Here we look at the total withiness and betweeness
fhdis <- data.frame(hdis$cluster, ha) #Cluster is appended to the data frame

aggregate(ha[, 2:12], by = list(hdis$cluster), FUN = mean) #Finally aggregating
