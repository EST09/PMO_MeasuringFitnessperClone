options(java.parameters = "-Xmx7g")
library(matlab)
library(GSVA)
library(cloneid)
library(ggplot2)
library(dplyr)
library(plyr)
library(tibble)
library(stringr)
library(scales)
library(rgl)

## Read Fiji output (3D coordinates of nuclei or mitochondria)
coord=read.csv("~/Projects/PMO/MeasuringFitnessPerClone/data/3Dbrightfield/allencell/F03_FijiOutput/210812_6hr_70nm_630X9_middleRight.csv");
XYCOLS=c("CX..pix.","CY..pix.","CZ..pix.")
colnames(coord)[match(XYCOLS,colnames(coord))]=c("x","y","z")
tmp = quantile(coord$z,c(0,1))
space=tmp[2]-tmp[1]

## Cluster coordinates
# o=dbscan::dbscan(coord[,c("x","y","z")],eps = 2,minPts = 5)
o=kmeans(coord[,c("x","y","z")],centers = 12)
fr=plyr::count(o$cluster)
fr=fr[fr$freq>160,]
head(fr)
ii=which(o$cluster %in% fr$x)

## Visualize
# scatterplot3d::scatterplot3d(coord$x[ii], coord$y[ii], coord$z[ii], pch=20,cex.symbols = 0.1, zlim=c(tmp[1]-space/1.5, tmp[2]+space/1.5),color =o$cluster[ii]+2)
rgl::plot3d(coord$x[ii], coord$y[ii], coord$z[ii], pch=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2), size=0.01, axes=F, xlab="",ylab="", zlab="",col=o$cluster[ii]+2)
colnames(coord)[match(c("x","y","z"),colnames(coord))]=XYCOLS
for(cell in fr$x){
  ii=which(o$cluster==cell);
  write.csv(coord[ii,],file = paste0("~/Downloads/cell_",cell,"_Mito.csv"))
}





