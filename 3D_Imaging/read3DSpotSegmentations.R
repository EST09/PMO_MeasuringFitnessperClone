setwd("~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCL/3Dbrightfield/NCI-N87/")
library(matlab)
library(geometry)
library(misc3d)
library(rgl)
IN="H05_multiOrganelles_Linked"
# IN="H03_CellposeOutput/All_Cells_coordinates"
OUT="H06_segmentationStats"
# OUT="H04_segmentationStats"
r3dDefaults$windowRect=c(0,50, 800, 800) 
ZSTACK_DISTANCE=1;#0.29
f=list.files(IN,full.names = T)
signals_per_id=plyr::count(sapply(strsplit(f,"_"), function(x) x[length(x)-1]))
## keep only cells with all three signals:
signals_per_id=signals_per_id[signals_per_id$freq==3,]

## Read in data
coord=c();
for(cell in signals_per_id$x){
  for(x in sapply(c("nucleus.p_cell_","mito.p_cell_","cytoplasm.p_cell_"), paste0,cell,"_coordinates.csv")){
    a=read.csv(paste0(IN,filesep,x))
    if(!"mito.p" %in% colnames(a)){
      a$mito.p=NA
    }
    if(!"cytoplasm.p" %in% colnames(a)){
      a$cytoplasm.p=NA
    }
    id=strsplit(fileparts(x)$name,"_")[[1]]
    a$organelle = id[3] 
    id=id[length(id)-1]
    a$id=id
    coord=rbind(coord,a)
  }
}
coord$id=as.numeric(coord$id)
# apply(coord,2,quantile,c(0,1))
coord$z=coord$z*ZSTACK_DISTANCE

## Backup
coord_=coord
tmp = quantile(coord_$z,c(0,1))
space=tmp[2]-tmp[1]
col=rainbow(length(unique(coord_$id)))
names(col)=as.character(unique(coord_$id))
rgl::plot3d(coord_$x, coord_$y, coord_$z, pch3d=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2), size=2, axes=F, xlab="",ylab="", zlab="",col=1+(coord$organelle=="mito"))
rgl::plot3d(coord_$x, coord_$y, coord_$z, pch=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2), size=2, axes=F, xlab="",ylab="", zlab="",col=col[as.character(coord_$id)])
# rgl::plot3d(coord_$x, coord_$y, coord_$z, pch=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2),  axes=F, xlab="",ylab="", zlab="",col=col[as.character(coord_$id)], type="s",radius=1+(coord$organelle=="mito"))


## Reassign cell ID based on cluster membership
o=dbscan::dbscan(coord_[,c("x","y","z")],eps = 2.5,minPts = 3)
coord_$id=o$cluster
fr=plyr::count(o$cluster)
fr=fr[fr$freq>200,]
## remove coordinates classified as noise
coord_=coord_[coord_$id!=0 & coord_$id %in% fr$x,]
## Gather stats
cells=unique(coord_$id)
organelles=unique(coord_$organelle)
imgStats=as.data.frame(matrix(NA,length(cells),1+3*length(organelles)))
rownames(imgStats)=as.character(cells)
colnames(imgStats)=c("MitoCount",sapply(c("volume_","area_","pixel_"),paste0,organelles))
for(id in cells){
  a=coord_[coord_$id==id,]
  imgStats[as.character(id),"MitoCount"]=length(unique(a$mito.p))-1
  for(organelle in organelles){
    a_=a[a$organelle==organelle,]
    hull <- convhulln(a_[,1:3], options = "FA")
    imgStats[as.character(id),paste0(c("volume_","area_","pixel_"),organelle)]=c(hull$vol,hull$area,nrow(a_))
  }
}
## Add more stats and save 
imgStats$pixel_per_mito_avg=imgStats$pixel_mito/imgStats$MitoCount
imgStats$pixel_per_volume_mito=imgStats$pixel_mito/imgStats$volume_mito
for(organelle in organelles){
  imgStats[,paste0("pixel_per_volume_",organelle)]=imgStats[,paste0("pixel_",organelle)]/imgStats[,paste0("volume_",organelle)]
}
write.table(imgStats,file=paste0(OUT,filesep,"0_prediction_c0.model_stats.txt"),sep="\t")
## Remove cells with very small volumes -- likely FPs
realCells=as.numeric(rownames(imgStats)[imgStats$volume>=8^3 & imgStats$volume<=20^3])
coord_=coord_[coord_$id %in% realCells,]

###########
## Plots ##
## plot clusters of cells
o=kmeans(coord_[,c("x","y","z")],centers = 20)
rgl::close3d()
rgl::plot3d(coord_$x, coord_$y, coord_$z, pch=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2), size=2, axes=F, xlab="",ylab="", zlab="",col=o$cluster+1)
plot(1);legend("topleft",as.character(unique(o$cluster)),fill = unique(o$cluster)+1)

## plot concave hull for cells within one cell cluster
fr=plyr::count(o$cluster)
fr=fr[order(fr$freq),]
rgl::close3d()
a=coord_[o$cluster==19,]
cells=unique(a$id)
rgl::plot3d(a$x, a$y, a$z, pch=20, size=2, axes=F, xlab="",ylab="", zlab="",col=sapply(a$id,match,cells), alpha=0.5,add = F)
rgl::close3d()
for(id in cells){
  a=coord_[coord_$id==id,]
  hull=Plot_ConcaveHull(a[,1], a[,2], a[,3], lcolor =which(cells==id), alpha=0.5,add = T)
  # rgl::plot3d(a$x, a$y, a$z, pch=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2), size=2, axes=F, xlab="",ylab="", zlab="",col=which(cells==id),add=T)
}
rgl::movie3d(
  movie="CellPose3D_output",
  rgl::spin3d( axis = c(1, 1, 1), rpm = 8),
  duration = 10,
  dir = "~/Downloads/",
  type = "gif",
  clean = TRUE
)