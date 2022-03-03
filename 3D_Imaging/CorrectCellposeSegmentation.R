setwd("~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCL/3Dbrightfield/NCI-N87/")
library(matlab)
library(geometry)
library(misc3d)
library(rgl)
r3dDefaults$windowRect=c(0,50, 800, 800) 
ZSTACK_DISTANCE=1;#0.29
INDIR="H03_CellposeOutput"
OUTDIR="I04_PostProcessCellposeOutput"

############################################################
## Correct segmentation: merge ids belonging to same cell ##
coord_=read.csv(paste0("./",INDIR, "/Cells_center_coordinates/nucleus.p_Cells_Centers.csv"))
o=dbscan::dbscan(coord_[,c("x","y","z")],eps = 2.5,minPts = 3)
coord_$id=o$cluster
fr=plyr::count(o$cluster)
colnames(fr)[1]="newCellID"
fr=fr[order(fr$freq,decreasing = T),]
fr=fr[fr$newCellID!=0,]; ##remove noise

fr$datapoints=NA
newCellCoord=list()
for(newCell in fr$newCellID){
  ##Gather all coordinates associated with new cell
  oldCells=which(newCell==coord_$id)
  tmp=c()
  for(oldid in oldCells){
    dm_=read.csv(paste0("./",INDIR, "/All_Cells_coordinates/nucleus.p_cell_",oldid,"_coordinates.csv"))
    dm_$oldid=oldid
    tmp=rbind(tmp,dm_)
  }
  newCellCoord[[as.character(newCell)]]=tmp
  ## Record # of coordinates:
  fr$datapoints[fr$newCellID==newCell]=nrow(tmp)
}
centroids=t(sapply(newCellCoord, function(x) apply(x[,c("x","y","z")],2,mean)))
colnames(centroids)=c("x","y","z")
fr=fr[order(fr$datapoints),]
hist(fr$freq)
rgl::close3d()
## Plot coordinates for cells of interest
coi=fr$newCellID[seq(1,nrow(fr),by=10)]
for(cell in coi){
  a=newCellCoord[[as.character(cell)]]
  hull=Plot_ConcaveHull(a$x, a$y, a$z, lcolor =which(cell==coi), alpha=0.5,add = T)
}
## Save output
tmp=sapply(1:length(newCellCoord), function(id) write.csv(newCellCoord[[id]],file=paste0("./",OUTDIR, "/All_Cells_coordinates/nucleus.p_cell_",id,"_coordinates.csv")))
write.csv(centroids,file=paste0("./",OUTDIR, "/Cells_center_coordinates/nucleus.p_Cells_Centers.csv"),quote = F, row.names = F)

## Also copy mitochondria and cytoplasm:
for(other in c("mito","cyto")){
  for(res in c("Cells_center_coordinates","All_Cells_coordinates")){
    sapply(list.files(paste0(INDIR,"/",res,"/"),pattern = other,full.names = T), function(x) file.copy(x,paste0(OUTDIR, "/",res,"/",fileparts(x)$name,".csv")))
  }
}