library(matlab)
setwd("/Users/4470246/Projects/PMO/MeasuringFitnessPerClone/data/LiveCellImaging/A01_190513_rawData/SNU-668/NA190509_Plate_13295_sub/byframe")
#setwd('/mnt/ix1/Projects/M027_160330_PMO_cisplatin/data/GastricCancerCLs/LiveCellImaging/SNU-668/NA190509_Plate_13295/byframe')

d=unique(sapply(strsplit(list.files(),"_"),"[[",1))
for(x in d){
  dir.create(paste0("../bywell/",x))
  for(x_ in list.files(pattern = x)){
    f = list.files(x_, full.names = T, pattern="full")
    for(src in f){
      target = paste0("../bywell/",x,filesep,x_,"_",fileparts(src)$name,".TIF")
      file.link(src, target)
    }
  }
}
