assignCompartment2Nucleus<-function(other_center_coord, nuc_center_coord){
  ZSTACK_DISTANCE=0.29
  other_coord=read.csv(file=other_center_coord,check.names = F,stringsAsFactors = F)
  nuc_coord=read.csv(file=nuc_center_coord,check.names = F,stringsAsFactors = F)
  other_coord$z=other_coord$z*ZSTACK_DISTANCE
  nuc_coord$z=nuc_coord$z*ZSTACK_DISTANCE
  
  ##@TODO: should not be necessary!
  other_center_coord=strsplit(other_center_coord,"_Cells_")[[1]][1]
  nuc_center_coord=strsplit(nuc_center_coord,"_Cells_")[[1]][1]
  
  d=flexclust::dist2(other_coord, nuc_coord)
  ##Assign mitochondria to closest nucleus
  i_nuc=apply(d,1,which.min)
  ## Aggregate mitochondria all_cells_coordinates for each nucleus 
  ## and name output file according to nucleus ID
  for(i in unique(i_nuc)){
    i_mito=which(i==i_nuc)
    dm=c()
    for(j in i_mito){
      f_mito=list.files("../All_Cells_coordinates",pattern =paste0(other_center_coord,"_cell_",j,"_"),full.names = T )
      ## @TODO: should not be necessary
      if(isempty(f_mito)){
        next
      }
      coord=read.csv(file=f_mito,check.names = F,stringsAsFactors = F)
      coord[,other_center_coord]=j
      dm=rbind(dm,coord)
    }
    #### WRITE LINKED COMPARTMENTS:
    ## Copy other compartment file (e.g. nucleus)
    f_nuc=list.files("../All_Cells_coordinates",pattern =paste0(nuc_center_coord,"_cell_",i,"_"),full.names = T )
    ## @TODO: should not be necessary
    if(isempty(f_nuc) || is.null(dm)){
      next
    }
    file.copy(f_nuc,paste0(OUTD,nuc_center_coord,"_cell_",i,"_coordinates.csv"))
    ## Also write mitochondria file
    write.csv(dm,paste0(OUTD,other_center_coord,"_cell_",i,"_coordinates.csv"), row.names = F, quote = F)
  }
}

setwd("~/Projects/PMO/MeasuringFitnessPerClone/data/GastricCancerCL/3Dbrightfield/NCI-N87/I04_PostProcessCellposeOutput/Cells_center_coordinates")
OUTD="../../I05_multiOrganelles_Linked/"
nuc_center_coord="nucleus.p_Cells_Centers.csv";
other_center_coord="mito.p_Cells_Centers.csv"
assignCompartment2Nucleus(other_center_coord, nuc_center_coord)
other_center_coord="cytoplasm.p_Cells_Centers.csv"
assignCompartment2Nucleus(other_center_coord, nuc_center_coord)

f=list.files(OUTD,full.names = T)
signals_per_id=plyr::count(sapply(strsplit(f,"_"), function(x) x[length(x)-1]))
## keep only cells with all three signals:
toRM=signals_per_id$x[signals_per_id$freq<3]
for(x in toRM){
  y=list.files(OUTD,full.names = T,pattern = paste0("_",x,"_"))
  file.remove(y)
}
