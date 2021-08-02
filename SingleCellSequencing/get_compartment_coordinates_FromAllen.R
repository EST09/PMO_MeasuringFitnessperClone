# Dependancies
library(RColorBrewer)


get_compartment_coordinates_FromAllen <-function(cytosolF="~/Downloads/fijitestout2.csv", mitoF="~/Downloads/fijitestout2.csv", nucleusF="~/Downloads/fijitestout3.csv"){
  
  # I'm not sure what the columns are in these cells or how we got the 
  XYZCOLS = c("CX..pix.", "CY..pix.", "CZ..pix.")
  COMPCOLS = c("cytosol", "endoplasmic reticulum", "Golgi apparatus", "nucleus", "mitochondrion", "endosome", "lysosome", "peroxisome")
  dummy =as.data.frame(matrix(0,1,length(COMPCOLS)))
  colnames(dummy) = COMPCOLS  
  dummy[rep(seq_len(nrow(dummy)), each = 2), ]
  
  # add nucleus data
  nucl=read.csv(nucleusF)
  nucl = cbind(nucl[,XYZCOLS], dummy[rep(seq_len(nrow(dummy)), each = nrow(nucl)), ])
  nucl$nucleus = 1;
  coord = nucl

  # add mito data if not null
  if(!is.null(mitoF)){
    mito=read.csv(mitoF)
    mito = cbind(mito[,XYZCOLS], dummy[rep(seq_len(nrow(dummy)), each = nrow(mito)), ])
    mito$`mitochondrion`=1
    coord = rbind(coord, mito);#, cyto)
  }
  colnames(coord)[1:3] = c("x","y","z")
  
  ## draw cell to check coordinate assignment 
  colormap = brewer.pal(length(COMPCOLS),"Paired")
  names(colormap) = COMPCOLS
  tmp = quantile(coord$z,c(0,1))
  space=tmp[2]-tmp[1]

  scatterplot3d::scatterplot3d(coord$x, coord$y, coord$z, pch=20,cex.symbols = 0.1, zlim=c(tmp[1]-space/1.5, tmp[2]+space/1.5))
  rgl::plot3d(coord$x, coord$y, coord$z, pch=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2), size=0.01, axes=F, xlab="",ylab="", zlab="")
  
  for(o in names(colormap)){
    coord_ = coord[coord[,o]==1,]
    rgl::points3d(coord_$x, coord_$y, coord_$z,col=colormap[o],add=T, size=0.01)
  }
  
  rgl::legend3d("topleft", names(colormap), fill=colormap, bty='n',cex=1.7)
  
  return(coord)
}