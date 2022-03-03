
get_compartment_coordinates_FromAllen <-function(cytosolF="~/Downloads/fijitestout2.csv", mitoF="~/Downloads/fijitestout2.csv", nucleusF="~/Downloads/fijitestout3.csv",XYZCOLS = c("CX..pix.", "CY..pix.", "CZ..pix."), size=0.01){

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
    coord = rbind(coord, mito);
  }
  if(!is.null(cytosolF)){
    cyto=read.csv(cytosolF)
    cyto = cbind(cyto[,XYZCOLS], dummy[rep(seq_len(nrow(dummy)), each = nrow(cyto)), ])
    cyto$cytosol = 1;
    coord = rbind(coord, cyto)
  }
  colnames(coord)[1:3] = c("x","y","z")
  
  ## draw cell to check coordinate assignment 
  colormap = brewer.pal(length(COMPCOLS),"Paired")
  names(colormap) = COMPCOLS
  tmp = quantile(coord$z,c(0,1))
  space=tmp[2]-tmp[1]

  scatterplot3d::scatterplot3d(coord$x, coord$y, coord$z, pch=20,cex.symbols = 0.1, zlim=c(tmp[1]-space/1.5, tmp[2]+space/1.5))

  rgl::plot3d(coord$x, coord$y, coord$z, pch=20, zlim=c(tmp[1]-space/2, tmp[2]+space/2), size=size, axes=F, xlab="",ylab="", zlab="")

  for(o in names(colormap)){
    print(o)
    coord_ = coord[coord[,o]==1,]
    rgl::points3d(coord_$x, coord_$y, coord_$z,col=colormap[o],add=T, size=size)
    print(nrow(coord_))
  }
  
  rgl::legend3d("topleft", names(colormap), fill=colormap, bty='n',cex=1.7)
  
  return(coord)
}