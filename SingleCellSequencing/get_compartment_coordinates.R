get_Compartment_coordinates <-function(resolution = 300){
  compartment_coordinates = matrix(0, (resolution^2)/2,8+2)
  colnames(compartment_coordinates) = c("x","y","cytosol", "endoplasmic reticulum", "Golgi apparatus", "nucleus", "mitochondrion", "endosome", "lysosome", "peroxisome")
  compartment_coordinates = as.data.frame(compartment_coordinates)
  compartment_coordinates$y <- sample.int(resolution, nrow(compartment_coordinates),replace =T)
  compartment_coordinates$x <- sample.int(resolution, nrow(compartment_coordinates), replace =T)
  
  #############################
  ## Deal with Compartments: ##
  getCircularCoordinates <- function(start_X, start_Y, r){
    idx = ((compartment_coordinates$x - start_X) * (compartment_coordinates$x - start_X) + (compartment_coordinates$y - start_Y) * (compartment_coordinates$y - start_Y)) <= r * r
    return(idx)
  }
  getMultipleCircularCoordinates <- function(howMany, radius){
    tmp = compartment_coordinates[sample(rownames(compartment_coordinates)[idx_cell], howMany),c("x","y")]
    idx_mitoch = apply(tmp, 1, function(xy) which(getCircularCoordinates(xy[1], xy[2], radius)))
    idx_mitoch = unlist(idx_mitoch)
    return(idx_mitoch)
  }
  
  ## Define compartment coordinates
  maxXY = max(compartment_coordinates$x)
  idx_cell = getCircularCoordinates(maxXY*(1/2), maxXY*(1/2), maxXY*(1/2))
  idx_nucleus = getCircularCoordinates(maxXY*(1/2), maxXY*(1/2), maxXY*0.25)
  idx_endreticulum = getCircularCoordinates(maxXY*(1/4), maxXY*(1/4), maxXY*0.1)
  idx_golgi = getCircularCoordinates(maxXY*(1/2), maxXY*(5/6), maxXY*0.075)
  idx_mitoch=getMultipleCircularCoordinates(10, maxXY*0.03)
  idx_lysosom = getMultipleCircularCoordinates(20, maxXY*0.02)
  
  ## draw cell to check coordinate assignment
  library(RColorBrewer)
  colormap = brewer.pal(6,"Paired")
  names(colormap) = c("cytosol","nucleus","reticulum","golgi","mitochondrion","lysosom")
  plot(compartment_coordinates[idx_cell,]$x, compartment_coordinates[idx_cell,]$y, pch=20, col=colormap["cytosol"],xaxt="n", xlab="", yaxt="n", ylab="")
  points(compartment_coordinates[idx_nucleus,]$x, compartment_coordinates[idx_nucleus,]$y, pch=20, col=colormap["nucleus"])
  points(compartment_coordinates[idx_endreticulum,]$x, compartment_coordinates[idx_endreticulum,]$y, pch=20, col=colormap["reticulum"])
  points(compartment_coordinates[idx_golgi,]$x, compartment_coordinates[idx_golgi,]$y, pch=20, col=colormap["golgi"])
  points(compartment_coordinates[idx_mitoch,]$x, compartment_coordinates[idx_mitoch,]$y, pch=20, col=colormap["mitochondrion"])
  points(compartment_coordinates[idx_lysosom,]$x, compartment_coordinates[idx_lysosom,]$y, pch=20, col=colormap["lysosom"])
  legend("topleft", names(colormap), fill=colormap, bty='n',cex=0.7)
  
  ## assign boolean values
  compartment_coordinates$nucleus[which(idx_nucleus)] = 1 
  compartment_coordinates$`endoplasmic reticulum`[which(idx_endreticulum)] = 1
  compartment_coordinates$`Golgi apparatus`[which(idx_golgi)] = 1
  compartment_coordinates$mitochondrion[idx_mitoch] = 1
  compartment_coordinates$lysosome[idx_lysosom] = 1
  idx_notUsed = apply(compartment_coordinates[idx_cell,-(1:2)]==0,1,all)
  compartment_coordinates$cytosol[which(idx_cell)][which(idx_notUsed)] = 1
  save("compartment_coordinates", file="~/Downloads/compartment_coordinates.RObj")  
  return(compartment_coordinates)
}