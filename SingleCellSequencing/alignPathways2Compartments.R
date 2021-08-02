# aligns Vae Autoencoder Output to Allen compartment model Output
alignPathways2Compartments <-function(cmprt, pthw){
  Plot_ConcaveHull <- function(xx, yy, zz, lcolor="black", alpha=0.4, add=T, level=0.5/length(xx)) {
    library(MASS) 
    ##Remove outliers
    hQ=0.975; lQ=0.025
    iK1=which(xx<=quantile(xx,hQ) & xx>=quantile(xx,lQ))
    iK2=which(yy<=quantile(yy,hQ) & yy>=quantile(yy,lQ))
    iK3=which(zz<=quantile(zz,hQ) & zz>=quantile(zz,lQ))
    iK=intersect(iK1,iK2)
    iK=intersect(iK,iK3)
    xx=xx[iK]; yy=yy[iK]; zz = zz[iK]
    
    ##Contour
    dens2 <- kde3d(xx, yy, zz, lims=c(min(xx)-sd(xx), max(xx)+sd(xx),   
                                      min(yy)-sd(yy), max(yy)+sd(yy),   
                                      min(zz)-sd(zz), max(zz)+sd(zz) ),n=55  )
    misc3d::contour3d(dens2$d, level=level, dens2$x, dens2$y, dens2$z, color=lcolor, add=add, alpha=alpha); #,drawlabels=F,lwd=2 
    # return(cbind(dens2$x,dens2$y, dens2$z))
  }
  
  shiftmin2zero <- function(a){
    a = apply(a, 2, function(x) x - median(x))
    return(a)
  }
  align <- function(b, to, q=0.95){
    r = sapply(1:ncol(to), function(i) quantile(to[,i],q)/max(b[,i]))
    b = sweep(b,2,r,FUN="*")
    return(b)
  }
  
  NDIM=3
  n=100
  
  ## Random pathways and compartments coordinates
  # cmprt = sapply(1:NDIM, function(i) rnorm(n, mean=8))
  # pthw = sapply(1:NDIM, function(i) rnorm(n, mean=5, sd = 2))
  mm = apply(rbind(cmprt,pthw),2,quantile,c(0,1))
  rgl::plot3d(cmprt[,1],cmprt[,2],cmprt[,3],col="gray", pch=20, xlim=mm[,1], ylim=mm[,2], zlim=mm[,3])
  rgl::points3d(pthw[,1],pthw[,2],pthw[,3],pch=20,col="blue") 
  
  
  ## Align
  cmprt_ = shiftmin2zero(cmprt)
  pthw_ = shiftmin2zero(pthw)
  pthw_ = align(pthw_, to=cmprt_, q=0.98)
  ## Shift back
  shiftback = apply(cmprt, 2, median)
  cmprt = sweep(cmprt_,2,shiftback,FUN="+")
  pthw = sweep(pthw_,2,shiftback,FUN="+")
  
  ## Visualize
  mm = apply(rbind(cmprt,pthw),2,quantile,c(0.02,0.98))
  rgl::rgl.close()
  # @TODO: color-code should be specific to pathway
  col = rainbow(nrow(pthw))
  rgl::plot3d(pthw[,1],pthw[,2],pthw[,3],xlim=mm[,1], ylim=mm[,2], zlim=mm[,3], add=F, size=8, col=col)
  Sys.sleep(5)
  Plot_ConcaveHull(cmprt[,1], cmprt[,2], cmprt[,3], lcolor ="gray", alpha=0.075)
  Plot_ConcaveHull(pthw[,1], pthw[,2], pthw[,3], lcolor ="blue", alpha=0.12, level = 0.1/nrow(pthw))
  ii = seq(1,nrow(pthw), by=10)
  legend("topright", c("Cell compartment", paste("Pathway",1:nrow(pthw)))[ii], fill=c("gray", col)[ii],cex=1.86)
  
  ## Save visualziation
  rgl::movie3d(
    movie="Allen2VAE", 
    rgl::spin3d( axis = c(0, 0, 1), rpm = 2),
    duration = 5, 
    dir = "~/Downloads/",
    type = "gif", 
    clean = TRUE
  )
  rgl.close()
  
}