library(matlab)
library(ggplot2)
setwd("~/Projects/PMO/MeasuringFitnessPerClone/code/SingleCellSequencing")
overlayHist<-function(this,ontothat){
  dat=as.data.frame(c(this,ontothat))
  colnames(dat)="data"
  dat$cat="ontothat"
  dat$cat[1:length(this)]="this"
  p=ggplot(dat,aes(x=data,fill=cat)) + 
    geom_histogram(data=dat, alpha = 0.4)
  return(list(dat=dat,p=p))
}
# this=imgStats[,1]; ontothat=sample(imgStats[,2],100)
overlayDistributions1D<-function(this, ontothat,q=c(0.05,0.5,0.95)){
  # dat=overlayHist(this,ontothat)
  qstat<-function(this, ontothat){
    q=sapply(list(this,ontothat), quantile, q)
    colnames(q)=c("this","ontothat")
    q=as.data.frame(q)
    return(q)
  }
  q1=qstat(this, ontothat)
  ## Shift both medians to 0
  this=this-q1$this[1]
  ontothat=ontothat-q1$ontothat[1]
  ## Overlay upper quantile
  q2=qstat(this, ontothat)
  this=this/(q2$this[3]/q2$ontothat[3])
  
  ## ontothat: Median back to orig
  ontothat=ontothat+q1$ontothat[2]
  this=this+q1$ontothat[2]
  dat=overlayHist(this,ontothat)
  return(dat)
}



## read imaging and seq stats
seqStats=read.table("../../data/RNAsequencing/B02_220112_seqStats/NCI-N87/Clone_0.244347_ID119967.txt",sep="\t",check.names = F,stringsAsFactors = F)
imgStats=read.table("../../data/3Dbrightfield/allencell/G06_segmentationStats/0_prediction_c0.model_stats.txt",sep="\t",check.names = F,stringsAsFactors = F)

# plot separately
ggplot(imgStats, aes(x=log(area_mito), y=log(volume_c0.model.p)) ) +
  geom_bin2d(bins = 10) +
  scale_fill_continuous(type = "viridis") +  theme_bw()

imgStats_=as.data.frame(umap::umap(imgStats)$layout)
plot(imgStats_,pch=20)

seqStats_=as.data.frame(umap::umap(seqStats)$layout)
plot(seqStats_,pch=20)


# plot together
par(mfrow=c(2,2))
sapply(colnames(seqStats), function(x) hist(seqStats[,x],xlab=x))
sapply(colnames(imgStats), function(x) hist(imgStats[,x],xlab=x))


## Overlay distributions
##@TODO:  ensure they imgStats have same ncol as seqStats and that they are in desired order (which pair of features match)
colnames(imgStats_)<-colnames(seqStats_)<-c("x","y")
seqStatsMapped=list()
for(i in 1:ncol(imgStats_)){
  overlayHist(seqStats_[,i],imgStats_[,i])$p
  seqStatsMapped[[colnames(imgStats_)[i]]]=overlayDistributions1D(seqStats_[,i],imgStats_[,i], q=c(0.1,0.5,0.9))
}
seqStatsMapped$x$p
ggsave(filename = "~/Downloads/volumeFeature.png",width = 4,height = 3)
## Overwrite seqStats
seqStats_=as.data.frame(sapply(seqStatsMapped,function(x) x$dat$data))
colnames(seqStats_)=colnames(imgStats_)
seqStats_=seqStats_[sample(nrow(seqStats_),size = nrow(imgStats_)),]
## co-cluster image and sequencing stats
imgStats_$type="img"
seqStats_$type="seq"
stats=rbind(imgStats_,seqStats_)
rownames(stats) = paste(rownames(stats), stats$type)
dd = dist(stats[,1:2])
tr = ape::nj(dd)
plot(tr)
col = rep("red", length(tr$tip.label))
col[grep("img",tr$tip.label) ] = "blue"
par(mfrow=c(1,1))
plot(tr,show.tip.label = T, tip.color = col, cex=0.36)
legend("topright",c("sequencing","imaging"),fill=c("red","blue"))
