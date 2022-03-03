devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")

setwd("~/Projects/PMO/MeasuringFitnessPerClone/data/LiveCellImaging/A03_190516_Tracking_ilastik")
wells_96=matrix(NA, 8, 12)
rownames(wells_96)= c("A","B","C","D","E","F","G","H")
colnames(wells_96)=c(paste0("0",1:9),10:12)
MAXCELLSPERWELL = 120
MINCELLSPERWELL = 60

wells_96 = wells_96[1:3,1:3]; ##Subset

pdf(file = "~/Downloads/countePerWell.pdf", width = 6, height = 6)
par(mgp = c(2, 1, 0),mfrow=c(nrow(wells_96), ncol(wells_96)), mai = c(0.4, 0.4, 0.3, 0.1))
for(i in rownames(wells_96)){
  for(j in colnames(wells_96)){
    dm=read.table(paste0(i,j,"_CellTracks_CSV-Table.csv"),sep=",", header = T, stringsAsFactors = F, check.names = F)
    dm=dm[dm$trackId!=-1,]
    cellcnt=sapply(unique(dm$frame), function(x) length(unique(dm$trackId[dm$frame==x])))
    names(cellcnt)=paste0(1:12,"h")
    wells_96[i,j] = cellcnt[length(cellcnt)]/cellcnt[1]
    cntA = plyr::count(dm$frame)
    rownames(cntA) = as.character(cntA$x)
    
    ## Cell cycle classification
    dm$CC = "G0G1"
    dm$CC[dm$Size>250] = "S"
    dm$CC[dm$Size>500] = "G2"
    dm$CC[dm$Size>1000] = "M"
    
    ## Dying cells classification
    cntD = grpstats(as.matrix(dm$frame), dm$lineageId, statscols = "max")$max
    # Last time cell was "seen" was prior to last frame:
    frame_lastAlive = cntD[cntD<max(dm$frame)]
    
    ## Visualize Time evolution: cycling cells
    cntG2 = grpstats(as.matrix(dm$CC=="G2"), dm$frame, statscols = "numel+")$`numel+`
    cntG2 = cntG2/cntA[rownames(cntG2),]$freq
    plot(as.numeric(rownames(cntG2)), 100*cntG2, xlab="hour", ylab = "%G2 cells")
    
    ## Visualize Time evolution: dying cells
    if(length(frame_lastAlive)>0){
      cntD = plyr::count(c(frame_lastAlive,unique(dm$frame)))
      rownames(cntD) = as.character(cntD$x)
      cntD$freq = cntD$freq - 1
      cntD$freq = cntD$freq/cntA[rownames(cntD),]$freq
      # plot(cntD$x, 100*cntD$freq, xlab="hour", ylab = "% Dying cells")
    }
    
    # plot(cellcnt, xlab="hour", ylab="cell count", ylim=c(MINCELLSPERWELL,MAXCELLSPERWELL), log="y", pch = 20, cex=2)
    title(paste("GR =",round(wells_96[i,j],2)), line=0.5)
  }
}
dev.off()


## Feature selection
dm$Size=dm$Object_Area_0*dm$Skewness_of_Intensity_0
coi = c(2:30)
ii = which(dm$lineageId %in% coi)
plot(dm$frame[ii],dm$Size[ii])
for(i in coi){
  ii=which(dm$lineageId==i)
  points(dm$frame[ii],dm$Size[ii], col=i, pch=20)
}
