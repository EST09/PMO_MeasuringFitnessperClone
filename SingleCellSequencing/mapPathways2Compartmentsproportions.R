### New Workflow ###

### NB: I think pathway expression in the second loop is given in a random order each time
### locations may not correlate with pathway
# do the first 100 in pathway expression match those first 100 in lpp?

# Attempt 1: Pathways as a proportion of a subcompartment according to expression - different colours for different pathways

### Script ###

options(java.parameters = "-Xmx7g")

#set working directory
setwd("/Users/esthomas/Andor_Rotation/PMO_MeasuringFitnessperClone/SingleCellSequencing")

## Load libraries and devtools ##

#I've cloned git@github.com:noemiandor/Utils.git into ./Projects/code/Rcode/github

library(matlab)
library(GSVA)
library(cloneid)
library(ggplot2)
library(plyr)
library(dplyr)
library(tibble)
library(stringr)
library(scales)
library(rgl)

#custom packages
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/Pathways/getGenesInvolvedIn.R?raw=TRUE")
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/Pathways/getAllPathways.R?raw=TRUE")
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")

#within ./SingleCellSequencing
source("get_compartment_coordinates.R")
source("get_compartment_coordinates_FromAllen.R")
source("alignPathways2Compartments.R")


## Get templates and set paths ##

pathwayMapFile = "NCBI2Reactome_PE_All_Levels_sapiens.txt"
CELLLINE="SNU-668"
ROOTD = tools::file_path_as_absolute("../data")
OUTD=paste0("../results/pathwayCoordinates_3D", filesep, CELLLINE)
ifelse(!dir.exists(OUTD), dir.create(OUTD, recursive = T), FALSE)

# these files are from: https://www.allencell.org - I'm not sure exactly what cell or if they're just an example
# we have asked for nucleus and mito here

# see table in evernote: https://www.evernote.com/shard/s744/sh/3d1c1dcc-8605-485f-ee58-c912dd75971f/8b9465487102f1ea9673e6a5276e10e5
# decide which we are making 
# another cell: choose 5:6 to start playing around with
coord = get_compartment_coordinates_FromAllen(cytosolF=NULL, nucleusF = paste0(ROOTD,filesep,"3Dbrightfield/allencell/D03_FijiOutput/DNA_anothercell.csv"), mitoF = paste0(ROOTD,filesep,"3Dbrightfield/allencell/D03_FijiOutput/Mito_anothercell.csv"));

# #create placeholder movie
# rgl::movie3d(
#   movie="CellCompartmentsIn3D_Placeholder",
#   rgl::spin3d( axis = c(1, 1, 1), rpm = 3),
#   duration = 20,
#   dir = "~/Downloads/",
#   type = "gif",
#   clean = TRUE
# )
# rgl.close()


pqFile = "../data/Clone_0.0878689_ID102953.RObj"

load(pqFile) #Clone_0.0878689_ID102953 saved as Robj

# ## Pathway information:

# #PathwayMap File is the NCBI file - prettifying
path2locmap<-read.table(pathwayMapFile, header = FALSE, sep = "\t", dec = ".", comment.char="", quote="", check.names = F, stringsAsFactors = F)

# Here we replace all compartment names in the path2locmap with their counterpart in coords column names
path2locmap$V3 <- sapply(strsplit(path2locmap$V3, "[", fixed=TRUE), function(x) (x)[2])
path2locmap$V3 <- gsub("]", "", path2locmap$V3)
aliasMap = list("endoplasmic reticulum membrane"="endoplasmic reticulum", "endoplasmic reticulum lumen" = "endoplasmic reticulum", "Golgi membrane" = "Gogli apparatus", "Golgi-associated vesicle lumen" = "Gogli apparatus", "Golgi-associated vesicle membrane" = "Gogli apparatus", "Golgi lumen" = "Gogli apparatus", "trans-Golgi network membrane" = "Gogli apparatus", "nucleoplasm" = "nucleus", "nucleolus" = "nucleus", "nuclear envelope" = "nucleus", "mitochondrial inner membrane" = "mitochondrion", "mitochondrial outer membrane" = "mitochondrion", "mitochondrial intermembrane space" = "mitochondrion", "mitochondrial matrix" = "mitochondrion", "endosome lumen" = "endosome", "endosome membrane" = "endosome", "late endosome lumen" = "endosome", "late endosome membrane" = "endosome", "lysomal lumen" = "lysosome", "lysomal membrane" = "lysosome", "peroxisomal matrix" = "peroxisome", "peroxisomal membrane" = "peroxisome")
for(x in names(aliasMap)){
  path2locmap$V3 = str_replace_all(path2locmap$V3, x, aliasMap[[x]])
}

## Exclude pathways that are active in undefined locations for now. @TODO: map all pathways later
path2locmap = path2locmap[path2locmap$V3 %in% colnames(coord)[apply(coord!=0,2,any)],] #x, y, z, nucleus, mitochonrion
## Exclude pathways that are in endosome or peroxisome: we did not set their coordinates. @TODO later
path2locmap = path2locmap[!path2locmap$V3 %in% c("endosome" ,  "peroxisome"  ),]
## Exclude pathways that are not expressed:
path2locmap = path2locmap[path2locmap$V6 %in% rownames(pq),]
## Rename columns for easier readability
colnames(path2locmap)[c(3,6)]=c("Location","pathwayname")

## locations per pathway:
lpp = sapply(unique(path2locmap$pathwayname), function(x) unique(path2locmap$Location[path2locmap$pathwayname==x]))
lpp = lpp[sample(length(lpp),100)]; ## use only subset for testing
save(file='~/Downloads/tmp_coord.RObj', list=c('coord','OUTD', 'lpp','pq','path2locmap')) #what's this doing?

LOI=c("nucleus","mitochondrion")

for (cellName in colnames(pq)[5]){
  dir.create(paste0(OUTD,filesep,cellName))
  # what do the numbers for expression mean - is it normalised to anything?
  pathwayExpressionPerCell <- pq[,cellName]
  names(pathwayExpressionPerCell) <- rownames(pq)
  
  #get a feel for the expression range of the pathways
  sum_exp = sum(pathwayExpressionPerCell)
  print(sum_exp)
  
  write.csv(coord,"~/Downloads/example_coord.csv", row.names = FALSE)
  
  n <- list()
  m <- list()

  
  # # names(lpp) is a pathway
  for(j in names(lpp)[1:100]){
    
    pmap = cbind(coord, matrix(0,nrow(coord),1)) # append empty matrix
    colnames(pmap)[ncol(pmap)]=j # add pathway to end of pmap
    outImage = paste0(OUTD,filesep,cellName,filesep,gsub(" ","_",gsub("/","-",j,fixed = T)),".gif")
    outTable = paste0(OUTD,filesep,cellName,filesep,gsub(" ","_",gsub("/","-",j,fixed = T)),".txt")

    if(file.exists(outImage)){
      print(paste("Skipping",j,"because image already saved"))
      next;
    }

    P = path2locmap[path2locmap$pathwayname==j,,drop=F] #select the rows which have the pathway of interest

    #rename V8 to Species and keep only columns Location, pathwayname and Species
    colnames(P)[8]="Species"
    P = subset(P, select = c("Location", "pathwayname", "Species"))

    #remove duplicate rows
    P = P[!duplicated(P), ]

    fr =  plyr::count(P$Location)

    #if there are two locations we are dividing the freq by 2, 3 locations diving by 3
    #to evenly distribute coordinates between compartments
    fr$freq = fr$freq/sum(fr$freq)
    rownames(fr) = fr$x
    
    #val_n <- ifelse(grepl("nucleus", fr$x), 1, 0)
    val_n <- str_count(fr$x, "nucleus")
    if (length(val_n) == 2) {
      n <- append(n, 1)
    } else { 
      n <- append(n, val_n)
      }
    #val_m <- ifelse(grepl("mitochondrion", fr$x), 1, 0)
    val_m <- str_count(fr$x, "mitochondrion")
    if (length(val_m) == 2){ 
      m <- append(m, 1) 
    } else { 
      m <- append(m, val_m)
    }
 
    ## Candidates of indices
    idx_Candid = lapply(rownames(fr), function(location) which(coord[,location]==1) ) #rownames(fr) specifies e.g. nucleus
    names(idx_Candid) = rownames(fr) # specify name of index list
    #print(names(idx_Candid))
    #write.csv(fr,"~/Downloads/example_fr.csv", row.names = FALSE)
    #write.csv(idx_Candid,"~/Downloads/example_idx_Candid.csv", row.names = FALSE)
    
    
    # Here we take a random sample of x coordinates for our pathway
    #idx = lapply(names(idx_Candid), function(x) sample(idx_Candid[[x]], pathwayExpressionPerCell[j]*fr[x,"freq"], replace = T)  )
    #names(idx) = names(idx_Candid)
# 
#     # And here we tally how many times each x coordinate appears in the sampling
#     idx = plyr::count(unlist(idx))
# 
#     # In this step we populate our pmap with our randomly selected x coordinate and it's matching y coordiate from the coord object
#     for(i in 1:nrow(idx)){
#       pmap[idx$x[i],j] =  idx$freq[i]
#     }
#     ## Print statement
#     print(paste("Processed pathway",j))
# 
#     # Image of the pathway map
#     pmap_ = pmap[pmap[,j]>0,]
#     write.table(pmap_[,c("x","y","z",j)], file = outTable,sep="\t",quote = F, row.names = F)
#     # Image of the pathway map
#     png(outImage,width = 400, height = 400)
#     image(pmap[j,,],col =rainbow(100),xaxt = "n",yaxt = "n"); #,main=paste(j,cloneid::extractID(cellName))
#     dev.off()
  }
  write.csv(pathwayExpressionPerCell,"~/Downloads/example_pathway_expr.csv")
  write.csv(m,"~/Downloads/mito.csv")
  write.csv(n,"~/Downloads/nucl.csv")
}
