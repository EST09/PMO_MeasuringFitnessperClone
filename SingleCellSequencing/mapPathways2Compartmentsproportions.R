### New Workflow ###

# Attempt 1: Pathways as a proportion of a subcompartment according to expression - different colours for different pathways

### Workflow ###

# 1. Get compartment coordinates from allen cell segmentation - combine organelle coordinates together into coords
# 2. Create a placeholder image
# 3. Run cloneid on SNU-668 cell line to get subpopulations
#       - CloneID is a database. Can learn more here: 10.1093/nargab/lqaa016
#       - SNU-668 is contained within CloneID
# 4. Move forward working with the 9th subpopulation, create a file and quantify pathway expression based on GSVA (pq)
#       - where colnames are the 430 cells in the subpop
#       - rownames gives 116 pathways (I think from running cloneid?) 
#       - getSubProfiles gets copy number and expression profile of each cell
#       - GSVA compares to all cells and within cell: https://doi.org/10.1186/1471-2105-14-7
# 5. Get the cell cycle State for each cell
# 6. Create a pathtolocmap - take NCBI file and generalise to Allen cell compartments
#       - we preprocessed the NCBI file from Reactome to ensure that locations are more standardised
# 7. From pathtolocmap create a list of unique pathway names - lpp (656 pathways)
#       - This different from the 1116 pathways in the 9th subpopulation (pq) as we are only considering mito and nucleus 
# 8. For each cell in 9th subpop, for each pathway in lpp find how many compartemnts it appears in in path2locmap
#       - We should be dividing by 2 if in 2 compartments, by 3 if 3 compartments (naive approach as not necessarily evenly distributed)
# 9. Get indices which are in pathway location e.g. nucleus
# 10. Sample the same number of coordinates as that pathways expression e.g. for RNA polymerase, expr=16096, sample 16096 coordinates
#       - Same as pathway expression? We should explore alternatives to this to make visualisation easier.
# 11. Because we allow resampling record how many times a coord has been sampled - append to the coord like table from allen cell
# 12. Create an image using these Coordinates

### Script ###

options(java.parameters = "-Xmx7g")

#set working directory  
setwd("./SingleCellSequencing")

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

#r3dDefaults$windowRect=c(0,50, 800, 800) 

#custom packages
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/Pathways/getGenesInvolvedIn.R?raw=TRUE")
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/Pathways/getAllPathways.R?raw=TRUE")
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")

#within ./SingleCellSequencing
source("get_compartment_coordinates.R")
source("get_compartment_coordinates_FromAllen.R")
source("alignPathways2Compartments.R")

## Get templates and set paths ##

# https://reactome.org/what-is-reactome
# indicates which external protein, gene or small molecule identifiers in the source database were mapped to Reactome pathway and reaction annotations
# Columns are: 
# 1. Source database identifier: NCBI Gene ID
# 2. Reactome Pathway Stable identifier
# Extra column for common name
# Extra column for URL ending
# 3. URL
# 4. Event (Pathway or Reaction) Name
# 5. Evidence Code
# 6. Species
pathwayMapFile = "NCBI2Reactome_PE_All_Levels_sapiens.txt"
CELLLINE="SNU-668"
ROOTD = tools::file_path_as_absolute("../data")
OUTD=paste0("../results/pathwayCoordinates_3D", filesep, CELLLINE)
ifelse(!dir.exists(OUTD), dir.create(OUTD, recursive = T), FALSE)

# these files are from: https://www.allencell.org - I'm not sure exactly what cell or if they're just an example 
# we have asked for nucleus and mito here
coord = get_compartment_coordinates_FromAllen(cytosolF=NULL, nucleusF = paste0(ROOTD,filesep,"3Dbrightfield/allencell/D03_FijiOutput/DNA_anothercell.csv"), mitoF = paste0(ROOTD,filesep,"3Dbrightfield/allencell/D03_FijiOutput/Mito_anothercell.csv"));

#create placeholder movie
rgl::movie3d(
  movie="CellCompartmentsIn3D_Placeholder", 
  rgl::spin3d( axis = c(1, 1, 1), rpm = 3),
  duration = 20, 
  dir = "~/Downloads/",
  type = "gif", 
  clean = TRUE
)
rgl.close()

## Expression profiles of all detected genes for clone 9 in the cell line.

# CLONEID is a framework that integrates measurements obtained from different technologies, 
# or from multi-spatial or longitudinal biopsies, into a comprehensive approximation of the identities of coexisting tumor clones. 
# The framework comes with a SQL database that keeps track of clones over multiple spatially or temporally connected sequencing experiments.
# The database also links these high-throughput measurements to the growth conditions of cells from which they were obtained. 
# A main goal of CLONEID is to facilitate tracking the pedigree of evolving cell lines over decades along with potentially changing 
# cell culture habits.
# This can reveal long-term trends in the clonal evolution of cell lines, that would have remained elusive at smaller time-scales.

# clone 9: not sure what that is? 
cID = 9

# getSubclones: display subclones of a clone 
# Need to ask how cloneid works exactly
# clones = cloneid::getSubclones(CELLLINE, whichP="TranscriptomePerspective")
# pqFile = paste0(OUTD,filesep,names(clones)[cID],".RObj")

load(pqFile) #Clone_0.0878689_ID102953 saved as Robj

# if(file.exists(pqFile)){
#   load(pqFile)

# }else{
#   p = getSubProfiles(as.numeric(cloneid::extractID(names(clones)[cID])))
#   # Exclude copy number profile (keep only expression profile)
#   require(DBI)
#   ex = p[-grep(":", rownames(p)),]
#   # Now let\'s quantify pathway expression based on this expression
#   gs=getAllPathways(include_genes=T, loadPresaved = T);     
#   gs=gs[sapply(gs, length)>=5]
#   pq <- gsva(ex, gs, kcdf="Poisson", mx.diff=T, verbose=FALSE, parallel.sz=2, min.sz=10)
#   pq <- rescale(pq, to = c(0,30000))
#   save(pq, file=paste0(OUTD,filesep,names(clones)[cID],".RObj"))
# }

#colnames(pq) gives the subclones of that clone? - presumably of clone 9
#getState gets the cell cycle state
ccState = sapply(colnames(pqFile), function(x) cloneid::getState(x, whichP = "TranscriptomePerspective"))


# ## Pathway information:

# #PathwayMap File is the NCBI file - prettifying
# path2locmap<-read.table(pathwayMapFile, header = FALSE, sep = "\t", dec = ".", comment.char="", quote="", check.names = F, stringsAsFactors = F)

# # Here we replace all compartment names in the path2locmap with their counterpart in coords column names
# path2locmap$V3 <- sapply(strsplit(path2locmap$V3, "[", fixed=TRUE), function(x) (x)[2])
# path2locmap$V3 <- gsub("]", "", path2locmap$V3) 
# aliasMap = list("endoplasmic reticulum membrane"="endoplasmic reticulum", "endoplasmic reticulum lumen" = "endoplasmic reticulum", "Golgi membrane" = "Gogli apparatus", "Golgi-associated vesicle lumen" = "Gogli apparatus", "Golgi-associated vesicle membrane" = "Gogli apparatus", "Golgi lumen" = "Gogli apparatus", "trans-Golgi network membrane" = "Gogli apparatus", "nucleoplasm" = "nucleus", "nucleolus" = "nucleus", "nuclear envelope" = "nucleus", "mitochondrial inner membrane" = "mitochondrion", "mitochondrial outer membrane" = "mitochondrion", "mitochondrial intermembrane space" = "mitochondrion", "mitochondrial matrix" = "mitochondrion", "endosome lumen" = "endosome", "endosome membrane" = "endosome", "late endosome lumen" = "endosome", "late endosome membrane" = "endosome", "lysomal lumen" = "lysosome", "lysomal membrane" = "lysosome", "peroxisomal matrix" = "peroxisome", "peroxisomal membrane" = "peroxisome")
# for(x in names(aliasMap)){
#   path2locmap$V3 = str_replace_all(path2locmap$V3, x, aliasMap[[x]])
# }
# ## Exclude pathways that are active in undefined locations for now. @TODO: map all pathways later
# path2locmap = path2locmap[path2locmap$V3 %in% colnames(coord)[apply(coord!=0,2,any)],] #x, y, z, nucleus, mitochonrion
# ## Exclude pathways that are in endosome or peroxisome: we did not set their coordinates. @TODO later
# path2locmap = path2locmap[!path2locmap$V3 %in% c("endosome" ,  "peroxisome"  ),]
# ## Exclude pathways that are not expressed:
# path2locmap = path2locmap[path2locmap$V6 %in% rownames(pq),]
# ## Rename columns for easier readability
# colnames(path2locmap)[c(3,6)]=c("Location","pathwayname")

# ## locations per pathway:
# lpp = sapply(unique(path2locmap$pathwayname), function(x) unique(path2locmap$Location[path2locmap$pathwayname==x]))
# lpp = lpp[sample(length(lpp),100)]; ## use only subset for testing
# save(file='~/Downloads/tmp_coord.RObj', list=c('coord','OUTD', 'lpp','pq','path2locmap')) #what's this doing?

# ## Calculate 3D pathway activity maps


# LOI=c("nucleus","mitochondrion")
# # so are the clones individual cells - that would make sense
# # why are we only considering 100 here?
# for (cellName in colnames(pq)[1:100]){ 
#   dir.create(paste0(OUTD,filesep,cellName))
#   # what do the numbers for expression mean - is it normalised to anything?
#   pathwayExpressionPerCell <- pq[,cellName]
#   names(pathwayExpressionPerCell) <- rownames(pq) 
  
#   ## The first pathway/location pair we're looking at

#   # names(lpp) is a pathway
#   for(j in names(lpp)){ 
#     pmap = cbind(coord, matrix(0,nrow(coord),1)) # append empty matrix
#     colnames(pmap)[ncol(pmap)]=j # add pathway to end of pmap
#     outImage = paste0(OUTD,filesep,cellName,filesep,gsub(" ","_",gsub("/","-",j,fixed = T)),".gif")
#     outTable = paste0(OUTD,filesep,cellName,filesep,gsub(" ","_",gsub("/","-",j,fixed = T)),".txt")
#     if(file.exists(outImage)){
#       print(paste("Skipping",j,"because image already saved"))
#       next;
#     }
#     P = path2locmap[path2locmap$pathwayname==j,,drop=F] #select the rows which have the pathway of interest 
#     #rename V8 to Species and keep only columns Location, pathwayname and Species
#     colnames(P)[8]="Species"
#     P = subset(P, select = c("Location", "pathwayname", "Species"))
#     #remove duplicate rows
#     P = P[!duplicated(P), ]

#     fr =  plyr::count(P$Location)
#     #if there are two locations we are dividing the freq by 2, 3 locations diving by 3
#     #to evenly distribute coordinates between compartments
#     fr$freq = fr$freq/sum(fr$freq) 
#     rownames(fr) = fr$x
    
#     ## Candidates of indices
#     idx_Candid = lapply(rownames(fr), function(location) which(coord[,location]==1) ) #rownames(fr) specifies e.g. nucleus
#     names(idx_Candid) = rownames(fr) # specify name of index list

#     # Here we take a random sample of x coordinates for our pathway 
#     idx = lapply(names(idx_Candid), function(x) sample(idx_Candid[[x]], pathwayExpressionPerCell[j]*fr[x,"freq"], replace = T)  ) # why are we multiplying by freq? # what is this doing?
#     names(idx) = names(idx_Candid)

#     # And here we tally how many times each x coordinate appears in the sampling
#     idx = plyr::count(unlist(idx))
    
#     # In this step we populate our pmap with our randomly selected x coordinate and it's matching y coordiate from the coord object
#     for(i in 1:nrow(idx)){      
#       pmap[idx$x[i],j] =  idx$freq[i]
#     }
#     ## Print statement
#     print(paste("Processed pathway",j))
    
#     # Image of the pathway map 
#     pmap_ = pmap[pmap[,j]>0,]
#     write.table(pmap_[,c("x","y","z",j)], file = outTable,sep="\t",quote = F, row.names = F)
#     # Image of the pathway map 
#     png(outImage,width = 400, height = 400)
#     image(pmap[j,,],col =rainbow(100),xaxt = "n",yaxt = "n"); #,main=paste(j,cloneid::extractID(cellName))
#     dev.off()
#  }
# }

# # ## Animation 3D pathway activity maps
# load('~/Downloads/tmp_coord.RObj')
# detach('package:GSVA', unload=TRUE)
# for (cellName in list.dirs(OUTD, full.names = F)){
#   print(cellName)
#   for(outTable in list.files(paste0(OUTD,filesep,cellName), pattern = ".txt", full.names = T )){
#     library(rgl)
#     library(magick)
#     outImage = gsub(".txt",".gif",outTable)
#     print(outImage)
#     if(!file.exists(outImage)){
#       pmap_ = read.table(file = outTable,sep="\t", header = T, check.names = F, stringsAsFactors = F)
      
#       r3dDefaults$windowRect = c(0,0,700,700)
#       rgl::material3d(alpha = 0.1)
#       rgl::plot3d(x=pmap_$x, y=pmap_$y, z=pmap_$z,add=F, size=4.91, col=pmap_[,ncol(pmap_)], xlim=quantile(coord$x,c(0,1)), ylim=quantile(coord$y,c(0,1)), zlim=quantile(coord$z,c(0,1)), axes=F, xlab="",ylab="", zlab="", alpha=0.2)
#       Sys.sleep(5)
#       try(rgl::movie3d(
#         movie=matlab::fileparts(outImage)$name, 
#         rgl::spin3d( axis = c(1, 1, 1), rpm = 12),
#         duration = 4, 
#         dir = matlab::fileparts(outImage)$path,
#         type = "gif", 
#         clean = F
#       ))
#       rgl::rgl.close()
      
# #       ##################
# #       #### clean up ####
# #       ##################
#       detach('package:rgl', unload=TRUE)
#       library(crosstalk)
#       library(manipulateWidget)
#       library(miniUI)
#       library(shiny)
#       library(shinythemes)
#       detach('package:crosstalk', unload=TRUE)
#       detach('package:manipulateWidget', unload=TRUE)
#       detach('package:miniUI', unload=TRUE)
#       detach('package:shinythemes', unload=TRUE)
#       detach('package:shiny', unload=TRUE)
#       detach("package:magick", unload = TRUE)
#       ## Remove cache generated by magick package
#       f=list.files('/tmp/Rtmp2quoCU', pattern='magick', full.names=T); 
#       if(length(f)>50){
#         for (x in f){ 
#           file.remove(x)
#         }
#       }
#     }
#   }
# }

