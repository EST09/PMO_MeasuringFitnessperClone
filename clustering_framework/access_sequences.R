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
library(DBI)

#r3dDefaults$windowRect=c(0,50, 800, 800)

#custom packages
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/Pathways/getGenesInvolvedIn.R?raw=TRUE")
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/Pathways/getAllPathways.R?raw=TRUE")
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")

#within ./SingleCellSequencing
source("get_compartment_coordinates.R")
source("get_compartment_coordinates_FromAllen.R")
source("alignPathways2Compartments.R")

#clones = cloneid::getSubclones("NCI-N87", whichP="TranscriptomePerspective")


#Tommy's code for accessing sequences

cellLine="NCI-N87"
gs = getAllPathways(include_genes = T, loadPresaved = T)
gs = gs[sapply(gs, length) >= 5]
chooseCRANmirror()
clones = cloneid::getSubclones(cellLine, whichP = "Identity")
# i2p = cloneid::identity2perspectiveMap(cellLine, persp = "TranscriptomePerspective")
# ## Expression profiles of each clone's cell members
# p = sapply(names(i2p), function(x)
#     cloneid::getSubProfiles(as.numeric(x)))
# membership = sapply(names(p), function(x)
#     rep(x, ncol(p[[x]])))
# membership = do.call("c", membership)
# ex = do.call(cbind, p)