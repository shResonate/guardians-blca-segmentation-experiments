#!/usr/bin/Rscript
packages <- c("sjmisc", "BayesLCA", "glmnet")
install.packages(setdiff(packages, rownames(installed.packages())), dependencies = TRUE, repos = "https://cran.r-project.org")
# lapply(packages, function(pkg) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     install.packages(pkg, dependencies = TRUE, repos = "https://cran.r-project.org")
#   }
# })
library(sjmisc)
library(data.table)
library(BayesLCA)
library(glmnet)

arguments <- commandArgs(trailingOnly = TRUE)
print(arguments)
print('Arguments received.')
inputFilePath <- arguments[1]
outputPath <- arguments[2]
num.segments <- as.numeric(arguments[3])
burn_in <- as.numeric(arguments[4])
thin <- as.numeric(arguments[5])
iterations <- as.numeric(arguments[6])
colsToDrop <- arguments[7]
useCase <- arguments[8]
segmentsOutputFilename <- paste0("blca-segments.csv")

print(inputFilePath)
print('Loading inputs.')
inputData <- read.csv(inputFilePath)
print('input data loaded.')
data_d <- dicho(inputData[, -which(names(inputData) == colsToDrop)], dich.by = "median", as.num = T,  var.label = NULL, val.labels = NULL, append = FALSE,  suffix = "_d")

###  Next run the Bayesian Latent Class Analysis with a preset number of classes (7)
###  We find 5 hidden classes in the binarized data.  The Gibbs sampling is used to sample from the parameters'
###  true distribution.
###  We run 1000 iterations after 100 burn-in.  We thin the samples (take every second) to achieve 'good-mixing'

time_taken <- system.time({
  assign(paste0("blca_", useCase), blca.gibbs(data_d , num.segments, start.vals = "single", burn.in = burn_in, thin = thin, iter = iterations))
})
print(time_taken)

col_IDs <- c(colsToDrop)
data_IDs <- data.table(inputData)[ , ..col_IDs]

print('Name of the model')
print(paste0("blca_",useCase))
#par(mar=c(1,1,1,1))
#plot(paste0("blca_",useCase), which = 3)
#lines(blca_2.aa, which = 3, lwd=rep(3.0,8))

Segments.Num <- apply(get(paste0("blca_", useCase))$Z, 1, which.max)
IDs_wSegment <- cbind(data_IDs, "segment"=Segments.Num)
IDs_wSegment$model = paste0("BLCA_", Sys.Date(), "_", Sys.time())

fwrite(IDs_wSegment, file= paste0(useCase, "_segments_", Sys.Date(), "_", Sys.time(), ".csv"))
# system(try(paste0("aws s3 cp /Users/samhawala/Documents/work2024/Segmentations/finalExperiments/autoExperiment/auto_segments_101824.csv  s3://resonate-datasci-dev/shawala/Segmentations/alcoholExperiment/auto_segments_101824.csv")))
#
save.image(file=paste0(outputPath, paste0(useCase, '_segments.rData')))
