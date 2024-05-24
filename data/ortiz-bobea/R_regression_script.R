setwd("environmental_bayesian_modeling/data/ortiz-bobea")

library(lfe)
regdata <- read.csv("data2/regdata_train.csv")
f <- as.formula("fd_log_tfp ~ fd_tmean + fd_tmean_sq + fd_prcp + fd_prcp_sq | ISO3 + year | 0 | ISO3 + block")
reg <- felm(f, regdata, Nboot=1000, keepCX=T, keepX=T, clustervar=as.factor(regdata$block), weights=regdata$weights)
summary(reg)

library(stringmagic)
library(fixest)
regdata <- read.csv("data2/regdata_train.csv")
f <- as.formula("fd_log_tfp ~ fd_tmean + fd_tmean_sq + fd_prcp + fd_prcp_sq | ISO3 + year")
reg <- feols(regdata, f)
summary(reg)