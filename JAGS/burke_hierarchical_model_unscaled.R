library(rjags)
library(coda)
library(tidyr)
library(data.table)

burke.data <- read.csv("environmental_bayesian_modeling/data/burke/data/input/GrowthClimateDataset.csv",header=TRUE)
burke.data <- burke.data %>% drop_na(c('growthWDI','UDel_temp_popweight','UDel_precip_popweight'))

# TODO: remove countries with no climate data
# TODO: add gradual effects

countries <- burke.data[,1]
years <- burke.data[,2]
gdp <- burke.data[,11]
temp <- burke.data[,28]
precip <- burke.data[,29]


grad_effects <- transpose(burke.data[, grep(pattern ='X_yi_*', names(burke.data))])
grad_effects_2 <- transpose(burke.data[, grep(pattern ='X_y2_*', names(burke.data))])

data_len <- dim(burke.data)[1]
num_countries <- length(unique(countries))
num_years <- length(unique(years))
grad_effects_len <- dim(grad_effects)[1]

countries_encoded <- as.integer(factor(countries))
years_encoded = as.integer(factor(years))

jags <- jags.model(
  "environmental_bayesian_modeling/JAGS/burke_hierarchical_model_covariate_deterministic.txt",
  data = list(
    'temp_x'=temp,
    'precip_x'=precip,
    'gdp_y'=gdp,
    'n'=data_len,
    'm'=num_countries,
    'o'=num_years,
    'p'=grad_effects_len,
    'countries'=countries_encoded,
    'year'=years_encoded,
    'grad_effects_data'=grad_effects,
    'grad_effects_2_data'=grad_effects_2
    )
)

jags.output.coda <- coda.samples(
  jags,
  c(
    'temp_gdp_coef',
    'precip_gdp_coef',
    'temp_gdp_coef_2',
    'precip_gdp_coef_2',
    'regression_intercept',
    'global_gdp_sigma'
  ),
  10000
)

temp_coef <- jags.output.coda[[1]][,5]
temp_coef_2 <- jags.output.coda[[1]][,6]
temp_denominator <- sapply(temp_coef_2, function(i){i*-2})
temp_vertex <- temp_coef / temp_denominator
print(mean(temp_vertex))
print(sd(temp_vertex))
