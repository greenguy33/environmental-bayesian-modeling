library(rjags)
library(coda)
library(tidyr)
library(data.table)

burke.data <- read.csv("environmental_bayesian_modeling/data/burke/data/input/GrowthClimateDataset.csv",header=TRUE)
# Omit rows with missing data
# burke.data <- burke.data %>% drop_na(c('growthWDI','UDel_temp_popweight','UDel_precip_popweight'))

# Remove countries with no climate data
countries_missing_climate_data <- c('LIE', 'KIR', 'TON', 'PLW', 'ROU', 'TUV', 'IMY', 'COD', 'KSV', 'FSM', 'DMA', 'SGP', 'MCO', 'SMR', 'ADO', 'TMP', 'MHL', 'HKG', 'MLT', 'MAC', 'KNA', 'GRD', 'BMU', 'LCA', 'BRB', 'MDV', 'CHI', 'TWN', 'ABW', 'WBG', 'SYC', 'ATG', 'BHR', 'MNE')
burke.data <- burke.data[!burke.data$iso %in% countries_missing_climate_data,]

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

scaled_temp <- scale(temp)
scaled_precip <- scale(precip)
scaled_gdp <- scale(gdp)

jags <- jags.model(
  # "environmental_bayesian_modeling/scripts/JAGS/burke_hierarchical_model_covariate_deterministic.txt",
  "environmental_bayesian_modeling/scripts/JAGS/burke_hierarchical_model_covariate_priors.txt",
  data = list(
    'temp_x'=as.numeric(scaled_temp),
    'precip_x'=as.numeric(scaled_precip),
    'gdp_y'=as.numeric(scaled_gdp),
    'n'=data_len,
    'm'=num_countries,
    'o'=num_years,
    'p'=grad_effects_len,
    'countries'=countries_encoded,
    'year'=years_encoded,
    'grad_effects_data'=grad_effects,
    'grad_effects_2_data'=grad_effects_2
  ),
  n.chains=4
)
# Burn-in samples
update(jags, 1000)
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
  n.iter=1000
)

temp_coef <- as.vector(rbind(jags.output.coda[[1]][,5],jags.output.coda[[2]][,5],jags.output.coda[[3]][,5],jags.output.coda[[4]][,5]))
temp_coef_2 <- as.vector(rbind(jags.output.coda[[1]][,6],jags.output.coda[[2]][,6],jags.output.coda[[3]][,6],jags.output.coda[[4]][,6]))
temp_denominator <- sapply(temp_coef_2, function(i){i*-2})
temp_vertex <- temp_coef / temp_denominator
temp_observed_mean <- attributes(scaled_temp)$`scaled:center`
temp_observed_sd <- attributes(scaled_temp)$`scaled:scale`
unscaled_temp_vertex <- (temp_vertex * temp_observed_sd) + temp_observed_mean
print(mean(unscaled_temp_vertex))
print(sd(unscaled_temp_vertex))
coda::gelman.diag(jags.output.coda)
coda::effective_size(jags.output.coda)
plot(jags.output.coda)
coda::gelman.plot(jags.output.coda)