import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from timeit import default_timer as timer

data = pd.read_csv("data/burke-dataset.csv")

# remove all data or year for countries where one of the 3 variables is entirely missing

country_temp_data = {}
country_precip_data = {}
country_gdp_data = {}

for row in data.iterrows():
    country = row[1].iso
    if country not in country_temp_data:
        country_temp_data[country] = []
    if country not in country_precip_data:
        country_precip_data[country] = []
    if country not in country_gdp_data:
        country_gdp_data[country] = []
    country_temp_data[country].append(row[1].UDel_temp_popweight)
    country_precip_data[country].append(row[1].UDel_precip_popweight)
    country_gdp_data[country].append(row[1].growthWDI)

countries_missing_temp = [country for country in country_temp_data if all(np.isnan(country_temp_data[country]))]
countries_missing_precip = [country for country in country_precip_data if all(np.isnan(country_precip_data[country]))]
countries_missing_gdp = [country for country in country_gdp_data if all(np.isnan(country_gdp_data[country]))]

countries_to_remove = set(countries_missing_temp + countries_missing_precip + countries_missing_gdp)

indices_to_drop = []
for index, row in enumerate(data.itertuples()):
    if row.iso in countries_to_remove:
        indices_to_drop.append(index)

data_len_before = len(data)
data = data.drop(indices_to_drop)
data = data.reset_index()
print(f"Removed {data_len_before - len(data)} rows for completely missing country and/or year data.")

# data scaling

precip_scaler, gdp_scaler, temp_scaler = StandardScaler(), StandardScaler(), StandardScaler()
precip_scaled = precip_scaler.fit_transform(np.array(data.UDel_precip_popweight).reshape(-1,1)).flatten()
gdp_scaled = gdp_scaler.fit_transform(np.array(data.growthWDI).reshape(-1,1)).flatten()
temp_scaled = temp_scaler.fit_transform(np.array(data.UDel_temp_popweight).reshape(-1,1)).flatten()

# year and country fixed effect coefficient matrices

data_len = len(data.year)
year_mult_mat = [np.zeros(data_len) for year in set(data.year)]
country_mult_mat = [np.zeros(data_len) for country in set(data.iso)]
country_index = -1
curr_country = ""

for row_index, row in enumerate(data.itertuples()):
   if row.iso != curr_country:
       country_index += 1
       curr_country = row.iso
   year_index = row.year - min(data.year)
   country_mult_mat[country_index][row_index] = 1
   year_mult_mat[year_index][row_index] = 1

grad_effects_data = np.transpose(np.array(data.loc[:, data.columns.str.startswith(('_y'))]))

# construct model and sample

with pm.Model() as model:

    global_temp_mean_prior = pm.Normal("global_temp_mean_prior",0,1)
    global_temp_sd_prior = pm.HalfNormal("global_temp_mean_sd",1)
    country_coefs_temp_prior = pt.expand_dims(pm.Normal("country_coefs_temp_prior", global_temp_mean_prior, global_temp_sd_prior, shape=(len(set(data.iso)))),axis=1)
    country_temp_priors = pm.Deterministic("temp_prior",pt.sum(country_coefs_temp_prior*country_mult_mat,axis=0))

    global_precip_mean_prior = pm.Normal("global_precip_mean_prior",0,1)
    global_precip_sd_prior = pm.HalfNormal("global_precip_mean_sd",1)
    country_coefs_precip_prior = pt.expand_dims(pm.Normal("country_coefs_precip_prior", global_precip_mean_prior, global_precip_sd_prior, shape=(len(set(data.iso)))),axis=1)
    country_precip_priors = pm.Deterministic("precip_prior",pt.sum(country_coefs_precip_prior*country_mult_mat,axis=0))
    
    # add country-specific prior to NaN values in observed data
    mixed_temp_prior = pm.Deterministic("mixed_temp_prior", pt.switch(
        [1 if np.isnan(val) else 0 for val in data.UDel_temp_popweight],
        country_temp_priors,
        temp_scaled)
    )
    mixed_precip_prior = pm.Deterministic("mixed_precip_prior", pt.switch(
        [1 if np.isnan(val) else 0 for val in data.UDel_precip_popweight],
        country_precip_priors,
        precip_scaled)
    )

    gdp_intercept = pm.Normal('gdp_intercept',1,2)
    temp_gdp_coef = pm.Normal('temp_gdp_coef',-.5,.5)
    temp_sq_gdp_coef = pm.Normal('temp_sq_gdp_coef',-.5,.5)
    precip_gdp_coef = pm.Normal("precip_gdp_coef",.05,.2)
    precip_sq_gdp_coef = pm.Normal("precip_sq_gdp_coef",-.05,.1)

    year_coefs = pt.expand_dims(pm.Normal("year_coefs", -.1, 2, shape=(len(set(data.year)))),axis=1)
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_coefs*year_mult_mat,axis=0))

    country_coefs = pt.expand_dims(pm.Normal("country_coefs", .1, 5, shape=(len(set(data.iso)))),axis=1)
    country_fixed_effects = pm.Deterministic("country_fixed_effects",pt.sum(country_coefs*country_mult_mat,axis=0))

    gradual_effect_coefs = pt.expand_dims(pm.Normal("grad_effect_coefs", -.1, 5, shape=(len(grad_effects_data))),axis=1)
    gradual_effects = pm.Deterministic("grad_effects",pt.sum(gradual_effect_coefs*grad_effects_data,axis=0))

    gdp_prior = pm.Deterministic(
        "gdp_prior",
        gdp_intercept +
        (mixed_temp_prior * temp_gdp_coef) +
        (temp_sq_gdp_coef * pt.sqr(mixed_temp_prior)) +
        (mixed_precip_prior * precip_gdp_coef) +
        (precip_sq_gdp_coef * pt.sqr(mixed_precip_prior)) +
        year_fixed_effects +
        country_fixed_effects +
        gradual_effects
    )

    gdp_std = pm.HalfNormal('gdp_std', sigma=.1)
    gdp_posterior = pm.Normal('gdp_posterior', mu=gdp_prior, sigma=gdp_std, observed=gdp_scaled)
    
    start = timer()

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    end = timer()

    print("Runtime:")
    print(end - start)

# save model to file

with open ('models/burke-reproduction-mcmc-fixed-effects-grad-effects-deterministic-observed-missing-rows-imputed-hierarchical-country-priors.pkl', 'wb') as buff:
  pkl.dump({
      "prior":prior,
      "trace":trace,
      "posterior":posterior,
      "precip_scaler":precip_scaler,
      "temp_scaler":temp_scaler,
      "gdp_scaler":gdp_scaler
  },buff)
