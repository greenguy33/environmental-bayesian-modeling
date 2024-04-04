import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl

data = pd.read_csv("data/burke-dataset.csv")

# drop missing data
indices_to_drop = []
no_nan_cols = ["UDel_temp_popweight","UDel_precip_popweight","growthWDI"]
for index, row in enumerate(data.itertuples()):
    if any(np.isnan(getattr(row,col)) for col in no_nan_cols):
        indices_to_drop.append(index)
data = data.drop(indices_to_drop)

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
    year_index = row.year - 1961
    country_mult_mat[country_index][row_index] = 1
    year_mult_mat[year_index][row_index] = 1

grad_effects_data = np.transpose(np.array(data.loc[:, data.columns.str.startswith(('_y'))]))

# construct model and sample

with pm.Model() as model:
    
    temp_prior = pm.Normal("temp_prior", 0, 1)
    temp_std = pm.HalfNormal("temp_std", 10)
    temp_posterior = pm.Normal("temp_posterior", temp_prior, temp_std, observed=temp_scaled)

    precip_year_coefs = pt.expand_dims(pm.Normal("precip_year_coefs", 0, 5, shape=(len(set(data.year)))),axis=1)
    precip_year_fixed_effects = pm.Deterministic("precip_year_fixed_effects",pt.sum(precip_year_coefs*year_mult_mat,axis=0))

    precip_country_coefs = pt.expand_dims(pm.Normal("precip_country_coefs", 0, 5, shape=(len(set(data.iso)))),axis=1)
    precip_country_fixed_effects = pm.Deterministic("precip_country_fixed_effects",pt.sum(precip_country_coefs*country_mult_mat,axis=0))

    precip_intercept = pm.Normal('precip_intercept',0,10)
    temp_precip_coef = pm.Normal('temp_precip_coef',0,10)
    temp_sq_precip_coef = pm.Normal('temp_sq_precip_coef',0,10)

    precip_prior = pm.Deterministic(
        "precip_prior", 
        precip_intercept + 
        (temp_posterior * temp_precip_coef) +
        (pt.sqr(temp_posterior) * temp_sq_precip_coef) +
        precip_year_fixed_effects +
        precip_country_fixed_effects
    )
    precip_std = pm.HalfNormal("precip_std", 10)
    precip_posterior = pm.Normal("precip_posterior", precip_prior, precip_std, observed=precip_scaled)

    gdp_intercept = pm.Normal('gdp_intercept',0,10)
    temp_gdp_coef = pm.Normal('temp_gdp_coef',0,10)
    temp_sq_gdp_coef = pm.Normal('temp_sq_gdp_coef',0,10)
    precip_gdp_coef = pm.Normal("precip_gdp_coef",0,10)
    precip_sq_gdp_coef = pm.Normal("precip_sq_gdp_coef",0,10)

    year_coefs = pt.expand_dims(pm.Normal("year_coefs", 0, 5, shape=(len(set(data.year)))),axis=1)
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_coefs*year_mult_mat,axis=0))

    country_coefs = pt.expand_dims(pm.Normal("country_coefs", 0, 5, shape=(len(set(data.iso)))),axis=1)
    country_fixed_effects = pm.Deterministic("country_fixed_effects",pt.sum(country_coefs*country_mult_mat,axis=0))

    gradual_effect_coefs = pt.expand_dims(pm.Normal("grad_effect_coefs", 0, 5, shape=(len(grad_effects_data))),axis=1)
    gradual_effects = pm.Deterministic("grad_effects",pt.sum(gradual_effect_coefs*grad_effects_data,axis=0))

    gdp_prior = pm.Deterministic(
        "gdp_prior",
        gdp_intercept +
        (temp_posterior * temp_gdp_coef) +
        (temp_sq_gdp_coef * pt.sqr(temp_posterior)) +
        (precip_posterior * precip_gdp_coef) +
        (precip_sq_gdp_coef * pt.sqr(precip_posterior)) +
        year_fixed_effects +
        country_fixed_effects +
        gradual_effects
    )

    gdp_std = pm.HalfNormal('gdp_std', sigma=10)
    gdp_posterior = pm.Normal('gdp_posterior', mu=gdp_prior, sigma=gdp_std, observed=gdp_scaled)

    prior = pm.sample_prior_predictive()
    trace = pm.sample()
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

# save model to file

with open ('models/burke-reproduction-mcmc-fixed-effects-grad-effects-missing-rows-omitted-precip-dep.pkl', 'wb') as buff:
	pkl.dump({
		"prior":prior,
		"trace":trace,
		"posterior":posterior,
		"precip_scaler":precip_scaler,
		"temp_scaler":temp_scaler,
		"gdp_scaler":gdp_scaler
	},buff)
