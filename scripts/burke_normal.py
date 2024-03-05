import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from sklearn.linear_model import LinearRegression
import csv

data = pd.read_csv("data/burke-dataset.csv")

# remove missing data

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

    # country_coefs_temp_prior = pt.expand_dims(pm.Normal("country_coefs_temp_prior", 0, 1, shape=(len(set(data.iso)))),axis=1)
    # temp_prior = pm.Deterministic("temp_prior",pt.sum(country_coefs_temp_prior*country_mult_mat,axis=0))
    temp_prior = pm.Normal("temp_prior", 0, 1)
    temp_std = pm.HalfNormal("temp_std", 10)
    temp_posterior = pm.Normal("temp_posterior", temp_prior, temp_std, observed=temp_scaled)

    # country_coefs_precip_prior = pt.expand_dims(pm.Normal("country_coefs_precip_prior", 0, 1, shape=(len(set(data.iso)))),axis=1)
    # precip_prior = pm.Deterministic("precip_prior",pt.sum(country_coefs_precip_prior*country_mult_mat,axis=0))
    precip_prior = pm.Normal("precip_prior", 0, 1)
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

# get coefficient multipliers using samples for unscaling coefficients

precip_data = precip_scaler.inverse_transform(np.array(posterior.posterior_predictive.precip_posterior.data.flatten()).reshape(-1,1))
gdp_data = gdp_scaler.inverse_transform(np.array(posterior.posterior_predictive.gdp_posterior.data.flatten()).reshape(-1,1))
temp_data = temp_scaler.inverse_transform(np.array(posterior.posterior_predictive.temp_posterior.data.flatten()).reshape(-1,1))
X = np.column_stack((
    temp_data,
    np.square(temp_data),
    precip_data,
    np.square(precip_data)
))
linreg = LinearRegression().fit(X,gdp_data)
print(linreg.coef_)
print(linreg.intercept_)

# use temperature coefficients to get model threshold

og_coef1 = linreg.coef_[0][0]
og_coef2 = linreg.coef_[0][1]
coef1 = trace.posterior.temp_gdp_coef.data.flatten()
coef2 = trace.posterior.temp_sq_gdp_coef.data.flatten()
mult1 = og_coef1/np.mean(coef1)
mult2 = og_coef2/np.mean(coef2)
numerator = [val*mult1 for val in coef1]
denominator = [-2*val*mult2 for val in coef2]
res = np.array(numerator) / np.array(denominator)
print("temp threshold:", np.mean(res), "lower bound:", np.mean(res) - np.std(res), "upper bound:", np.mean(res) + np.std(res))

# save model to file

with open ('models/burke-reproduction-mcmc-fixed-effects-grad-effects-missing-rows-omitted.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior,
        "precip_scaler":precip_scaler,
        "temp_scaler":temp_scaler,
        "gdp_scaler":gdp_scaler
    },buff)
