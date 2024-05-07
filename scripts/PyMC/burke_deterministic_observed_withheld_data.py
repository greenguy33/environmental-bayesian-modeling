import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from sklearn.linear_model import LinearRegression
import csv
from timeit import default_timer as timer

data = pd.read_csv("data/burke-dataset-random-withheld.csv")

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

min_year = min(data.year)
for row_index, row in enumerate(data.itertuples()):
    if row.iso != curr_country:
        country_index += 1
        curr_country = row.iso
    year_index = row.year - min_year
    country_mult_mat[country_index][row_index] = 1
    year_mult_mat[year_index][row_index] = 1

# grad effects scaled back to not overpower the other scaled covariates
grad_effects_data1 = np.transpose(np.array(data.loc[:, data.columns.str.startswith(('_yi'))]))/1000
grad_effects_data2 = np.transpose(np.array(data.loc[:, data.columns.str.startswith(('_y2'))]))/1000000
grad_effects_data = np.concatenate([grad_effects_data1, grad_effects_data2])

# start timer

start = timer()

# construct model and sample

with pm.Model() as model:

    gdp_intercept = pm.Normal('gdp_intercept',0,5)
    temp_gdp_coef = pm.Normal('temp_gdp_coef',0,2)
    temp_sq_gdp_coef = pm.Normal('temp_sq_gdp_coef',0,2)
    precip_gdp_coef = pm.Normal("precip_gdp_coef",0,2)
    precip_sq_gdp_coef = pm.Normal("precip_sq_gdp_coef",0,2)

    year_coefs = pt.expand_dims(pm.Normal("year_coefs", 0, 5, shape=(len(set(data.year))-1)),axis=1)
    year_coefs = pm.math.concatenate([[[0]],year_coefs])
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_coefs*year_mult_mat,axis=0))

    country_coefs = pt.expand_dims(pm.Normal("country_coefs", 0, 5, shape=(len(set(data.iso))-1)),axis=1)
    country_coefs = pm.math.concatenate([[[0]],country_coefs])
    country_fixed_effects = pm.Deterministic("country_fixed_effects",pt.sum(country_coefs*country_mult_mat,axis=0))

    gradual_effect_coefs = pt.expand_dims(pm.Normal("grad_effect_coefs", 0, 5, shape=(len(grad_effects_data))),axis=1)
    gradual_effects = pm.Deterministic("grad_effects",pt.sum(gradual_effect_coefs*grad_effects_data,axis=0))

    gdp_prior = pm.Deterministic(
        "gdp_prior",
        gdp_intercept +
        (temp_scaled * temp_gdp_coef) +
        (temp_sq_gdp_coef * pt.sqr(temp_scaled)) +
        (precip_scaled * precip_gdp_coef) +
        (precip_sq_gdp_coef * pt.sqr(precip_scaled)) +
        year_fixed_effects +
        country_fixed_effects +
        gradual_effects
    )

    gdp_std = pm.HalfNormal('gdp_std', sigma=1)
    gdp_posterior = pm.Normal('gdp_posterior', mu=gdp_prior, sigma=gdp_std, observed=gdp_scaled)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

# save model to file

with open ('models/PyMC/burke-reproduction-mcmc-fixed-effects-scaled-back-grad-effects-deterministic-observed-withheld.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior,
        "precip_scaler":precip_scaler,
        "temp_scaler":temp_scaler,
        "gdp_scaler":gdp_scaler
    },buff)

# end timer

end = timer()

print("Runtime:")
print(end - start)