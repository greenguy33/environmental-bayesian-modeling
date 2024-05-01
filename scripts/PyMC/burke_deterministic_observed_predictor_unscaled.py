import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from sklearn.linear_model import LinearRegression
import csv
from timeit import default_timer as timer

data = pd.read_csv("data/burke-dataset-missing-value-averages.csv")

# remove missing data

indices_to_drop = []
no_nan_cols = ["UDel_temp_popweight","UDel_precip_popweight","growthWDI"]
for index, row in enumerate(data.itertuples()):
    if any(np.isnan(getattr(row,col)) for col in no_nan_cols):
        indices_to_drop.append(index)
data = data.drop(indices_to_drop)

# data scaling

precip_scaler, temp_scaler = StandardScaler(), StandardScaler()
precip_scaled = precip_scaler.fit_transform(np.array(data.UDel_precip_popweight).reshape(-1,1)).flatten()
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

    gdp_intercept = pm.Normal('gdp_intercept',0,5)
    temp_gdp_coef = pm.Normal('temp_gdp_coef',0,2)
    temp_sq_gdp_coef = pm.Normal('temp_sq_gdp_coef',0,2)
    precip_gdp_coef = pm.Normal("precip_gdp_coef",0,2)
    precip_sq_gdp_coef = pm.Normal("precip_sq_gdp_coef",0,2)

    year_coefs = pt.expand_dims(pm.Normal("year_coefs", 0, 5, shape=(len(set(data.year)))),axis=1)
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_coefs*year_mult_mat,axis=0))

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
        gradual_effects
    )

    gdp_std = pm.HalfNormal('gdp_std', sigma=1)
    gdp_posterior = pm.Normal('gdp_posterior', mu=gdp_prior, sigma=gdp_std, observed=data.growthWDI)

    # start timer
    start = timer()

    init_vals = {"gdp_intercept":0}
    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4, init="adapt_diag", initvals=init_vals)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # end timer
    end = timer()

# save model to file

with open ('models/PyMC/burke-reproduction-mcmc-fixed-effects-grad-effects-missing-value-averages-predictor-unscaled.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior,
        "precip_scaler":precip_scaler,
        "temp_scaler":temp_scaler
    },buff)

print("Runtime:")
print(end - start)
