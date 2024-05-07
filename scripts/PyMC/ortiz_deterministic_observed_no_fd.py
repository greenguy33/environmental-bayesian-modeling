import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from sklearn.linear_model import LinearRegression
import csv
import random
from timeit import default_timer as timer

data = pd.read_csv("data/ortiz-bobea/data2/regdata_preferred_case_2_years_withheld.csv")

# year and country fixed effect coefficient matrices

data_len = len(data.fd_tmean)
year_mult_mat = [np.zeros(data_len) for year in set(data.year)]
country_mult_mat = [np.zeros(data_len) for country in set(data.ISO3)]
country_index = -1
curr_country = ""
for row_index, row in enumerate(data.itertuples()):
    if row.ISO3 != curr_country:
        country_index += 1
        curr_country = row.ISO3
    year_index = row.year - 1962
    country_mult_mat[country_index][row_index] = 1
    year_mult_mat[year_index][row_index] = 1

precip_scaler, tfp_scaler, temp_scaler = StandardScaler(), StandardScaler(), StandardScaler()
precip_scaled = precip_scaler.fit_transform(np.array(data.prcp).reshape(-1,1)).flatten()
tfp_scaled = tfp_scaler.fit_transform(np.array(data.fd_log_tfp).reshape(-1,1)).flatten()
temp_scaled = temp_scaler.fit_transform(np.array(data.tmean).reshape(-1,1)).flatten()

# construct model and sample

with pm.Model() as model:

    tfp_intercept = pm.Normal('tfp_intercept',0,5)
    fd_temp_tfp_coef = pm.Normal('fd_temp_tfp_coef',0,1)
    fd_sq_temp_tfp_coef = pm.Normal('fd_sq_temp_tfp_coef',0,1)
    fd_precip_tfp_coef = pm.Normal("fd_precip_tfp_coef",0,1)
    fd_sq_precip_tfp_coef = pm.Normal("fd_sq_precip_tfp_coef",0,1)

    # year_coefs = pt.expand_dims(pm.Normal("year_coefs", .1, .5, shape=(len(set(data.year)))),axis=1)
    # year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum((year_coefs*year_mult_mat),axis=0))

    # country_coefs = pt.expand_dims(pm.Normal("country_coefs", .2, 1, shape=(len(set(data.ISO3)))),axis=1)
    # country_fixed_effects = pm.Deterministic("country_fixed_effects",pt.sum((country_coefs*country_mult_mat),axis=0))

    tfp_prior = pm.Deterministic(
        "tfp_prior",
        tfp_intercept +
        (fd_temp_tfp_coef * temp_scaled) +
        (fd_sq_temp_tfp_coef * np.square(temp_scaled)) +
        (fd_precip_tfp_coef * precip_scaled) +
        (fd_sq_precip_tfp_coef * np.square(precip_scaled)) #+
        # year_fixed_effects +
        # country_fixed_effects
    )

    tfp_std = pm.HalfNormal('tfp_std', sigma=1)
    tfp_posterior = pm.Normal('tfp_posterior', mu=tfp_prior, sigma=tfp_std, observed=tfp_scaled)

    start = timer()

    prior = pm.sample_prior_predictive()
    trace = pm.sample(cores=4, target_accept=0.99)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    end = timer()

    print("Runtime:")
    print(end - start)


# save model to file

with open ('models/nature_reproduction/ortiz-bobea-reproduction-year-country-fixed-effects-deterministic-observed-no-fd.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior
    },buff)
