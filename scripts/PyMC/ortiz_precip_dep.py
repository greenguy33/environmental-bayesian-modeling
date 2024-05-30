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

data = pd.read_csv("data/ortiz-bobea-dataset-random-withheld.csv")
# data = pd.read_csv("../../data/ortiz-bobea/data2/regdata_preferred_case_encoded_iso_id.csv")

# year and country fixed effect coefficient matrices

data_len = len(data.fd_tmean)
min_year = min(set(data.year))
year_mult_mat = [np.zeros(data_len) for year in set(data.year)]
country_mult_mat = [np.zeros(data_len) for country in set(data.ISO3)]
country_index = -1
curr_country = ""
for row_index, row in enumerate(data.itertuples()):
    if row.ISO3 != curr_country:
        country_index += 1
        curr_country = row.ISO3
    year_index = row.year - min_year
    country_mult_mat[country_index][row_index] = 1
    year_mult_mat[year_index][row_index] = 1

# construct model and sample

with pm.Model() as model:
    
    precip_country_coefs = pt.expand_dims(pm.Normal("precip_country_coefs", 0, 5, shape=(len(set(data.ISO3))-1)),axis=1)
    precip_country_coefs = pm.math.concatenate([[[0]],precip_country_coefs])
    precip_country_fixed_effects = pm.Deterministic("precip_country_fixed_effects",pt.sum(precip_country_coefs*country_mult_mat,axis=0))

    precip_year_coefs = pt.expand_dims(pm.Normal("precip_year_coefs", 0, 5, shape=(len(set(data.year))-1)),axis=1)
    precip_year_coefs = pm.math.concatenate([[[0]],precip_year_coefs])
    precip_year_fixed_effects = pm.Deterministic("precip_year_fixed_effects_sq",pt.sum(precip_year_coefs*year_mult_mat,axis=0))

    precip_intercept = pm.Normal('precip_intercept',0,10)
    temp_precip_coef = pm.Normal('temp_precip_coef',0,10)
    temp_sq_precip_coef = pm.Normal('temp_sq_precip_coef',0,10)

    fd_precip_prior = pm.Deterministic(
    	"fd_precip_prior",
    	precip_intercept + 
    	(np.array(data.fd_tmean) * temp_precip_coef) +
    	(np.array(data.fd_tmean_sq) * temp_sq_precip_coef) +
    	precip_country_fixed_effects +
        precip_year_fixed_effects
    )

    fd_precip_std = pm.HalfNormal("fd_precip_std", 10)
    fd_precip_posterior = pm.Normal("fd_precip_posterior", fd_precip_prior, fd_precip_std, observed=data.fd_prcp)

    precip_sq_year_coefs = pt.expand_dims(pm.Normal("precip_sq_year_coefs", 0, 5, shape=(len(set(data.year))-1)),axis=1)
    precip_sq_year_coefs = pm.math.concatenate([[[0]],precip_sq_year_coefs])
    precip_sq_year_fixed_effects = pm.Deterministic("precip_sq_year_fixed_effects_sq",pt.sum(precip_sq_year_coefs*year_mult_mat,axis=0))

    precip_sq_country_coefs = pt.expand_dims(pm.Normal("precip_sq_country_coefs", 0, 5, shape=(len(set(data.ISO3))-1)),axis=1)
    precip_sq_country_coefs = pm.math.concatenate([[[0]],precip_sq_country_coefs])
    precip_sq_country_fixed_effects = pm.Deterministic("precip_sq_country_fixed_effects",pt.sum(precip_sq_country_coefs*country_mult_mat,axis=0))

    precip_sq_intercept = pm.Normal('precip_sq_intercept',0,10)
    temp_precip_sq_coef = pm.Normal('temp_precip_sq_coef',0,10)
    temp_sq_precip_sq_coef = pm.Normal('temp_sq_precip_sq_coef',0,10)

    fd_sq_precip_prior = pm.Deterministic(
    	"fd_sq_precip_prior",
    	precip_sq_intercept + 
    	(np.array(data.fd_tmean) * temp_precip_sq_coef) +
    	(np.array(data.fd_tmean_sq) * temp_sq_precip_sq_coef) +
    	precip_sq_year_fixed_effects +
    	precip_sq_country_fixed_effects
    )

    fd_sq_precip_std = pm.HalfNormal("fd_sq_precip_std", 10)
    fd_sq_precip_posterior = pm.Normal("fd_sq_precip_posterior", fd_sq_precip_prior, fd_sq_precip_std, observed=data.fd_prcp_sq)

    tfp_intercept = pm.Normal('tfp_intercept',0,5)
    fd_temp_tfp_coef = pm.Normal('fd_temp_tfp_coef',0,2)
    fd_sq_temp_tfp_coef = pm.Normal('fd_sq_temp_tfp_coef',0,2)
    fd_precip_tfp_coef = pm.Normal("fd_precip_tfp_coef",0,2)
    fd_sq_precip_tfp_coef = pm.Normal("fd_sq_precip_tfp_coef",0,2)

    year_coefs = pt.expand_dims(pm.Normal("year_coefs", 0, 5, shape=(len(set(data.year))-1)),axis=1)
    year_coefs = pm.math.concatenate([[[0]],year_coefs])
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_coefs*year_mult_mat,axis=0))

    country_coefs = pt.expand_dims(pm.Normal("country_coefs", 0, 5, shape=(len(set(data.ISO3))-1)),axis=1)
    country_coefs = pm.math.concatenate([[[0]],country_coefs])
    country_fixed_effects = pm.Deterministic("country_fixed_effects",pt.sum(country_coefs*country_mult_mat,axis=0))

    tfp_prior = pm.Deterministic(
        "tfp_prior",
        tfp_intercept +
        (fd_temp_tfp_coef * np.array(data.fd_tmean)) +
        (fd_sq_temp_tfp_coef * np.array(data.fd_tmean_sq)) +
        (fd_precip_tfp_coef * fd_precip_posterior) +
        (fd_sq_precip_tfp_coef * fd_sq_precip_posterior) +
        year_fixed_effects +
        country_fixed_effects
    )

    tfp_std_scale = pm.HalfNormal("tfp_std_scale", 5)
    tfp_std = pm.HalfNormal('tfp_std', sigma=tfp_std_scale)
    tfp_posterior = pm.Normal('tfp_posterior', mu=tfp_prior, sigma=tfp_std, observed=data.fd_log_tfp)

    start = timer()

    prior = pm.sample_prior_predictive()
    trace = pm.sample(cores=4, target_accept=0.99)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    end = timer()

print("Runtime:")
print(end - start)

# save model to file

with open ('models/nature_reproduction/ortiz-bobea-reproduction-year-country-fixed-effects-precip-dep.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior
    },buff)
