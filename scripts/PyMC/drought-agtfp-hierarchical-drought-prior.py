import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from timeit import default_timer as timer

data = pd.read_csv("data/drought-agtfp-gdp-data.csv")
print(len(data))

# remove missing data rows

indices_to_drop = []
no_nan_cols = ["ln_TFP_change"]
for index, row in enumerate(data.itertuples()):
    if any(np.isnan(getattr(row,col)) for col in no_nan_cols):
        indices_to_drop.append(index)
data = data.drop(indices_to_drop)
data = data.reset_index()
print(len(data))

# year and country fixed effect coefficient matrices

min_year = min(data.year)
data_len = len(data.year)
year_fixed_effect_matrix = [np.zeros(data_len) for year in set(data.year)]
country_fixed_effect_matrix = [np.zeros(data_len) for country in set(data.country)]
country_index = -1
curr_country = ""
for row_index, row in enumerate(data.itertuples()):
    if row.country != curr_country:
        country_index += 1
        curr_country = row.country
    year_index = row.year - min_year
    country_fixed_effect_matrix[country_index][row_index] = 1
    year_fixed_effect_matrix[year_index][row_index] = 1

# gradual effect coefficient matrix

country_counters = {}
country_grad_effect_matrix = [np.zeros(data_len) for country in set(data.country)]
country_index = -1
curr_country = ""
for row_index, row in enumerate(data.itertuples()):
    if row.country != curr_country:
        country_index += 1
        curr_country = row.country
        if curr_country not in country_counters:
            country_counters[curr_country] = 1
    country_grad_effect_matrix[country_index][row_index] = country_counters[curr_country]
    country_counters[curr_country] += 1

# scale dependent variable

tfp_scaler = StandardScaler()
tfp_scaled = tfp_scaler.fit_transform(np.array(data.ln_TFP_change).reshape(-1,1)).flatten()

# construct model and sample

with pm.Model() as model:

    global_drought_prior = pm.HalfNormal("global_drought_prior", 1)
    global_no_drought_prior = pm.HalfNormal("global_no_drought_prior", 10)
    
    country_drought_prior_coefs = pt.expand_dims(pm.HalfNormal("country_drought_prior_coefs", global_drought_prior, shape=(len(set(data.country)))),axis=1)
    country_no_drought_prior_coefs = pt.expand_dims(pm.HalfNormal("country_no_drought_prior_coefs", global_no_drought_prior, shape=(len(set(data.country)))),axis=1)

    country_drought_priors = pm.Deterministic("country_drought_prior",pt.sum(country_drought_prior_coefs*country_fixed_effect_matrix,axis=0))
    country_no_drought_priors = pm.Deterministic("country_no_drought_prior",pt.sum(country_no_drought_prior_coefs*country_fixed_effect_matrix,axis=0))
    
    drought_prior = pm.Dirichlet("drought_prior", pt.transpose([country_no_drought_priors, country_drought_priors]))
    drought_posterior = pm.Categorical("drought_posterior", drought_prior, observed=data.drought)

    drought_tfp_coef = pm.Normal("drought_tfp_coef", 0, .1)
    intercept = pm.Normal("intercept", 0, .1)

    year_fixed_effect_coefs = pt.expand_dims(pm.Normal("year_fixed_effect_coefs", 0, 10, shape=(len(set(data.year)))),axis=1)
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_fixed_effect_coefs*year_fixed_effect_matrix,axis=0))

    country_grad_effect_coefs = pt.expand_dims(pm.Normal("country_grad_effect_coefs", 0, 10, shape=(len(set(data.country)))),axis=1)
    country_grad_effects = pm.Deterministic("grad_effects",pt.sum(country_grad_effect_coefs*country_grad_effect_matrix,axis=0))
    
    tfp_prior = pm.Normal(
        "tfp_prior", 
        (drought_tfp_coef * drought_posterior) + 
        year_fixed_effects + 
        country_grad_effects +
        intercept
    )

    tfp_sd = pm.HalfNormal("tfp_sd", 1)
    tfp_posterior = pm.Normal("tfp_posterior", tfp_prior, tfp_sd, observed = tfp_scaled)

    start = timer()

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    end = timer()

    print("Runtime:")
    print(end - start)

    with open ('models/PyMC/drought-agtfp-hierarchical-drought-prior.pkl', 'wb') as buff:
        pkl.dump({
            "prior":prior,
            "trace":trace,
            "posterior":posterior,
            "tfp_scaler":tfp_scaler
        },buff)