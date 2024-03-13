import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from timeit import default_timer as timer
import csv

# Missing climate data integrated from https://climateknowledgeportal.worldbank.org/country/{country}/trends-variability-projections
data = pd.read_csv("data/burke-dataset.csv")

# Set reasonable priors from Wikipedia climate data for completely missing climate data
missing_climate_data = {
    "ATG" : {"temp":27, "precip":990},
    "CHI" : {"temp":11, "precip":1126},
    "FSM" : {"temp":28.5, "precip":2380},
    "HKG" : {"temp":23, "precip":2220},
    "IMY" : {"temp":10.5, "precip":1766},
    "KNA" : {"temp":25.3, "precip":2372},
    "KSV" : {"temp":9.5, "precip":950},
    "MAC" : {"temp":23.1, "precip":1965},
    "WBG" : {"temp":22.6, "precip":363}
}
with open("data/missing_country_climate_means.csv") as missing_climate_means:
    reader = csv.reader(missing_climate_means)
    next(reader)
    for row in reader:
        missing_climate_data[row[0]] = {}
        missing_climate_data[row[0]]["temp"] = row[1]
        missing_climate_data[row[0]]["precip"] = row[2]

# remove all data for years/countries where one of the 3 variables is entirely missing

country_temp_data = {}
country_precip_data = {}
country_gdp_data = {}

for row in data.iterrows():
    country = row[1].iso
    year = row[1].year
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

countries_missing_all = set(countries_missing_temp).intersection(set(countries_missing_precip)).intersection(set(countries_missing_gdp))
print("Removing all rows from ", list(countries_missing_all))

indices_to_drop = []
for index, row in enumerate(data.itertuples()):
    if row.iso in countries_missing_all:
        indices_to_drop.append(index)
        
data_len_before = len(data)
data = data.drop(indices_to_drop)
data = data.reset_index()
print(f"Removed {data_len_before - len(data)} rows for completely missing country data.")

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

# set country specific priors for climate variables
country_temps, country_precips, country_temp_priors, country_precip_priors = {}, {}, {}, {}
country_temp_means, country_temp_sds, country_precip_means, country_precip_sds = [], [], [], []
for row_index, row in enumerate(data.itertuples()):
    country = row.iso
    if country not in country_temps:
        country_temps[country] = []
        country_precips[country] = []
    if not np.isnan(temp_scaled[row_index]):
        country_temps[country].append(temp_scaled[row_index])
        country_precips[country].append(precip_scaled[row_index])
for row_index, row in enumerate(data.itertuples()):
    country = row.iso
    if len(country_temps[country]) > 0:
        country_temp_mean = np.mean(country_temps[country])
        country_temp_std = np.std(country_temps[country])
    else:
        country_temp_mean = temp_scaler.transform(np.array(missing_climate_data[country]["temp"]).reshape(-1,1)).flatten()[0]
        country_temp_std = abs(country_temp_mean / 3)
    country_temp_means.append(country_temp_mean)
    country_temp_sds.append(country_temp_std)
    if len(country_precips[country]) > 0:
        country_precip_mean = np.mean(country_precips[country])
        country_precip_std = np.std(country_precips[country])        
    else:
        country_precip_mean = precip_scaler.transform(np.array(missing_climate_data[country]["precip"]).reshape(-1,1)).flatten()[0]
        country_precip_std = abs(country_precip_mean / 3)
    country_precip_means.append(country_precip_mean)
    country_precip_sds.append(country_precip_std)

# construct model and sample

with pm.Model() as model:

    country_temp_priors = pm.Normal("country_temp_priors", country_temp_means, country_temp_sds)
    country_precip_priors = pm.Normal("country_precip_priors", country_precip_means, country_precip_sds)
    
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

with open ('models/burke-reproduction-mcmc-fixed-effects-grad-effects-deterministic-observed-missing-rows-imputed-original-dataset.pkl', 'wb') as buff:
  pkl.dump({
      "prior":prior,
      "trace":trace,
      "posterior":posterior,
      "precip_scaler":precip_scaler,
      "temp_scaler":temp_scaler,
      "gdp_scaler":gdp_scaler
  },buff)
