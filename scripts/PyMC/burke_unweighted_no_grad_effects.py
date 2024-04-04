import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from sklearn.linear_model import LinearRegression

# Import data

data = pd.read_csv("data/burke_ortizbobea_integrated_dataset_with_custom_temp.csv")

# Remove all data for countries where one of the variables is entirely missing

country_temp_data = {}
country_precip_data = {}
country_gdp_data = {}

for row in data.iterrows():
    country = row[1].country
    if country not in country_temp_data:
        country_temp_data[country] = []
    if country not in country_precip_data:
        country_precip_data[country] = []
    if country not in country_gdp_data:
        country_gdp_data[country] = []

    country_temp_data[country].append(row[1].unweighted_temp)
    country_precip_data[country].append(row[1].unweighted_precip)
    country_gdp_data[country].append(row[1].ln_gdp_change)

countries_missing_temp = [country for country in country_temp_data if all(np.isnan(country_temp_data[country]))]
countries_missing_precip = [country for country in country_precip_data if all(np.isnan(country_precip_data[country]))]
countries_missing_gdp = [country for country in country_gdp_data if all(np.isnan(country_gdp_data[country]))]

countries_to_remove = set(countries_missing_temp + countries_missing_precip + countries_missing_gdp)
print(countries_to_remove)

indices_to_drop = []
for index, row in enumerate(data.itertuples()):
    if row.country in countries_to_remove:
        indices_to_drop.append(index)
        
data_len_before = len(data)
data = data.drop(indices_to_drop)
data = data.reset_index()
print(f"Removed {data_len_before - len(data)} rows for completely missing country data.")

# Scale data

precip_scaler, gdp_scaler, temp_scaler = StandardScaler(), StandardScaler(), StandardScaler()
precip_scaled = precip_scaler.fit_transform(np.array(data.unweighted_precip).reshape(-1,1)).flatten()
gdp_scaled = gdp_scaler.fit_transform(np.array(data.ln_gdp_change).reshape(-1,1)).flatten()
temp_scaled = temp_scaler.fit_transform(np.array(data.unweighted_temp).reshape(-1,1)).flatten()

# Year and country fixed effects

data_len = len(data.year)
year_mult_mat = [np.zeros(data_len) for year in set(data.year)]
country_mult_mat = [np.zeros(data_len) for country in set(data.country)]
country_index = -1
curr_country = ""
for row_index, row in enumerate(data.itertuples()):
    if row.country != curr_country:
        country_index += 1
        curr_country = row.country
    year_index = row.year - 1960
    country_mult_mat[country_index][row_index] = 1
    year_mult_mat[year_index][row_index] = 1

# Construct model

with pm.Model() as model:

    country_coefs_temp_prior = pt.expand_dims(pm.Normal("country_coefs_temp_prior", 0, 1, shape=(len(set(data.country)))),axis=1)
    temp_prior = pm.Deterministic("temp_prior",pt.sum(country_coefs_temp_prior*country_mult_mat,axis=0))    
    temp_std = pm.HalfNormal("temp_std", 1)
    temp_posterior = pm.Normal("temp_posterior", temp_prior, temp_std, observed=temp_scaled)
    
    temp_gdp_coef = pm.Normal('temp_gdp_coef',0,1)
    temp_gdp_coef2 = pm.Normal('temp_gdp_coef2',0,1)
    temp_gdp_intercept = pm.Normal('temp_gdp_intercept',0,1)

    country_coefs_precip_prior = pt.expand_dims(pm.Normal("country_coefs_precip_prior", 0, 1, shape=(len(set(data.country)))),axis=1)
    precip_prior = pm.Deterministic("precip_prior",pt.sum(country_coefs_precip_prior*country_mult_mat,axis=0))
    precip_std = pm.HalfNormal("precip_std", 1)
    precip_posterior = pm.Normal("precip_posterior", precip_prior, precip_std, observed=precip_scaled)

    precip_gdp_coef = pm.Normal('precip_gdp_coef',0,1)
    precip_gdp_coef2 = pm.Normal('precip_gdp_coef2',0,1)
    precip_gdp_intercept = pm.Normal('precip_gdp_intercept',0,1)

    gdp_year_coefs = pt.expand_dims(pm.Normal("gdp_year_coefs", 0, 10, shape=(len(set(data.year)))),axis=1)
    gdp_year_fixed_effects = pm.Deterministic("gdp_year_fixed_effects",pt.sum(gdp_year_coefs*year_mult_mat,axis=0))
    gdp_country_coefs = pt.expand_dims(pm.Normal("gdp_country_coefs", 0, 10, shape=(len(set(data.country)))),axis=1)
    gdp_country_fixed_effects = pm.Deterministic("gdp_country_fixed_effects",pt.sum(gdp_country_coefs*country_mult_mat,axis=0))
    
    gdp_intercept = pm.Normal("gdp_intercept", 0, 1)
    
    gdp_prior = pm.Deterministic(
        "gdp_prior", 
        gdp_intercept + 
        (temp_gdp_coef * temp_posterior) + 
        (temp_gdp_coef2 * pt.sqr(temp_posterior)) +
        (precip_gdp_coef * precip_posterior) +
        (precip_gdp_coef2 * pt.sqr(precip_posterior)) +
        gdp_year_fixed_effects +
        gdp_country_fixed_effects
    )
    gdp_std = pm.HalfNormal('gdp_std', sigma=10)
    gdp_posterior = pm.Normal('gdp_posterior', mu=gdp_prior, sigma=gdp_std, observed=data["ln_gdp_change"])
    
    prior = pm.sample_prior_predictive()
    trace = pm.sample(cores=4, target_accept=.99)
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

with open ('../models/burke_unweighted_temp_no_grad_effects.pkl', 'wb') as buff:
    pkl.dump ({
        "prior": prior, 
        "trace": trace, 
        "posterior": posterior,
        "temp_scaler": temp_scaler,
        "precip_scaler": precip_scaler,
        "gdp_scaler": gdp_scaler
    }, buff)


