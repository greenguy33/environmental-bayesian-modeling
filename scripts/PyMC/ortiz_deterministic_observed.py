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

# construct model and sample

with pm.Model() as model:

    tfp_intercept = pm.Normal('tfp_intercept',0,5)
    fd_temp_tfp_coef = pm.Normal('fd_temp_tfp_coef',0,1)
    # fd_sq_temp_tfp_coef = pm.Normal('fd_sq_temp_tfp_coef',0,1)
    # fd_precip_tfp_coef = pm.Normal("fd_precip_tfp_coef",0,1)
    # fd_sq_precip_tfp_coef = pm.Normal("fd_sq_precip_tfp_coef",0,1)

    # year_coefs = pt.expand_dims(pm.Normal("year_coefs", .1, .5, shape=(len(set(data.year)))),axis=1)
    # year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum((year_coefs*year_mult_mat),axis=0))

    # country_coefs = pt.expand_dims(pm.Normal("country_coefs", .2, 1, shape=(len(set(data.ISO3)))),axis=1)
    # country_fixed_effects = pm.Deterministic("country_fixed_effects",pt.sum((country_coefs*country_mult_mat),axis=0))

    tfp_prior = pm.Deterministic(
        "tfp_prior",
        tfp_intercept +
        (fd_temp_tfp_coef * data.fd_tmean) #+
        # (fd_sq_temp_tfp_coef * data.fd_tmean_sq) +
        # (fd_precip_tfp_coef * data.fd_prcp) #+
        # (fd_sq_precip_tfp_coef * data.fd_prcp_sq) #+
        # year_fixed_effects +
        # country_fixed_effects
    )

    tfp_std = pm.HalfNormal('tfp_std', sigma=1)
    tfp_posterior = pm.Normal('tfp_posterior', mu=tfp_prior, sigma=tfp_std, observed=data.fd_log_tfp)

    start = timer()

    prior = pm.sample_prior_predictive()
    trace = pm.sample(cores=4, target_accept=0.99)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    end = timer()

    print("Runtime:")
    print(end - start)

# read gcm data from file

# gcm_data = pd.read_pickle("data/ortiz_gcm_data.pkl")

# # get coefficients from model

# coef1 = trace.posterior.fd_temp_tfp_coef.data.flatten()
# coef2 = trace.posterior.fd_sq_temp_tfp_coef.data.flatten()
# coef3 = trace.posterior.fd_precip_tfp_coef.data.flatten()
# coef4 = trace.posterior.fd_sq_precip_tfp_coef.data.flatten()

# print("coef1")
# print(np.mean(coef1))
# print("coef2")
# print(np.mean(coef2))
# print("coef3")
# print(np.mean(coef3))
# print("coef4")
# print(np.mean(coef4))

# # revenue weights

# revenue_data = pd.read_csv("data/ortiz_revenue_shares.csv")
# country_weights = {}
# for row in revenue_data.itertuples():
#     if row[3] in set(data.ISO3):
#         country_weights[row[3]] = np.mean([row[5],row[6],row[7],row[8],row[9],row[10]])
# weight_sum = sum(list(country_weights.values()))
# for country, val in country_weights.items():
#     country_weights[country] = val/weight_sum


# # Historical climate means

# country_climate_means = {}
# country_1961_means = {}
# ccm_file = pd.read_csv("data/ortiz_country_climate_means.csv")
# for row in ccm_file.itertuples():
#     country_climate_means[row[2]] = {"mean_temp":row[3], "mean_prcp":row[4]}
# c1961_file = pd.read_csv("data/ortiz_country_climate_1961.csv")
# for row in c1961_file.itertuples():
#     country_1961_means[row[2]] = {"mean_temp":row[6],"mean_prcp":row[4]}

# temp_hist_dev, temp_nat_dev, prcp_hist_dev, prcp_nat_dev = {}, {}, {}, {}
# t1, t2, p1, p2 = {}, {}, {}, {}
# t_diff, t_2_diff, p_diff, p_2_diff, = {}, {}, {}, {}

# for gcm in gcm_data:
    
#     temp_hist_dev[gcm] = {}
#     temp_nat_dev[gcm] = {}
#     prcp_hist_dev[gcm] = {}
#     prcp_nat_dev[gcm] = {}
#     t1[gcm] = {}
#     t2[gcm] = {}
#     p1[gcm] = {}
#     p2[gcm] = {}
#     t_diff[gcm] = {}
#     p_diff[gcm] = {}
#     t_2_diff[gcm] = {}
#     p_2_diff[gcm] = {}
    
#     for country in set(data.ISO3):
    
#         temp_hist_dev[gcm][country] = np.array(gcm_data[gcm]["hist_temp"][country]) - country_climate_means[country]["mean_temp"]
#         temp_nat_dev[gcm][country] = np.array(gcm_data[gcm]["hist_nat_temp"][country]) - country_climate_means[country]["mean_temp"]
#         prcp_hist_dev[gcm][country] = 1 + (np.array(gcm_data[gcm]["hist_prcp"][country]) - country_climate_means[country]["mean_prcp"]) / country_climate_means[country]["mean_prcp"]
#         prcp_nat_dev[gcm][country] = 1 + (np.array(gcm_data[gcm]["hist_nat_prcp"][country]) - country_climate_means[country]["mean_prcp"]) / country_climate_means[country]["mean_prcp"]
        
#         t1[gcm][country] = temp_nat_dev[gcm][country] + country_1961_means[country]["mean_temp"]
#         t2[gcm][country] = temp_hist_dev[gcm][country] + country_1961_means[country]["mean_temp"]
#         p1[gcm][country] = prcp_nat_dev[gcm][country] * country_1961_means[country]["mean_prcp"]
#         p2[gcm][country] = prcp_hist_dev[gcm][country] * country_1961_means[country]["mean_prcp"]
    
#         t_diff[gcm][country] = t2[gcm][country] - t1[gcm][country]
#         p_diff[gcm][country] = p2[gcm][country] - p1[gcm][country]
#         t_2_diff[gcm][country] = pow(t2[gcm][country],2) - pow(t1[gcm][country],2)
#         p_2_diff[gcm][country] = pow(p2[gcm][country],2) - pow(p1[gcm][country],2)

# random.seed(0)
# gcm_sample = random.choices(list(gcm_data.keys()), k=len(coef1))
# global_impacts = []
# country_res = {}
# for country in set(data.ISO3):
#     country_res[country] = []
#     for i in range(len(coef1)):
#         country_res[country].append(
#             t_diff[gcm_sample[i]][country]*coef1[i] + 
#             t_2_diff[gcm_sample[i]][country]*coef2[i] + 
#             p_diff[gcm_sample[i]][country]*coef3[i] + 
#             p_2_diff[gcm_sample[i]][country]*coef4[i]
#             )
# gcm_global_impacts = []
# for coef in range(len(coef1)):
#     coef_vals = []
#     for year in range(0,60):
#         year_vals = []
#         for country, values in country_res.items():
#             year_vals.append(values[coef][year] * country_weights[country])
#         coef_vals.append(np.sum(year_vals))
#     gcm_global_impacts.append(np.cumsum(coef_vals))
# global_impacts.append([arr[-1] for arr in gcm_global_impacts])

# print(np.mean(global_impacts),np.std(global_impacts))

# with open("results/ortiz_precip_dep.txt", "w") as write_file:
#     writer = csv.writer(write_file)
#     writer.writerow(["mean","sd"])
#     writer.writerow([np.mean(global_impacts),np.std(global_impacts)])

# save model to file

with open ('models/nature_reproduction/ortiz-bobea-reproduction-year-country-fixed-effects-deterministic-observed.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior
    },buff)
