with pm.Model() as model:

    global_temp_mean_prior = pm.Normal("global_temp_mean_prior",0,1)
    global_temp_sd_prior = pm.HalfNormal("global_temp_mean_sd",1)
    country_coefs_temp_prior = pt.expand_dims(pm.Normal("country_coefs_temp_prior", global_temp_mean_prior, global_temp_sd_prior, shape=(len(set(data.iso)))),axis=1)
    country_temp_priors = pm.Deterministic("temp_prior",pt.sum(country_coefs_temp_prior*country_mult_mat,axis=0))

    global_precip_mean_prior = pm.Normal("global_precip_mean_prior",0,1)
    global_precip_sd_prior = pm.HalfNormal("global_precip_mean_sd",1)
    country_coefs_precip_prior = pt.expand_dims(pm.Normal("country_coefs_precip_prior", global_precip_mean_prior, global_precip_sd_prior, shape=(len(set(data.iso)))),axis=1)
    country_precip_priors = pm.Deterministic("precip_prior",pt.sum(country_coefs_precip_prior*country_mult_mat,axis=0))
    
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

with open ('models/burke-reproduction-mcmc-fixed-effects-grad-effects-deterministic-observed-missing-rows-imputed-hierarchical-country-priors.pkl', 'wb') as buff:
  pkl.dump({
      "prior":prior,
      "trace":trace,
      "posterior":posterior,
      "precip_scaler":precip_scaler,
      "temp_scaler":temp_scaler,
      "gdp_scaler":gdp_scaler
  },buff)
