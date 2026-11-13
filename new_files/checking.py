import arviz as az

idata = az.from_netcdf("results_pymc/20261107_095327/pl_inference_data.nc")  # your run's folder

summary = az.summary(idata, var_names=["e_centre", "fwhm", "area", "sigma"])
print(summary.round(3).to_string())

# optional: save it
summary.to_csv("derived_summary.csv")