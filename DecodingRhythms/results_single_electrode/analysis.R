analyze_model_perf = function()
{
  dataFile = '/auto/tdrive/mschachter/data/aggregate/glmm_correlation_single.csv'
  
  ds = read.table(dataFile, sep=',', header=TRUE)
  ds$electrode = as.factor(ds$electrode)
  
  return(ds)
}



fit_models = function()
{
  m_total = MCMCglmm(pcc ~ hemi + decomp + region, random=~ bird + site + protocol, data=ds)
  
  m_decomp = MCMCglmm(pcc ~ decomp, random=~ bird + site + protocol, data=ds)
  
}