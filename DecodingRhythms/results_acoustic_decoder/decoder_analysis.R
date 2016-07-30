library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/decoder_perfs_for_glm.csv')

i = (d$decomp == 'full_psds') | (d$decomp == 'spike_rate') | (d$decomp == 'spike_rate+spike_sync')
i = i & (d$r2 >= 0)
d = subset(d, i)
d$decomp = factor(d$decomp)


unique_aprops = unique(d$aprop)
for (k in 1:length(unique_aprops))
{
  i = d$aprop == unique_aprops[k]
  di = subset(d, i)
  
  i_rate = di$decomp == 'spike_rate'
  i_psd = di$decomp == 'full_psds'
  i_sync = di$decomp == 'spike_rate+spike_sync'
  
  cat(sprintf("-------------------------%s --------------------------\n", unique_aprops[k]))
  
  print(t.test(di$r2[i_psd], di$r2[i_rate]))
  print(t.test(di$r2[i_psd], di$r2[i_sync]))
  print(t.test(di$r2[i_rate], di$r2[i_sync]))
}
