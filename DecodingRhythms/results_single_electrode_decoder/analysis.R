library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/single_electrode_decoder.csv')
i = (d$region != '?') & (d$r2 >= 0) & (d$region != 'HP') & (d$region != 'L')
d = subset(d, i)
d$region = factor(d$region)
d$electrode = factor(d$electrode)

m = lm(r2 ~ aprop:region, data=d)
Anova(m)
summary(m)

effect("aprop:region", m)

unique_aprops = unique(d$aprop)
for (k in 1:length(unique_aprops))
{
  i = d$aprop == unique_aprops[k]
  di = subset(d, i)
  
  print('\n')
  print('\n')
  print(sprintf("-------------------------%s --------------------------", unique_aprops[k]))
  m = lm(r2 ~ region + dist_l2a + dist_midline, data=di)
  Anova(m)
  print(summary(m))
}



