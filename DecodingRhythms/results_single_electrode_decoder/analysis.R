library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/single_electrode_decoder.csv')
i = (d$region != '?') & (d$r2 >= 0) & (d$region != 'HP') & (d$region != 'L')
d = subset(d, i)
d$region = factor(d$region)
d$electrode = factor(d$electrode)

unique_aprops = unique(d$aprop)
for (k in 1:length(unique_aprops))
{
  i = d$aprop == unique_aprops[k]
  di = subset(d, i)
  
  cat(sprintf("-------------------------%s --------------------------\n", unique_aprops[k]))
  
  m1 = lm(r2 ~ region, data=di)
  s1 = summary(m1)
  
  # m2 = lm(r2 ~ dist_l2a + dist_midline, data=di)
  # s2 = summary(m2)
  
  cat(sprintf('Region R2=%0.2f\n', s1$adj.r.squared))
  s1$adj.r.squared
}

i = d$aprop == 'maxAmp'
di = subset(d, i)
m = lm(r2 ~ region, data=di)
s = summary(m)
cat(sprintf('N=%d\n', nrow(di)))
effect('region', m)
Anova(m)

i = d$aprop == 'meanspect'
di = subset(d, i)
m = lm(r2 ~ region, data=di)
s = summary(m)
cat(sprintf('N=%d\n', nrow(di)))
effect('region', m)
Anova(m)

i = d$aprop == 'sal'
di = subset(d, i)
m = lm(r2 ~ region, data=di)
s = summary(m)
cat(sprintf('N=%d\n', nrow(di)))
effect('region', m)
Anova(m)

i = d$aprop == 'skewtime'
di = subset(d, i)
m = lm(r2 ~ region, data=di)
s = summary(m)
cat(sprintf('N=%d\n', nrow(di)))
effect('region', m)
Anova(m)

