library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/single_electrode_decoder.csv')
d$aprop = relevel(d$aprop, 'stdtime')

i = (d$region != '?') & (d$r2 > 0) & (d$region != 'HP')
d = subset(d, i)
d$region = factor(d$region)

m = lm(r2 ~ aprop, data=d)
Anova(m)
summary(m)

m = lm(r2 ~ region + aprop, data=d)
Anova(m)
summary(m)

m = lm(r2 ~ dist_l2a + dist_midline + aprop, data=d)
Anova(m)
summary(m)

m = lm(r2 ~ region + dist_l2a + dist_midline + aprop, data=d)
Anova(m)
summary(m)

effect("region", m)
effect("dist_l2a", m)
effect("dist_midline", m)
effect("aprop", m)


m = lm(r2 ~ aprop*(region + dist_l2a + dist_midline), data=d)
Anova(m)
summary(m)

effect("aprop*region", m)
effect("aprop*dist_l2a", m)
effect("aprop*dist_midline", m)

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



