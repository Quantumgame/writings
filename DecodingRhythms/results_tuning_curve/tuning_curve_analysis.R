library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/tuning_curve_for_glm.csv')
d$electrode = factor(d$electrode)
d$freq = factor(d$freq)
d$cell_index = factor(d$cell_index)

i = (d$reg != '?') & (d$reg != 'HP') & (d$reg != 'L')
d = subset(d, i)
d$region = factor(d$region)

summary(d)

m = lm(r2 ~ aprop:freq + region, data=d)

Anova(m)
summary(m)

effect("aprop:freq", m)
effect('region', m)
