library(car)
library(nlme)
library(effects)

#######################################
# LFP Encoder Performance Analysis
#######################################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder.csv')
i = (d$region != '?') & (d$region != 'HP') & (d$region != 'L')

d = subset(d, i)
d$electrode = as.factor(d$electrode)
d$region = factor(d$region)

m = lm(cc ~ region, data=d)
summary(m)
Anova(m)
effect("region", m)

m = lm(cc ~ dist_l2a + dist_midline, data=d)
summary(m)
Anova(m)

