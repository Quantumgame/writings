library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/rnn_best.csv')
d$electrode = factor(d$electrode)
d$n_unit = factor(d$n_unit)

i = (d$region != 'HP') & (d$region != '?') & (d$region != 'L')
d = subset(d, i)
d$region = factor(d$region)

m = lm(cc ~ region, data=d)
summary(m)
Anova(m)
effect("region", m)

d$cc_imp = d$cc - d$linear_cc

m = lm(cc_imp ~ region, data=d)
summary(m)
Anova(m)
effect("region", m)


d = read.csv('/auto/tdrive/mschachter/data/aggregate/lfp_amp_env_cfreqs.csv')
i = (d$reg != 'HP') & (d$reg != 'L') & (d$reg != '?')
d = subset(d, i)
d$reg = factor(d$reg)
summary(d)

m = lm(freq ~ reg, data=d)
Anova(m)
summary(m)
effect("reg", m)


