library(car)
library(nlme)
library(effects)

#######################################
# PSD Encoder Analysis
#######################################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/encoder_perfs_for_glm.csv')
d$freq = as.factor(d$freq)

i = (d$r2 > 0) & (d$region != '?') & (d$region != 'HP')
d = subset(d, i)
d$region = factor(d$region)
d$region = relevel(d$region, 'NCM')

m = lm(r2 ~ region + freq, data=subset(d, i))
summary(m)
Anova(m)
effect("freq", m)
effect("region", m)

m_lme = lme(r2 ~ region + freq, random=~1 | site , data=d)

