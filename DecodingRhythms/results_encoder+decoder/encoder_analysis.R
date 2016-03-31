library(car)
library(nlme)
library(effects)

#######################################
# PSD Encoder Analysis
#######################################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/encoder_perfs_for_glm.csv')
d$freq = as.factor(d$freq)

m = lm(r2 ~ region + freq, data=d)
summary(m)
Anova(m)

m_lme = lme(r2 ~ region + freq, random=~1 | site , data=d)


# weight data
d = read.csv('/auto/tdrive/mschachter/data/aggregate/encoder_weights_for_glm.csv')
d$freq = as.factor(d$freq)

m = lm(w ~ aprop + freq + region, data=d)
Anova(m)

m = lm(w ~ aprop:freq + aprop:region, data=d)
Anova(m)
summary(m)

# m = lm(w ~ aprop*freq*region, data=d)  # triplet interactions are not very significant


#####################################
# Pairwise Encoder Analysis
#####################################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/pairwise_encoder_perfs_for_glm.csv')
d$lag = as.factor(d$lag)
d = subset(d, d$r2 > 0)

m = lm(r2 ~ regions + lag, data=d)
summary(m)
Anova(m)


d = read.csv('/auto/tdrive/mschachter/data/aggregate/pairwise_encoder_weights_for_glm.csv')
d$lag = as.factor(d$lag)
i = abs(d$w) > 3e-1
d = subset(d, i)
m = lm(w ~ aprop:lag, data=d)
summary(m)
