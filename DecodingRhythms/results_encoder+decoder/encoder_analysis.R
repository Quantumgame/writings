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
d = subset(d, !is.na(d$dist))
d = subset(d, d$r2 > 0)

m = lm(r2 ~ lag + regions + dist, data=d)
summary(m)
Anova(m)


d = read.csv('/auto/tdrive/mschachter/data/aggregate/pairwise_encoder_weights_for_glm.csv')
d$lag = as.factor(d$lag)
d = subset(d, !is.na(d$dist))
d = subset(d, abs(d$w) > 1e-1)

m = lm(w ~ aprop:lag + dist + regions, data=d)
summary(m)
