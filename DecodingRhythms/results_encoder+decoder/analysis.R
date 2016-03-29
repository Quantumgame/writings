library(car)
library(nlme)
library(effects)

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
