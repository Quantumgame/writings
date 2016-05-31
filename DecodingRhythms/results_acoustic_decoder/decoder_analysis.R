library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/decoder_perfs_for_glm.csv')
d$decomp = relevel(d$decomp, 'self_spike_rate')
d$aprop = relevel(d$aprop, 'meantime')

m = lm(r2 ~ aprop*decomp, data=d)

Anova(m)
summary(m)


m = lm(r2 ~ aprop:decomp, data=d)
Anova(m)
summary(m)
