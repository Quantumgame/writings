library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/decoder_weights_for_glm.csv')
summary(d)

d$electrode = factor(d$electrode)
d$f = factor(d$f)

i = (d$r2 > 0) & (d$reg != '?') & (d$reg != 'HP')
m = lm(w^2 ~ aprop*f, data=subset(d, i))
Anova(m)
summary(m)

effect("aprop:f", m)
