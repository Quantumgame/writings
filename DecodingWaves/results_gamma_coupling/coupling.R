
library(car)

fname = '/auto/tdrive/mschachter/data/aggregate/coupling.csv'
df = read.csv(fname, header=TRUE, sep=",")
df$electrode1 = as.factor(df$electrode1)
df$electrode2 = as.factor(df$electrode2)

summary(df)

m1 = lm(silent_weight ~ distance + region1:region2, data=df)
summary(m1)
Anova(m1)

m2 = lm(evoked_weight ~ distance + region1:region2, data=df)
summary(m2)
Anova(m2)


fname = '/auto/tdrive/mschachter/data/aggregate/spike_phase.csv'
df = read.csv(fname, header=TRUE, sep=",")
df$electrode = as.factor(df$electrode)
df$entropy_diff = df$evoked_entropy - df$silent_entropy

summary(df)

m = lm(silent_entropy ~ region, data=df)
summary(m)

m = lm(entropy_diff ~ region, data=df)
summary(m)


