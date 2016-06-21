library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/decoder_perfs_for_glm.csv')

i = (d$decomp == 'full_psds') | (d$decomp == 'spike_rate') | (d$decomp == 'spike_rate+spike_sync')
i = i & (d$r2 >= 0)
d = subset(d, i)
d$decomp = factor(d$decomp)

m = lm(r2 ~ aprop:decomp, data=d)

Anova(m)
summary(m)

effect("aprop:decomp", m)

i_psd = d$decomp == 'full_psds'
i_pcf = d$decomp == 'full_psds+full_cfs'
i_sync = d$decomp == 'spike_rate+spike_sync'

t.test(d$r2[i_psd], d$r2[i_pcf])
t.test(d$r2[i_psd], d$r2[i_sync])
