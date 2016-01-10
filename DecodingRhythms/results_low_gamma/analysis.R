library(car)
library(nlme)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/gamma.csv')

# replace any ambiguous regions with a question mark
d$region[grep('-', d$region)] = '?'

# eliminate some of the messier data
i = (d$bird != 'BlaBro09xxF') & (d$spike_rate > 0) & (d$spike_freq > 0) & (d$spike_freq_std > 0) & (d$region != '?')
d = subset(d, i)

d$region = relevel(d$region, ref="HP")
d$electrode = as.factor(d$electrode)
d$cell_index = as.factor(d$cell_index)

for (k in 1:nrow(d))
{
  d$site[k] = paste(c(toString(d$bird[k]), toString(d$block[k]), toString(d$segment[k]), toString(d$hemi[k])), collapse='_')
}

d$site = as.factor(d$site)

X = cbind(d$lfp_freq, d$lfp_freq_std, d$spike_freq, d$spike_freq_std, d$spike_rate, d$spike_rate_std, d$maxAmp, d$sal, d$meanspect, d$q1, d$q2, d$q3, d$entropyspect, d$meantime, d$entropytime)

# lfp center freq encoder
m_lfp = lm(lfp_freq ~ maxAmp + sal + meanspect + q1 + q2 + q3 + entropyspect + meantime + entropytime + region + stim_type, data=d)

m_lfp = lme(lfp_freq ~ maxAmp + sal + meanspect + q1 + q2 + q3 + entropyspect + meantime + entropytime + region + stim_type, data=d, random=~ 1 | site)

# spike center freq encoder
m_spike = lm(spike_freq ~ stim_type, data=d)
summary(m_spike)

m_spike = lm(spike_freq ~ region, data=d)
summary(m_spike)

m_spike = lm(spike_freq ~ region*stim_type, data=d)
summary(m_spike)

m_spike = lm(spike_freq ~ maxAmp + sal + meanspect + q1 + q2 + q3 + entropyspect + meantime + entropytime + region + stim_type, data=d)


