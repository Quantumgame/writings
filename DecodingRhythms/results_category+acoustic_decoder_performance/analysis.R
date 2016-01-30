library(car)
library(nlme)
library(effects)

############################
# GLM Analysis of Peformance
############################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/multi_electrode_perfs_for_glm.csv')

# narrow focus to 3 top decomps
i = (d$decomp == 'spike_psd') | (d$decomp == 'locked') | (d$decomp == 'spike_rate')
d = d[i,]
d$decomp = factor(d$decomp)
d$decomp = relevel(d$decomp, 'spike_rate')

# aprop = 'category'
aprop = 'maxAmp'

run_analysis = function(ds, aprop)
{
  print(sprintf('#################################'))
  print(sprintf('Running Analysis for %s', aprop))
  print(sprintf('#################################'))
  
  i = ds$aprop == aprop
  d_sub = subset(d, i) 
  
  m_sub = lm(perf ~ decomp, data=d_sub)
  print(summary(m_sub))
  print(Anova(m_sub))
  
  effect("decomp", m_sub)
  
  i_lfp = d_sub$decomp == 'locked'
  i_spike_psd = d_sub$decomp == 'spike_psd'
  i_spike_rate = d_sub$decomp == 'spike_rate'
  
  # locked vs spike psd
  print(t.test(d_sub$perf[i_lfp],  d_sub$perf[i_spike_psd], paired=FALSE))
  
  # locked vs spike rate
  print(t.test(d_sub$perf[i_lfp],  d_sub$perf[i_spike_rate], paired=FALSE))
  
  # spike psd vs spike rate
  print(t.test(d_sub$perf[i_spike_psd],  d_sub$perf[i_spike_rate], paired=FALSE))
  
}

# aprops = c('maxAmp', 'sal', 'q1', 'q2', 'q3')
aprops = c('category')
for (a in aprops)
{
  run_analysis(d, a)  
}



#########################
# Multielectrode analysis
#########################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/multi_electrode_perfs.csv')
d0 = subset(d, d$band == 0)
attach(d0)

hist(perf_category_lfp)
hist(perf_category_spike)

hist(perf_q2_lfp)
hist(perf_q2_spike)


################################
# Single Electrode Analysis
################################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/single_electrode_perfs.csv')
attach(d)

m_cat = lm(lkrat_category ~ region, data=d, subset=d$region != '?')
summary(m_cat)
Anova(m_cat)

m_cat_lme = lme(perf_category ~ region, random=~1 | bird , data=d)
summary(m_cat_lme)

m_q2 = lm(lkrat_q2 ~ region, data=d)
summary(m_q2)
Anova(m_q2)


######################
# Single Cell Analysis
######################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/cell_perfs.csv')
d$region = relevel(d$region, ref="HP")

i = d$region != '?' # exclude cells from unidentified anatomical regions
d = subset(d, i)

###########################
# can likelihood ratio for categories be predicted from likelihood ratios of acoustic features?
m_lk = lm(lkrat_category ~ lkrat_maxAmp + lkrat_sal + lkrat_q1 + lkrat_q2 + lkrat_q3 + lkrat_meanspect + lkrat_meantime + lkrat_entropytime, data=d[d$lkrat_category > 1,])

# same question, but for individual perforances
m_perf = lm(perf_category ~ perf_maxAmp + perf_sal + perf_q1 + perf_q2 + perf_q3 + perf_meanspect + perf_meantime + perf_entropytime, data=d[d$lkrat_category > 1,])


###########################
# do likelihood ratios/R2s covary with eachother? 
cor(cbind(d$lkrat_category, d$lkrat_maxAmp, d$lkrat_sal, d$lkrat_q1, d$lkrat_q2, d$lkrat_q3, d$lkrat_meanspect, d$lkrat_meantime, d$lkrat_entropytime))

cor(cbind(d$perf_category, d$perf_maxAmp, d$perf_sal, d$perf_q1, d$perf_q2, d$perf_q3, d$perf_meanspect, d$perf_meantime, d$perf_entropytime))

# For R2, yes, the properties covary with eachother. For likelihood ratio, most do not covary with eachother,
# except for q1 and q2.


##########################
# is there tonotopy? does decoding of frequency quantiles depend on region?

lkrat_good = d$lkrat_q1 > 1
m1_lk = lm(lkrat_q1 ~ region, data=d[lkrat_good,])
m1_r2 = lm(perf_q1 ~ region, data=d)

lkrat_good = d$lkrat_q2 > 1
m2_lk = lm(lkrat_q2 ~ region, data=d[lkrat_good,])
m2_r2 = lm(perf_q2 ~ region, data=d)

lkrat_good = d$lkrat_q3 > 1
m3_lk = lm(lkrat_q3 ~ region, data=d[lkrat_good,])
m3_r2 = lm(perf_q3 ~ region, data=d)

summary(m1_lk)
summary(m2_lk)
summary(m3_lk)

summary(m1_r2)
summary(m2_r2)
summary(m3_r2)

# There does not seem to be any strong relationship between
# the ability to decode the quantiles from a cell and the
# anatomical region that cell is from. The R2 values are
# very low (< 0.03)


####################
# are any regions particularly important for decoding amplitude?
lkrat_good = d$lkrat_maxAmp > 1
m_amp_lk = lm(lkrat_maxAmp ~ region, data=d[lkrat_good,])
m_amp_r2 = lm(perf_maxAmp ~ region, data=d)

summary(m_amp_lk)
summary(m_amp_r2)

# No, ability to decode amplitude does not seem to covary with region.


####################
# are any regions particularly important for decoding meantime?
lkrat_good = d$lkrat_meantime > 1
m_meantime_lk = lm(lkrat_meantime ~ region, data=d[lkrat_good,])
m_meantime_r2 = lm(perf_meantime ~ region, data=d)

summary(m_meantime_lk)
summary(m_meantime_r2)

# No, not really.


####################
# are any regions particularly important for decoding saliency?
lkrat_good = d$lkrat_sal > 1
m_sal_lk = lm(lkrat_sal ~ region, data=d[lkrat_good,])
m_sal_r2 = lm(perf_sal ~ region, data=d)

summary(m_sal_lk)
summary(m_sal_r2)

# Nope.


#####################
# Are certain regions better at decoding category than others?
lkrat_good = d$lkrat_category > 1
m_cat_lk = lm(lkrat_category ~ region, data=d[lkrat_good,])
m_cat_r2 = lm(perf_category ~ region, data=d)

summary(m_cat_lk)
summary(m_cat_r2)

# No, doesn't appear as though category decoding performance
# covaries by region.



###############################
# LFP Weight analysis
##############################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/lfp_weight_data.csv')
d$region = relevel(d$region, ref="HP")
d$electrode = as.factor(d$electrode)
d$f = as.factor(round(d$f))

d = subset(d, d$region != '?')
d = subset(d, abs(d$weight) > 0.05) # filter by 2 standard deviations


####################
# what parameters influence spectral quantiles?
i = d$aprop == 'q1'
m_q1 = lm(weight ~ row + col +region + f, data=d[i,])

i = d$aprop == 'q2'
m_q2 = lm(weight ~ row + col +  region + f, data=d[i,])

i = d$aprop == 'q3'
m_q3 = lm(weight ~ row + col +  region + f, data=d[i,])

Anova(m_q1)
Anova(m_q2)
Anova(m_q3)


summary(m_q1)
summary(m_q2)
summary(m_q3)

# Weights are positive and higher for medial electrodes, more negative moving caudally, and higher for CMM,
# with no other regional effects. No effect from hemisphere (removed from regression). Top 5 contributors
# to decrease in weight are 17Hz, 33Hz, 50Hz, 182Hz. Average R2 = 0.11

# When weights below 2SD are removed:
# q1: R2 = 0.53, and only 33-50Hz are significant
# q2: R2 = 0.48, 17-66Hz, and 182Hz are signficant. L2, L3, NCM are significant
# q3: R2 = 0.53, 17-66Hz, and 182Hz are significant


#################
# what parameters influence salience weights?
i = d$aprop == 'sal'
m_sal = lm(weight ~ row + col + hemi + region + f, data=d[i,])

Anova(m_sal)
summary(m_sal)
hist(d$weight[i])

# Effects from rostral-caudal axis, region, frequency. L1 and L3 tend to have negative weights,
# higher in magnitude than other significant regions (L2, CML). Caudal
# has lower weight than rostral. Top 5 contributors to decrease in weight
# are 33Hz, 50Hz, 116Hz, and 182Hz. R2 = 0.09


#################
# what parameters influence maxAmp?
i = d$aprop == 'maxAmp'
m_amp = lm(weight ~ row + col + hemi + region + f, data=d[i,])

Anova(m_amp)
summary(m_amp)

# No effect from rostral-caudal axis, some from medial/lateral, region, and frequency. Increase
# in weight for medial, and region L. Only significant frequencies are 17Hz and 182Hz. 


##########################
# Cell Weight Analysis

d = read.csv('/auto/tdrive/mschachter/data/aggregate/cell_weight_data.csv')
d$region = relevel(d$region, ref="HP")
d$electrode = as.factor(d$electrode)
d$f = as.factor(round(d$f))

d = subset(d, d$region != '?')
d = subset(d, abs(d$weight) > sd(abs(d$weight))) # filter by a standard deviation

# is the cell weight related to the electrode weight for the quantiles?
i = d$aprop == 'q1'
m_q1 = lm(weight ~ row + region + f + electrode_weight, data=d[i,])

i = d$aprop == 'q2'
m_q2 = lm(weight ~ row + region + f + electrode_weight, data=d[i,])

i = d$aprop == 'q3'
m_q3 = lm(weight ~ row + region + f + electrode_weight, data=d[i,])

Anova(m_q1)
Anova(m_q2)
Anova(m_q3)

summary(m_q1)
summary(m_q2)
summary(m_q3)

# cell weight is most influenced by electrode weight

