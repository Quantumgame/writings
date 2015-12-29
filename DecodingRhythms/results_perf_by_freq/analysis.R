library(car)
library(nlme)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/multi_electrode_perfs.csv')
d0 = subset(d, d$band == 0)
attach(d0)

hist(perf_category_lfp)
hist(perf_category_spike)

hist(perf_q2_lfp)
hist(perf_q2_spike)


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



d = read.csv('/auto/tdrive/mschachter/data/aggregate/cell_perfs.csv')






