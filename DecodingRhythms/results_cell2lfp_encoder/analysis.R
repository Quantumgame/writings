library(car)
library(nlme)
library(effects)

#######################################
# LFP Encoder Performance Analysis
#######################################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder_perfs_both.csv')
i = (d$region != '?') & (d$region != 'HP') & (d$region != 'L') & (d$r2 > 0)

d = subset(d, i)
d$f = as.factor(d$f)
d$electrode = as.factor(d$electrode)
d$region = factor(d$region)

drate = subset(d, d$ein == 'rate')
m = lm(r2 ~ region, data=drate)
Anova(m)
summary(m)
effect("region", m)

m = lm(r2 ~ f, data=drate)
Anova(m)
summary(m)
effect("f", m)


dboth = subset(d, d$ein == 'both')
m = lm(r2 ~ region, data=dboth)
Anova(m)
summary(m)
effect("region", m)

m = lm(r2 ~ f, data=dboth)
Anova(m)
summary(m)
effect("f", m)


#######################################
# LFP Encoder Weight Analysis
#######################################

# rate weights

d = read.csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder_weights.csv')
d$f = as.factor(d$f)
d$electrode = as.factor(d$electrode)
d$cell_index = as.factor(d$cell_index)
d$same_electrode = as.factor(d$same_electrode)
d$cells_same_electrode = as.factor(d$cells_same_electrode)

dr = subset(d, d$wtype == 'rate')
dr = subset(dr, !is.na(d$dist_from_electrode))
dr = subset(dr, d$r2 > 0.20)
# dr = subset(dr, d$w > 0.05)

m = lm(w ~ dist_from_electrode + same_electrode, data=dr)
m = lm(abs(w) ~ dist_from_electrode + same_electrode, data=dr)

# synchrony weights
ds = subset(d, d$wtype == 'sync')
ds = subset(ds, !is.na(ds$dist_from_electrode))
ds = subset(ds, !is.na(ds$dist_cell2cell))
ds = subset(ds, ds$r2 > 0.20)

# m = lm(abs(w) ~ dist_cell2cell + dist_from_electrode + same_electrode + cells_same_electrode, data=ds)
m = lm(abs(w) ~ dist_cell2cell, data=ds)
