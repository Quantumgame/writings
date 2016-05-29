library(car)
library(nlme)
library(effects)

#######################################
# LFP Encoder Performance Analysis
#######################################

d = read.csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder_perfs_both.csv')
d$f = as.factor(d$f)
d$electrode = as.factor(d$electrode)

d = subset(d, (d$region != '?') & (d$region != 'HP'))
d$region = factor(d$region)
d$region = relevel(d$region, 'NCM')

m = lm(r2 ~ region + f + dist_midline + dist_l2a, data=d)
Anova(m)
summary(m)

plot(effect('region', m))
plot(effect('f', m))


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
