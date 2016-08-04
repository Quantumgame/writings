library(car)
library(nlme)
library(effects)

d = read.csv('/auto/tdrive/mschachter/data/aggregate/rnn_best.csv')
d$electrode = factor(d$electrode)
d$n_unit = factor(d$n_unit)
