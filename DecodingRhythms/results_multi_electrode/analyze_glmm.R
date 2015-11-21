

df = read.table('/auto/tdrive/mschachter/data/aggregate/decoder_coherence_multi.csv', sep=',', header=TRUE)
summary(df)

m = lm(pcc ~ decomp + hemi + order, data=df)
summary(m)

library(effects)
effect('hemi', m)
effect('decomp', m)
effect('order', m)

