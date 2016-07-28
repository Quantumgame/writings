library(splines)
library(car)
library(glmnet)

df = read.csv('/auto/tdrive/mschachter/data/aggregate/tuning_curve_data.csv')
df$electrode = factor(df$electrode)
df$cell_index = factor(df$cell_index)
df$band = factor(df$band)

i = !is.na(df$dist_l2a) & !is.na(df$dist_midline) & (df$region != '?') & (df$r2 > 0.05)
i[is.na(i)] = FALSE
sum(i)
sum(is.na(i))

gdf = subset(df, i)
gdf$neg_amp_slope = gdf$amp_slope < 0


# band 0-30Hz
i = (gdf$aprop == 'meanspect') & (gdf$center_freq > 0) & (gdf$decomp == 'full_psds') & (gdf$band == 0)
df_freq = subset(gdf, i)
m = lm(center_freq ~ dist_l2a + dist_midline, data=df_freq)
summary(m)

# band 30-80Hz
i = (gdf$aprop == 'meanspect') & (gdf$center_freq > 0) & (gdf$decomp == 'full_psds') & (gdf$band == 1)
df_freq = subset(gdf, i)
m = lm(center_freq ~ dist_l2a + dist_midline + region, data=df_freq)
summary(m)

# band 80-190Hz
i = (gdf$aprop == 'meanspect') & (gdf$center_freq > 0) & (gdf$decomp == 'full_psds') & (gdf$band == 2)
df_freq = subset(gdf, i)
m = lm(center_freq ~ dist_l2a + dist_midline + region, data=df_freq)
summary(m)
  
# spike rate 
i = (gdf$aprop == 'meanspect') & (gdf$center_freq > 0) & (gdf$decomp == 'spike_rate')
df_freq = subset(gdf, i)
m = lm(center_freq ~ dist_l2a + dist_midline, data=df_freq)
summary(m)
