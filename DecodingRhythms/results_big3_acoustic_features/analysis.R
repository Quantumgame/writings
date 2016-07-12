library(splines)
library(car)

# df = read.csv('/auto/tdrive/mschachter/data/aggregate/acoustic_big3.csv')
df = read.csv('/auto/tdrive/mschachter/data/aggregate/big3_spike_rate.csv')
df$cell = factor(df$cell)
summary(df)


fit_all_cells = function()
{
  ncells = length(unique(df$cell))
  for (k in 1:ncells)
  {
    cat('\n')
    cat(sprintf('Cell %d\n', k-1))

    formulas = c("spike_rate ~ maxAmp",
                 "spike_rate ~ sal",
                 "spike_rate ~ meanspect",
                 "spike_rate ~ maxAmp + meanspect",
                 "spike_rate ~ maxAmp + meanspect + sal",
                 "spike_rate ~ maxAmp + meanspect + sal + call_type",
                 "spike_rate ~ maxAmp:meanspect",
                 "spike_rate ~ maxAmp:meanspect:sal",
                 "spike_rate ~ maxAmp:meanspect:sal + call_type",
                 "spike_rate ~ ns(maxAmp, 3) + ns(meanspect, 3) + ns(sal, 3)",
                 "spike_rate ~ ns(maxAmp, 3) + ns(meanspect, 3) + ns(sal, 3) + call_type"
                )

    for (j in 1:length(formulas))
    {
      fstr = formulas[j]
      m = lm(formula(fstr), data=subset(df, df$cell == k-1))
      s = summary(m)
      cat(sprintf('\t%s\n', fstr))
      cat(sprintf('\t\tAdjusted R2: %0.2f\n', s$adj.r.squared))
    }
  }
}





