library(splines)
library(car)
library(glmnet)

# df = read.csv('/auto/tdrive/mschachter/data/aggregate/acoustic_big3.csv')
df = read.csv('/auto/tdrive/mschachter/data/aggregate/big3_spike_rate.csv')
df$cell = factor(df$cell)
summary(df)


fit_all_cells = function()
{
  ncells = length(unique(df$cell))
  for (k in 1:ncells)
  {
    # cat('\n')
    # cat(sprintf('Cell %d\n', k-1))

    # formulas = c("spike_rate ~ maxAmp",
    #              "spike_rate ~ sal",
    #              "spike_rate ~ meanspect",
    #              "spike_rate ~ maxAmp + meanspect",
    #              "spike_rate ~ maxAmp + meanspect + sal",
    #              "spike_rate ~ maxAmp + meanspect + sal + call_type",
    #              "spike_rate ~ maxAmp:meanspect",
    #              "spike_rate ~ maxAmp:meanspect:sal",
    #              "spike_rate ~ maxAmp:meanspect:sal + call_type",
    #              "spike_rate ~ ns(maxAmp, 3) + ns(meanspect, 3) + ns(sal, 3)",
    #              "spike_rate ~ ns(maxAmp, 3) + ns(meanspect, 3) + ns(sal, 3) + call_type"
    #             )

    # formulas = c("spike_rate ~ maxAmp + meanspect + sal",
    #              "spike_rate ~ ns(maxAmp, 3) + ns(meanspect, 3) + ns(sal, 3)")
    # 
    # for (j in 1:length(formulas))
    # {
    #   fstr = formulas[j]
    #   m = lm(formula(fstr), data=subset(df, df$cell == k-1))
    #   s = summary(m)
    #   cat(sprintf('\t%s: %0.2f\n', fstr, s$adj.r.squared))
    # }
    
    fit_cell_glmnet(k-1, nat_spline=FALSE)
  }
}

read_cv_index = function()
{
  df = read.csv('/tmp/cv_index.csv', header=FALSE)
  
  train_i = list()
  test_i = list()
  
  for (k in 1:nrow(df))
  {
    x = data.matrix(df[k,])
    ntrain = as.numeric(x[1])
    # cat(sprintf('ntrain=%d\n', ntrain))
    train_i[[k]] = x[2:(ntrain+1)]
    test_i[[k]] = x[(ntrain+1):length(x)]
  }
  
  return(list("train"=train_i, "test"=test_i))
}


fit_cell_glmnet = function(cell_num, nat_spline=False)
{
  # cell_df = subset(df, df$cell == cell_num)

  dfcd = read.csv(sprintf('/tmp/cell_data_%d.csv', cell_num))
  y = dfcd$y
  
  if (nat_spline) {
    X = mat.or.vec(nrow(dfcd), ncol(dfcd)*3)
    X[, 1:3] = ns(dfcd$x0, 3)
    X[, 4:6] = ns(dfcd$x1, 3)
    X[, 7:9] = ns(dfcd$x2, 3)
  } else {
    num_features = ncol(dfcd)-1
    X = mat.or.vec(nrow(dfcd), num_features)
    X[, 1:num_features] = as.matrix(dfcd[, 1:num_features])
  }
  
  sz = dim(X)
  # cat(sprintf('sz[1]=%d, sz[2]=%d\n', sz[1], sz[2]))
  
  cv_sets = read_cv_index()
  nfolds = length(cv_sets$train)
  r2_glmnet = array(0, nfolds)
  r2_test = array(0, nfolds)
  for (k in 1:nfolds)
  {
    train_i = unlist(cv_sets$train[k])
    test_i = unlist(cv_sets$test[k])
    
    Xtrain = X[train_i, ]
    ytrain = y[train_i]
    
    Xtest = X[test_i, ]
    ytest = y[test_i]
    
    m = glmnet(Xtrain, ytrain)
    best_lambda = m$lambda[which.max(m$dev.ratio)]
    r2_glmnet[k] = max(m$dev.ratio)
    
    ypred = predict(m, Xtest, s=best_lambda)
    
    sst = sum((ytest - mean(ytrain))**2)
    sse = sum((ytest - ypred)**2)
    r2_test[k] = 1.0 - (sse / sst)
    
    # cat(sprintf('Fold %d, r2_glmnet=%0.2f, r2_test=%0.2f, best_alpha=%0.6f\n', k, r2_glmnet[k], r2_test[k], best_lambda))
  }
  
  mean_r2_glmnet = mean(r2_glmnet)
  mean_r2_test = mean(r2_test)
  
  cat(sprintf('Cell %d: R2(glmnet)=%0.2f, R2(test)=%0.2f\n', cell_num, mean_r2_glmnet, mean_r2_test))
}





