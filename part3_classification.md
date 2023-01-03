Final Project Part 3: Classification
================
Nick Tedesco
2022-12-13

## Package and Data Loading

``` r
library(ggplot2)
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ## ✔ tibble  3.1.8      ✔ dplyr   1.0.10
    ## ✔ tidyr   1.2.1      ✔ stringr 1.4.1 
    ## ✔ readr   2.1.3      ✔ forcats 0.5.2 
    ## ✔ purrr   0.3.5      
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(caret)
```

    ## Loading required package: lattice
    ## 
    ## Attaching package: 'caret'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.
    ## 
    ## Attaching package: 'pROC'
    ## 
    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
data <- read.csv('/Users/nick/Documents/GSPH/INFSCI 2595/fall2022_finalproject.csv')

head(data)
```

    ##         x1       x2       x3       x4       v1       v2       v3       v4
    ## 1 0.025878 0.255934 0.492830 0.012770 0.275651 0.033657 1.166214 0.408402
    ## 2 0.030768 0.261575 0.498460 0.055779 0.343204 0.027082 1.260579 0.664248
    ## 3 0.019325 0.020877 0.258360 0.012424 4.998508 0.030259 1.298285 0.412870
    ## 4 0.306212 0.033379 0.255385 0.056190 5.090153 0.052342 1.322005 0.652111
    ## 5 0.031296 0.259342 0.264387 0.056594 5.031107 0.517705 1.368195 0.533701
    ## 6 0.031073 0.027119 0.260915 0.055192 9.977407 0.532436 1.298797 0.857509
    ##         v5 m output
    ## 1 0.525226 A  0.786
    ## 2 2.866343 A  0.730
    ## 3 0.409007 A  0.996
    ## 4 0.861594 A  0.326
    ## 5 6.451933 A  0.735
    ## 6 0.958574 A  0.954

``` r
data <- data %>% 
  mutate(
    x5 = 1 - (x1 + x2 + x3 + x4), 
    w = x2 / (x3 + x4), 
    z = (x1 + x2) / (x4 + x5), 
    t = v1 * v2, 
    y = boot::logit(output), 
    outcome = ifelse(output < 0.33, 'event', 'non_event'),
    outcome = factor(outcome, levels = c("event", "non_event"))
  )

head(data)
```

    ##         x1       x2       x3       x4       v1       v2       v3       v4
    ## 1 0.025878 0.255934 0.492830 0.012770 0.275651 0.033657 1.166214 0.408402
    ## 2 0.030768 0.261575 0.498460 0.055779 0.343204 0.027082 1.260579 0.664248
    ## 3 0.019325 0.020877 0.258360 0.012424 4.998508 0.030259 1.298285 0.412870
    ## 4 0.306212 0.033379 0.255385 0.056190 5.090153 0.052342 1.322005 0.652111
    ## 5 0.031296 0.259342 0.264387 0.056594 5.031107 0.517705 1.368195 0.533701
    ## 6 0.031073 0.027119 0.260915 0.055192 9.977407 0.532436 1.298797 0.857509
    ##         v5 m output       x5          w          z           t          y
    ## 1 0.525226 A  0.786 0.212588 0.50619858 1.25050808 0.009277586  1.3009808
    ## 2 2.866343 A  0.730 0.153418 0.47195344 1.39745312 0.009294651  0.9946226
    ## 3 0.409007 A  0.996 0.689014 0.07709835 0.05731369 0.151249854  5.5174529
    ## 4 0.861594 A  0.326 0.348834 0.10712990 0.83844661 0.266428788 -0.7263327
    ## 5 6.451933 A  0.735 0.388381 0.80796683 0.65315580 2.604629249  1.0201407
    ## 6 0.958574 A  0.954 0.625701 0.08579057 0.08546424 5.312330673  3.0320223
    ##     outcome
    ## 1 non_event
    ## 2 non_event
    ## 3 non_event
    ## 4     event
    ## 5 non_event
    ## 6 non_event

## Part 3: Classification

In this section, we will again evaluate our models using test set
performance (accuracy, AUC). We will use the same train/test split that
was used in the Regression section. Let’s start by generating our data.

``` r
## set seed for reproducibility
set.seed(15213)

## generate samples
sub_sample <- sample(nrow(data), size = nrow(data)*0.80)

train <- data[sub_sample, ]
test <- data[-sub_sample, ]

## subset to base set, 
class_base_train <- train %>% select(x1:m, outcome) 
class_base_test <- test %>% select(x1:m, outcome) 

## subset to expanded set, 
class_expanded_train <- train %>% select(x1:m, x5:t, outcome) 
class_expanded_test <- test %>% select(x1:m, x5:t, outcome) 
```

Now, let’s fit the models. First, fit the three models using the base
feature set.

``` r
## all linear additive features
class_baseMod1 <- glm(formula = outcome ~ ., family = "binomial", data = class_base_train)

## interaction of the categorical input with all continuous inputs
class_baseMod2 <- glm(formula = outcome ~ m * ., family = "binomial", data = class_base_train)

## all pair-wise interactions of the continuous inputs
class_baseMod3 <- glm(formula = outcome ~ .^2, family = "binomial", data = (class_base_train %>% select(-m)))
```

Next, fit the three models using the expanded feature set.

``` r
## linear additive features
class_expandedMod1 <- glm(formula = outcome ~ ., family = "binomial", data = class_expanded_train)

## interaction of the categorical input with continuous features
class_expandedMod2 <- glm(formula = outcome ~ m * ., family = "binomial", data = class_expanded_train)

## pair-wise interactions between the continuous features
class_expandedMod3 <- glm(formula = outcome ~ .^2, family = "binomial", data = (class_expanded_train %>% select(-m)))
```

Finally, fit the three models using linear basis functions.

``` r
class_basisMod1 <- glm(formula = outcome ~ m + (x1 + I(x1^2)) * (x2 + x3 + x4 + z + I(z^2)) + 
                                 v1 + v2 + v3 + v4 + v5 + w + I(w^2), 
                       family = "binomial", data = class_expanded_train)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
class_basisMod2 <- glm(formula = outcome ~ m + (x1 + I(x1^2) + I(x1^3)) * (x2 + x3 + x4 + z + I(z^2)) + 
                                 w + v1 + v2 + v3 + v4 + v5, 
                       family = "binomial", data = class_expanded_train)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
class_basisMod3 <- glm(formula = outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                                 (v1 + v2 + v3 + v4 + v5), 
                       family = "binomial", data = class_expanded_train)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

I received the following warning after fitting the three basis models:
Warning: glm.fit: fitted probabilities numerically 0 or 1
occurredWarning: glm.fit: fitted probabilities numerically 0 or 1
occurredWarning: glm.fit: fitted probabilities numerically 0 or 1
occurred.

This simply means that some of the observations in our models have fit
(predicted probabilities) of almost exactly 0 or 1. After doing some
Googling, it looks like we can simply ignore this warning and proceed.

``` r
class_base_pred1 <- predict(class_baseMod1, newdata = class_base_test, type = 'response')
class_base_auc1 <- auc(class_base_test$outcome, as.numeric(class_base_pred1))
```

    ## Setting levels: control = event, case = non_event

    ## Setting direction: controls < cases

``` r
class_base_pred2 <- predict(class_baseMod2, newdata = class_base_test, type = 'response')
class_base_auc2 <- auc(class_base_test$outcome, as.numeric(class_base_pred2))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_base_pred3 <- predict(class_baseMod3, newdata = class_base_test, type = 'response')
class_base_auc3 <- auc(class_base_test$outcome, as.numeric(class_base_pred3))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_expanded_pred1 <- predict(class_expandedMod1, newdata = class_expanded_test, type = 'response')
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
class_expanded_auc1 <- auc(class_expanded_test$outcome, as.numeric(class_expanded_pred1))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_expanded_pred2 <- predict(class_expandedMod2, newdata = class_expanded_test, type = 'response')
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
class_expanded_auc2 <- auc(class_expanded_test$outcome, as.numeric(class_expanded_pred2))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_expanded_pred3 <- predict(class_expandedMod3, newdata = class_expanded_test, type = 'response')
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
class_expanded_auc3 <- auc(class_expanded_test$outcome, as.numeric(class_expanded_pred3))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_basis_pred1 <- predict(class_basisMod1, newdata = class_expanded_test, type = 'response')
class_basis_auc1 <- auc(class_expanded_test$outcome, as.numeric(class_basis_pred1))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_basis_pred2 <- predict(class_basisMod2, newdata = class_expanded_test, type = 'response')
class_basis_auc2 <- auc(class_expanded_test$outcome, as.numeric(class_basis_pred2))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_basis_pred3 <- predict(class_basisMod3, newdata = class_expanded_test, type = 'response')
class_basis_auc3 <- auc(class_expanded_test$outcome, as.numeric(class_basis_pred3))
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls < cases

``` r
class_auc_df <- data.frame(model = c("class_baseMod1", "class_baseMod2", "class_baseMod3", 
                                     "class_expandedMod1", "class_expandedMod2", "class_expandedMod3", 
                                     "class_basisMod1", "class_basisMod2", "class_basisMod3"),
                           auc = c(class_base_auc1, class_base_auc2, class_base_auc3, 
                                   class_expanded_auc1, class_expanded_auc2, class_expanded_auc3, 
                                   class_basis_auc1, class_basis_auc2, class_basis_auc3))

class_auc_df %>% ggplot() + 
  geom_point(aes(x = as.factor(model), y = auc)) + 
  xlab("Model") + 
  ylab("AUC") + 
  ggtitle("Test Set AUC for each of the Nine Logistic Regression Models") + 
  theme_bw() +
  theme(axis.text.x = element_text(face = "bold", angle = 90))
```

![](part3_classification_files/figure-gfm/Classification%20Model%20Test%20Set%20AUC-1.png)<!-- -->

Similar to the regression section, the three basis models resulted in
the best test set performance (in terms of AUC). class_basisMod3 was the
best at predicting the outcome.

Now, lets take a look at the coefficient summary for these three models.

``` r
summary(class_basisMod3)
```

    ## 
    ## Call:
    ## glm(formula = outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + 
    ##     x3 + x4 + z + I(z^2) + w) + (v1 + v2 + v3 + v4 + v5), family = "binomial", 
    ##     data = class_expanded_train)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -2.31325  -0.52569   0.09796   0.53822   2.53254  
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)     4.746e+02  5.086e+02   0.933   0.3507  
    ## mB             -4.263e-01  3.135e-01  -1.360   0.1739  
    ## mC             -2.031e-01  3.051e-01  -0.666   0.5055  
    ## mD             -2.843e-01  3.058e-01  -0.929   0.3526  
    ## mE              2.320e-01  3.126e-01   0.742   0.4580  
    ## x1             -1.636e+03  2.261e+03  -0.724   0.4693  
    ## sin(x1)         1.594e+03  2.215e+03   0.720   0.4718  
    ## cos(x1)        -4.738e+02  5.057e+02  -0.937   0.3488  
    ## x2             -2.264e+03  2.463e+03  -0.919   0.3581  
    ## x3              9.182e+02  1.028e+03   0.893   0.3719  
    ## x4             -4.125e+02  2.228e+03  -0.185   0.8531  
    ## z               6.800e+02  4.054e+02   1.677   0.0935 .
    ## I(z^2)         -1.639e+02  9.117e+01  -1.798   0.0722 .
    ## w               1.014e+03  7.597e+02   1.335   0.1818  
    ## v1             -9.381e-02  3.661e-02  -2.562   0.0104 *
    ## v2             -1.278e-01  3.751e-01  -0.341   0.7332  
    ## v3             -8.470e-02  3.979e-02  -2.129   0.0333 *
    ## v4              1.693e-01  4.054e-01   0.418   0.6762  
    ## v5             -4.260e-02  2.918e-02  -1.460   0.1443  
    ## x1:x2           1.128e+03  6.997e+03   0.161   0.8719  
    ## x1:x3          -4.159e+03  3.492e+03  -1.191   0.2335  
    ## x1:x4           1.442e+03  7.091e+03   0.203   0.8389  
    ## x1:z           -2.084e+03  1.256e+03  -1.659   0.0971 .
    ## x1:I(z^2)       6.144e+02  3.315e+02   1.853   0.0639 .
    ## x1:w           -2.603e+03  2.269e+03  -1.147   0.2514  
    ## sin(x1):x2     -5.592e+02  6.693e+03  -0.084   0.9334  
    ## sin(x1):x3      4.013e+03  3.387e+03   1.185   0.2362  
    ## sin(x1):x4     -1.436e+03  6.810e+03  -0.211   0.8330  
    ## sin(x1):z       1.985e+03  1.219e+03   1.629   0.1034  
    ## sin(x1):I(z^2) -5.938e+02  3.229e+02  -1.839   0.0659 .
    ## sin(x1):w       2.430e+03  2.165e+03   1.122   0.2618  
    ## cos(x1):x2      2.183e+03  2.434e+03   0.897   0.3699  
    ## cos(x1):x3     -8.991e+02  1.019e+03  -0.882   0.3778  
    ## cos(x1):x4      4.295e+02  2.209e+03   0.194   0.8459  
    ## cos(x1):z      -6.743e+02  4.020e+02  -1.677   0.0934 .
    ## cos(x1):I(z^2)  1.632e+02  9.055e+01   1.803   0.0714 .
    ## cos(x1):w      -9.911e+02  7.512e+02  -1.319   0.1870  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1300.65  on 1000  degrees of freedom
    ## Residual deviance:  721.31  on  964  degrees of freedom
    ## AIC: 795.31
    ## 
    ## Number of Fisher Scoring iterations: 9

``` r
summary(class_basisMod2)
```

    ## 
    ## Call:
    ## glm(formula = outcome ~ m + (x1 + I(x1^2) + I(x1^3)) * (x2 + 
    ##     x3 + x4 + z + I(z^2)) + w + v1 + v2 + v3 + v4 + v5, family = "binomial", 
    ##     data = class_expanded_train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.3958  -0.5262   0.1196   0.5471   2.5527  
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)       5.73753    2.54033   2.259  0.02391 *  
    ## mB               -0.40392    0.31176  -1.296  0.19511    
    ## mC               -0.19034    0.30483  -0.624  0.53235    
    ## mD               -0.25661    0.30547  -0.840  0.40087    
    ## mE                0.25121    0.31206   0.805  0.42082    
    ## x1              -90.21823   43.56050  -2.071  0.03835 *  
    ## I(x1^2)         351.72361  221.82741   1.586  0.11284    
    ## I(x1^3)        -336.67867  351.36220  -0.958  0.33796    
    ## x2              -22.64201   13.16478  -1.720  0.08545 .  
    ## x3                4.90222    6.01036   0.816  0.41471    
    ## x4               -4.74546   17.49449  -0.271  0.78620    
    ## z                 1.53220    4.04935   0.378  0.70515    
    ## I(z^2)           -0.20103    0.79056  -0.254  0.79927    
    ## w                 6.40571    1.34534   4.761 1.92e-06 ***
    ## v1               -0.09650    0.03654  -2.641  0.00827 ** 
    ## v2               -0.17427    0.37273  -0.468  0.64011    
    ## v3               -0.08216    0.03961  -2.074  0.03804 *  
    ## v4                0.18520    0.40358   0.459  0.64632    
    ## v5               -0.04365    0.02910  -1.500  0.13365    
    ## x1:x2            13.78813  168.78345   0.082  0.93489    
    ## x1:x3            16.52699   85.56890   0.193  0.84685    
    ## x1:x4           238.19506  257.63327   0.925  0.35520    
    ## x1:z            -65.29023   56.68382  -1.152  0.24939    
    ## x1:I(z^2)        16.45384   11.64482   1.413  0.15766    
    ## I(x1^2):x2      288.98478  556.51982   0.519  0.60357    
    ## I(x1^2):x3      -97.98542  308.12088  -0.318  0.75048    
    ## I(x1^2):x4     -898.15392  986.84196  -0.910  0.36275    
    ## I(x1^2):z       269.35724  200.00051   1.347  0.17805    
    ## I(x1^2):I(z^2)  -74.25748   45.90652  -1.618  0.10575    
    ## I(x1^3):x2     -759.68777  533.17247  -1.425  0.15420    
    ## I(x1^3):x3     -103.32582  365.75536  -0.282  0.77756    
    ## I(x1^3):x4      840.36507 1061.42908   0.792  0.42852    
    ## I(x1^3):z      -321.21533  209.34403  -1.534  0.12493    
    ## I(x1^3):I(z^2)  100.50471   55.96186   1.796  0.07250 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1300.65  on 1000  degrees of freedom
    ## Residual deviance:  725.22  on  967  degrees of freedom
    ## AIC: 793.22
    ## 
    ## Number of Fisher Scoring iterations: 9

``` r
summary(class_basisMod1)
```

    ## 
    ## Call:
    ## glm(formula = outcome ~ m + (x1 + I(x1^2)) * (x2 + x3 + x4 + 
    ##     z + I(z^2)) + v1 + v2 + v3 + v4 + v5 + w + I(w^2), family = "binomial", 
    ##     data = class_expanded_train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5041  -0.5433   0.1017   0.5544   2.3670  
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)       3.45593    1.51175   2.286 0.022252 *  
    ## mB               -0.47074    0.30741  -1.531 0.125692    
    ## mC               -0.20963    0.29964  -0.700 0.484167    
    ## mD               -0.27708    0.30084  -0.921 0.357036    
    ## mE                0.20285    0.30559   0.664 0.506823    
    ## x1              -50.94514   15.80696  -3.223 0.001269 ** 
    ## I(x1^2)         145.85898   39.65895   3.678 0.000235 ***
    ## x2              -15.61993    9.56158  -1.634 0.102340    
    ## x3               12.94378    3.93442   3.290 0.001002 ** 
    ## x4                5.49088   10.91112   0.503 0.614798    
    ## z                -4.56058    2.67440  -1.705 0.088144 .  
    ## I(z^2)            1.22134    0.54736   2.231 0.025660 *  
    ## v1               -0.09711    0.03600  -2.697 0.006991 ** 
    ## v2               -0.08700    0.36387  -0.239 0.811023    
    ## v3               -0.07614    0.03873  -1.966 0.049305 *  
    ## v4                0.19720    0.40225   0.490 0.623969    
    ## v5               -0.03864    0.02862  -1.350 0.176961    
    ## w                 7.34265    2.09121   3.511 0.000446 ***
    ## I(w^2)           -1.07306    1.39944  -0.767 0.443214    
    ## x1:x2           -67.21655   51.30941  -1.310 0.190188    
    ## x1:x3          -105.31026   29.25704  -3.599 0.000319 ***
    ## x1:x4            51.27507   84.89210   0.604 0.545842    
    ## x1:z             39.03749   18.16566   2.149 0.031637 *  
    ## x1:I(z^2)        -8.16250    4.13958  -1.972 0.048630 *  
    ## I(x1^2):x2      285.51147   78.73691   3.626 0.000288 ***
    ## I(x1^2):x3      209.72096   51.59496   4.065 4.81e-05 ***
    ## I(x1^2):x4     -134.47921  144.98217  -0.928 0.353637    
    ## I(x1^2):z      -120.75110   29.55910  -4.085 4.41e-05 ***
    ## I(x1^2):I(z^2)   24.81878    7.93914   3.126 0.001771 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1300.65  on 1000  degrees of freedom
    ## Residual deviance:  732.68  on  972  degrees of freedom
    ## AIC: 790.68
    ## 
    ## Number of Fisher Scoring iterations: 9

Since so many of the features are insignificant across the three models,
it is relatively difficult to determine which features are important in
predicting the binary outcome. However, we see that x1, w, v1, and v3
have significant features in two of the three models. According to
class_basisMod3, z also seems to be an important input. class_basisMod3
also values the interactions between I(x1^2) and many of the other x
variables (x2, x3), as well as z and I(z^2).

### iiiB) Bayesian Logistic Regression Models

First, we must define our functions for using Bayesian Logistic
Regression with Laplace Approximation.

``` r
logistic_logpost <- function(unknowns, my_info)
{
  # extract the design matrix and assign to X
  X <- my_info$design_matrix
  
  # calculate the linear predictor
  eta <- as.vector(X %*% as.matrix(unknowns))
  
  # calculate the event probability
  mu <- boot::inv.logit(eta)
  
  # evaluate the log-likelihood
  log_lik <- sum(dbinom(x = my_info$yobs, 
                        size = 1,
                        prob = mu,
                        log = TRUE))
  
  # evaluate the log-prior
  log_prior <- sum(dnorm(x = unknowns,
                         mean = my_info$mu_beta,
                         sd = my_info$tau_beta,
                         log = TRUE))
  
  # sum together
  log_lik + log_prior
  
}
```

``` r
my_laplace <- function(start_guess, logpost_func, ...)
{
  # code adapted from the `LearnBayes`` function `laplace()`
  fit <- optim(start_guess,
               logpost_func,
               gr = NULL,
               ...,
               method = "BFGS",
               hessian = TRUE,
               control = list(fnscale = -1, maxit = 1001))
  
  mode <- fit$par
  post_var_matrix <- -solve(fit$hessian)
  p <- length(mode)
  int <- p/2 * log(2 * pi) + 0.5 * log(det(post_var_matrix)) + logpost_func(mode, ...)
  # package all of the results into a list
  list(mode = mode,
       var_matrix = post_var_matrix,
       log_evidence = int,
       converge = ifelse(fit$convergence == 0,
                         "YES", 
                         "NO"),
       iter_counts = as.numeric(fit$counts[1]))
}
```

``` r
generate_glm_post_samples <- function(mvn_result, num_samples)
{
  # specify the number of unknown beta parameters
  length_beta <- length(mvn_result$mode)
  
  # generate the random samples
  beta_samples <- MASS::mvrnorm(n = num_samples, 
                                mu = mvn_result$mode,
                                Sigma = mvn_result$var_matrix)
  
  # change the data type and name
  beta_samples %>% 
    as.data.frame() %>% tibble::as_tibble() %>% 
    purrr::set_names(sprintf("beta_%02d", (1:length_beta) - 1))
}
```

``` r
post_logistic_pred_samples <- function(Xnew, Bmat)
{
  # calculate the linear predictor at all prediction points and posterior samples
  eta_mat <- Xnew %*% t(Bmat)
  
  # calculate the event probability
  mu_mat <- boot::inv.logit(eta_mat)
  
  # book keeping
  list(eta_mat = eta_mat, mu_mat = mu_mat)
}
```

``` r
summarize_logistic_pred_from_laplace <- function(mvn_result, Xtest, num_samples)
{
  # generate posterior samples of the beta parameters
  betas <- generate_glm_post_samples(mvn_result, num_samples)
  
  # data type conversion
  betas <- as.matrix(betas)
  
  # make posterior predictions on the test set
  pred_test <- post_logistic_pred_samples(Xtest, betas)
  
  # calculate summary statistics on the posterior predicted probability
  # summarize over the posterior samples
  
  # posterior mean, should you summarize along rows (rowMeans) or 
  # summarize down columns (colMeans) ???
  mu_avg <- rowMeans(pred_test$mu_mat)
  
  # posterior quantiles
  mu_q05 <- apply(pred_test$mu_mat, 1, stats::quantile, probs = 0.05)
  mu_q95 <- apply(pred_test$mu_mat, 1, stats::quantile, probs = 0.95)
  
  # book keeping
  tibble::tibble(
    mu_avg = mu_avg,
    mu_q05 = mu_q05,
    mu_q95 = mu_q95
  ) %>% 
    tibble::rowid_to_column("pred_id")
}
```

For the next part part, I will refit the best model from iA
(class_basisMod3). I will also refit class_expandedMod3 - I chose this
model because it seems like the interactions between continuous features
(especially when including those in the expanded feature set) are
important for predicting the outcome. This is evident from both test set
AUC and the model summaries.

(In other words, I’m fitting the same two models from the regression
part, but their classification counterparts! It seems like the
classification section is going very similarly to the regression
section).

First, we must make the outcome binary to comply with our functions.

``` r
binary.outcome <- case_when(class_expanded_train$outcome == "event" ~ 1, TRUE ~ 0)
```

Now, we can fit the models

``` r
class_bayes_X01 <- model.matrix(outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                               (v1 + v2 + v3 + v4 + v5), data = class_expanded_train)

class_bayes_info01 <- list(
  yobs = binary.outcome,
  design_matrix = class_bayes_X01,
  mu_beta = 0,
  tau_beta = 5
)
```

``` r
class_bayes_laplace01 <- my_laplace(rep(0, ncol(class_bayes_X01)), logistic_logpost, class_bayes_info01)
class_bayes_laplace01 %>% glimpse()
```

    ## List of 5
    ##  $ mode        : num [1:37] -4.5488 0.2553 0.0324 0.3098 -0.2935 ...
    ##  $ var_matrix  : num [1:37, 1:37] 12.2119 -0.01351 -0.01414 -0.01143 -0.00826 ...
    ##  $ log_evidence: num -511
    ##  $ converge    : chr "YES"
    ##  $ iter_counts : num 97

``` r
class_bayes_X02 <- model.matrix(outcome ~ .^2, data = (class_expanded_train %>% select(-m)))

class_bayes_info02 <- list(
  yobs = binary.outcome,
  design_matrix = class_bayes_X02,
  mu_beta = 0,
  tau_beta = 5
)
```

``` r
class_bayes_laplace02 <- my_laplace(rep(0, ncol(class_bayes_X02)), logistic_logpost, class_bayes_info02)
class_bayes_laplace02 %>% glimpse()
```

    ## List of 5
    ##  $ mode        : num [1:92] -0.0649 2.4588 0.8062 0.0395 -0.2347 ...
    ##  $ var_matrix  : num [1:92, 1:92] 7.69 -2.57 -1.11 -5.5 -1.69 ...
    ##  $ log_evidence: num -633
    ##  $ converge    : chr "YES"
    ##  $ iter_counts : num 216

Once again, we will use test set AUC to evaluate the models. First,
let’s generate our posterior predictions.

``` r
class_bayes_testX01 <- model.matrix(outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                      (v1 + v2 + v3 + v4 + v5), data = class_expanded_test)

class_bayes_pred01 <- summarize_logistic_pred_from_laplace(class_bayes_laplace01, class_bayes_testX01, 5000)
class_bayes_pred01 %>% glimpse()
```

    ## Rows: 251
    ## Columns: 4
    ## $ pred_id <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,…
    ## $ mu_avg  <dbl> 0.4941590947, 0.6317977244, 0.3810324339, 0.0044696843, 0.0168…
    ## $ mu_q05  <dbl> 3.270497e-01, 5.300638e-01, 2.697784e-01, 7.670045e-04, 5.0109…
    ## $ mu_q95  <dbl> 0.662197666, 0.725944685, 0.495988935, 0.012723460, 0.03878900…

``` r
class_bayes_testX02 <- model.matrix(outcome ~ .^2, data = (class_expanded_test %>% select(-m)))

class_bayes_pred02 <- summarize_logistic_pred_from_laplace(class_bayes_laplace02, class_bayes_testX02, 5000)
class_bayes_pred02 %>% glimpse()
```

    ## Rows: 251
    ## Columns: 4
    ## $ pred_id <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,…
    ## $ mu_avg  <dbl> 0.515289566, 0.715401677, 0.466286590, 0.016292847, 0.02036744…
    ## $ mu_q05  <dbl> 0.1868748657, 0.6254883420, 0.3250261943, 0.0055351354, 0.0044…
    ## $ mu_q95  <dbl> 0.835142626, 0.795824754, 0.607404815, 0.035192061, 0.05359398…

Now, calculate AUC by comparing mu_avg to the actual outcome.

``` r
class_bayes_auc1 <- auc(class_expanded_test$outcome, class_bayes_pred01$mu_avg)
```

    ## Setting levels: control = event, case = non_event

    ## Setting direction: controls > cases

``` r
class_bayes_auc2 <- auc(class_expanded_test$outcome, class_bayes_pred02$mu_avg)
```

    ## Setting levels: control = event, case = non_event
    ## Setting direction: controls > cases

``` r
class_bayes_auc_df <- data.frame(model = c("class_bayesMod1", "class_bayesMod2"),
                          auc = c(class_bayes_auc1, class_bayes_auc2))

class_bayes_auc_df %>% ggplot() + 
  geom_point(aes(x = as.factor(model), y = auc)) + 
  xlab("Model") + 
  ylab("AUC") + 
  ggtitle("Test Set AUC for each of the Two Bayesian Regression Models") + 
  theme_bw() +
  theme(axis.text.x = element_text(face = "bold", angle = 90))
```

![](part3_classification_files/figure-gfm/Bayesian%20Logistic%20Model%20AUC-1.png)<!-- -->

The best of the two models was identified as class_bayesMod2, as
according to test set AUC Next, let’s show the regression coefficient
posterior summary statistics for our best model.

``` r
class_bayes_summary <- function(laplace_object, design_matrix){
  
  names <- c(colnames(design_matrix))
  modes <- laplace_object$mode
  sd <- sqrt(diag(laplace_object$var_matrix))
  
  df <- data.frame(Parameter = names, Estimate = modes, sdev = sd)
  
  df
    
}
```

``` r
class_bayes_summary02 <- class_bayes_summary(class_bayes_laplace02, class_bayes_X02)
class_bayes_summary02
```

    ##      Parameter      Estimate       sdev
    ## 1  (Intercept) -0.0648593415 2.77392057
    ## 2           x1  2.4588058894 3.90172840
    ## 3           x2  0.8061551765 4.57375042
    ## 4           x3  0.0395081426 3.68019425
    ## 5           x4 -0.2346980471 4.63450306
    ## 6           v1  0.7846400063 2.06453946
    ## 7           v2  0.5152789351 2.87554483
    ## 8           v3 -0.0301306132 2.05517615
    ## 9           v4 -0.6108304922 2.77620043
    ## 10          v5 -0.1343009029 2.05363576
    ## 11          x5 -3.1345868086 3.40383659
    ## 12           w  0.5541786522 2.84164006
    ## 13           z -1.1767195195 2.44250387
    ## 14           t  0.0811847942 3.70270875
    ## 15       x1:x2 11.7369198719 4.64441246
    ## 16       x1:x3 19.3899557930 3.94402256
    ## 17       x1:x4  0.5621024509 4.87802920
    ## 18       x1:v1  0.4296100798 2.08396526
    ## 19       x1:v2  1.0147232980 3.60621039
    ## 20       x1:v3  0.1552232400 2.06367855
    ## 21       x1:v4  0.8020874927 3.16613992
    ## 22       x1:v5  0.0226993849 2.05817043
    ## 23       x1:x5 -6.5249456993 4.24841486
    ## 24        x1:w  0.4815392803 3.43144019
    ## 25        x1:z -5.3456572302 2.53407098
    ## 26        x1:t  0.2365429230 2.24411517
    ## 27       x2:x3 -8.4137322767 4.71732027
    ## 28       x2:x4 -0.1560667616 4.95179063
    ## 29       x2:v1  1.4406191413 2.39036995
    ## 30       x2:v2 -0.4549266424 4.55125867
    ## 31       x2:v3  0.1560684176 2.22794688
    ## 32       x2:v4  1.0435237767 4.44068366
    ## 33       x2:v5 -0.4008210058 2.20378589
    ## 34       x2:x5  1.0302470623 4.78874290
    ## 35        x2:w  5.0857709227 3.60969875
    ## 36        x2:z -2.6564826161 3.36339166
    ## 37        x2:t  0.7317025838 2.68454308
    ## 38       x3:x4 -0.3420366616 4.89594497
    ## 39       x3:v1 -0.6055309950 2.11774011
    ## 40       x3:v2 -0.2779102243 3.70772749
    ## 41       x3:v3 -0.3163056223 2.08681548
    ## 42       x3:v4 -3.4675874280 3.48623815
    ## 43       x3:v5  0.2250380724 2.07852354
    ## 44       x3:x5 -0.0568934844 4.01780351
    ## 45        x3:w  1.5265191691 4.39285563
    ## 46        x3:z  3.5627918657 2.73919314
    ## 47        x3:t -0.4228414036 2.27850539
    ## 48       x4:v1 -0.3418515543 2.40506509
    ## 49       x4:v2 -1.6526204846 4.74553703
    ## 50       x4:v3 -0.1974148985 2.22976365
    ## 51       x4:v4 -0.9581154086 4.65152979
    ## 52       x4:v5  0.2984919905 2.20696334
    ## 53       x4:x5 -0.3988065183 4.85613616
    ## 54        x4:w -0.7203636688 4.65228536
    ## 55        x4:z -0.1214423935 3.67990403
    ## 56        x4:t  0.0397053964 2.79254841
    ## 57       v1:v2  0.0811847941 3.70270973
    ## 58       v1:v3 -0.0128021146 0.01897222
    ## 59       v1:v4 -0.3595328279 0.24976391
    ## 60       v1:v5 -0.0002654865 0.01844194
    ## 61       v1:x5 -0.1348564701 2.08135583
    ## 62        v1:w -0.7717939458 0.41852974
    ## 63        v1:z -0.0518732751 0.09148607
    ## 64        v1:t -0.0598308648 0.02062475
    ## 65       v2:v3  0.1608509306 0.20095835
    ## 66       v2:v4 -2.7961608643 2.44335358
    ## 67       v2:v5  0.2277745502 0.19481945
    ## 68       v2:x5  1.8860214513 3.62495474
    ## 69        v2:w -3.0629436755 2.26548305
    ## 70        v2:z -0.2274413447 0.81684771
    ## 71        v2:t  0.1270573596 0.18987800
    ## 72       v3:v4  0.0582734638 0.15264054
    ## 73       v3:v5  0.0083612749 0.01028948
    ## 74       v3:x5  0.1807039037 2.06099917
    ## 75        v3:w -0.1336595194 0.29587392
    ## 76        v3:z  0.0354617243 0.06054190
    ## 77        v3:t -0.0067530873 0.03428244
    ## 78       v4:v5  0.0961327594 0.12223774
    ## 79       v4:x5  1.9692697866 3.28216310
    ## 80        v4:w  0.8837726661 1.80768076
    ## 81        v4:z  0.5069863772 0.56228854
    ## 82        v4:t  0.5664579202 0.42975794
    ## 83       v5:x5 -0.2692878685 2.05780499
    ## 84        v5:w  0.2683133081 0.26370437
    ## 85        v5:z -0.0289387497 0.05144327
    ## 86        v5:t -0.0263481456 0.03228211
    ## 87        x5:w -5.8192640765 3.56925694
    ## 88        x5:z  3.3864036632 3.42753179
    ## 89        x5:t -0.5034018274 2.23247240
    ## 90         w:z  0.1031620405 1.11947591
    ## 91         w:t  0.4561067091 0.60308777
    ## 92         z:t -0.0412104450 0.15351569

The standard deviations for the model coefficients are quite high
compared to their corresponding coefficient estimates. This is
consistent with the multicolinearity issues we discussed earlier, as
well as the fact that logistic regression / classification tends to
produce more uncertain coefficients.

It is worth mentioning that reg_bayesMod1 was found to be better than
reg_bayesMod2 in the regression section. However, here we found that
class_bayesMod1 is worse than class_bayesMod2. Perhaps this is because
of our prior specifications - before, the prior wasn’t as big of a deal
since the regression coefficients weren’t very large in magnitude.
However, since the regression coefficients appear to be much larger in
the classification section, perhaps the prior had more of an impact on
our final Bayesian coefficients (as compared to the corresponding parts
of the regression section).

### iiC) GLM Predictions

Once again, I will use the linear (as opposed to Bayesian) models for
making predictions

Start by defining the prediction grid. We will do this in the same way
as the regression case.

``` r
class_pred_viz_grid <- expand.grid(x1 = seq(min(data$x1), max(data$x1), length.out = 101),
                                 z = seq(min(data$z), max(data$z), length.out = 6),
                                 x2 = mean(data$x2),
                                 x3 = mean(data$x3),
                                 x4 = mean(data$x4),
                                 x5 = mean(data$x5),
                                 v1 = mean(data$v1),
                                 v2 = mean(data$v2),
                                 v3 = mean(data$v3),
                                 v4 = mean(data$v4),
                                 v5 = mean(data$v5),
                                 w = mean(data$w),
                                 t = mean(data$t),
                                 m = c("C", "D"), 
                                 KEEP.OUT.ATTRS = FALSE,
                                 stringsAsFactors = FALSE) %>% 
  as.data.frame() %>% tibble::as_tibble()
```

Now we will make the predictions and retrieve the confidence intervals.
Since the predict function cannot calculate confidence intervals for
logistic regression models, we must compute these ourselves. We will do
this by retrieving the standard error and mean predictions (not
transformed to the scale of probability).

``` r
class_linear_pred01 <- predict(class_basisMod3, newdata = class_pred_viz_grid, type = 'response')

class_linear_untransformed_pred01 <- predict(class_basisMod3, newdata = class_pred_viz_grid, se.fit = TRUE)

class01_upr_untransformed <- class_linear_untransformed_pred01$fit + (1.96 * class_linear_untransformed_pred01$se.fit)
class01_lwr_untransformed <- class_linear_untransformed_pred01$fit - (1.96 * class_linear_untransformed_pred01$se.fit)

class01_upr <- boot::inv.logit(class01_upr_untransformed)
class01_lwr <- boot::inv.logit(class01_lwr_untransformed)

class_linear_pred01_df <- cbind(class_pred_viz_grid, class_linear_pred01, class01_upr, class01_lwr)
```

``` r
class_linear_pred02 <- predict(class_expandedMod3, newdata = class_pred_viz_grid, type = 'response')
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
class_linear_untransformed_pred02 <- predict(class_expandedMod3, newdata = class_pred_viz_grid, se.fit = TRUE)
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = residual.scale, type = if
    ## (type == : prediction from a rank-deficient fit may be misleading

``` r
class02_upr_untransformed <- class_linear_untransformed_pred02$fit + (1.96 * class_linear_untransformed_pred02$se.fit)
class02_lwr_untransformed <- class_linear_untransformed_pred02$fit - (1.96 * class_linear_untransformed_pred02$se.fit)

class02_upr <- boot::inv.logit(class02_upr_untransformed)
class02_lwr <- boot::inv.logit(class02_lwr_untransformed)

class_linear_pred02_df <- cbind(class_pred_viz_grid, class_linear_pred02, class02_upr, class02_lwr)
```

Finally, visualize the results.

``` r
class_linear_pred01_df %>% ggplot(aes(x = x1)) + 
  geom_ribbon(aes(ymin = class01_lwr, ymax = class01_upr), fill = 'grey') + 
  geom_line(aes(y = class_linear_pred01)) + 
  facet_wrap(facets = ~z) + 
  coord_cartesian(ylim = c(0, 1))
```

![](part3_classification_files/figure-gfm/class_linear_pred01%20viz-1.png)<!-- -->

``` r
class_linear_pred02_df %>% ggplot(aes(x = x1)) + 
  geom_ribbon(aes(ymin = class02_lwr, ymax = class02_upr), fill = 'grey') + 
  geom_line(aes(y = class_linear_pred02)) + 
  facet_wrap(facets = ~z) + 
  coord_cartesian(ylim = c(0, 1))
```

![](part3_classification_files/figure-gfm/class_linear_pred02%20viz-1.png)<!-- -->

Once again, the predictive trends are nowhere near the same across the
two models. This makes sense, considering that we are using relatively
complex basis functions in class_basisMod3 and simple linear interaction
terms in class_expandedMod3.

It is worth noting that the confidence intervals on the predictions are
extreme for both models. Although this doesn’t look pretty, considering
the extremely large coefficient standard errors/deviations that we
observed in both the linear and Bayesian models, this makes sense…

### iiiD) Train/Tune with Resampling

First, we will prepare the caret dataset to fit our models.

``` r
class_df_caret <- data %>% 
  select(-c(y, output))

class_df_caret %>% glimpse()
```

    ## Rows: 1,252
    ## Columns: 15
    ## $ x1      <dbl> 0.025878, 0.030768, 0.019325, 0.306212, 0.031296, 0.031073, 0.…
    ## $ x2      <dbl> 0.255934, 0.261575, 0.020877, 0.033379, 0.259342, 0.027119, 0.…
    ## $ x3      <dbl> 0.492830, 0.498460, 0.258360, 0.255385, 0.264387, 0.260915, 0.…
    ## $ x4      <dbl> 0.012770, 0.055779, 0.012424, 0.056190, 0.056594, 0.055192, 0.…
    ## $ v1      <dbl> 0.275651, 0.343204, 4.998508, 5.090153, 5.031107, 9.977407, 0.…
    ## $ v2      <dbl> 0.033657, 0.027082, 0.030259, 0.052342, 0.517705, 0.532436, 1.…
    ## $ v3      <dbl> 1.166214, 1.260579, 1.298285, 1.322005, 1.368195, 1.298797, 1.…
    ## $ v4      <dbl> 0.408402, 0.664248, 0.412870, 0.652111, 0.533701, 0.857509, 0.…
    ## $ v5      <dbl> 0.525226, 2.866343, 0.409007, 0.861594, 6.451933, 0.958574, 0.…
    ## $ m       <chr> "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A…
    ## $ x5      <dbl> 0.212588, 0.153418, 0.689014, 0.348834, 0.388381, 0.625701, 0.…
    ## $ w       <dbl> 0.50619858, 0.47195344, 0.07709835, 0.10712990, 0.80796683, 0.…
    ## $ z       <dbl> 1.25050808, 1.39745312, 0.05731369, 0.83844661, 0.65315580, 0.…
    ## $ t       <dbl> 0.009277586, 0.009294651, 0.151249854, 0.266428788, 2.60462924…
    ## $ outcome <fct> non_event, non_event, non_event, event, non_event, non_event, …

Next, let’s define our metric and resampling method. We will have two
sets of evaluation metrics, and therefore train two sets of models: one
for accuracy, and one using AUC. In both cases, we will perform 10-fold
5-repeat cross validation.

``` r
my_class_ctrl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                        returnData = FALSE, classProbs = TRUE,
                        summaryFunction = twoClassSummary)

my_class_metric1 <- 'ROC'

my_class_ctrl2 <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                        returnData = FALSE, classProbs = TRUE)

my_class_metric2 <- 'Accuracy'
```

Now, let’s start fitting models. We will start by fitting the four
linear models. The seed will be set in each chunk to ensure
reproducibility.

``` r
set.seed(1234)

class_caret_lm_base_roc <- train(outcome ~ ., 
                             data = (class_df_caret %>% select(-c(x5, w, z, t))),
                             method = "glm",
                             family = "binomial",
                             metric = my_class_metric1,
                             preProcess = c("center", "scale"),
                             trControl = my_class_ctrl1)

class_caret_lm_base_acc <- train(outcome ~ ., 
                             data = (class_df_caret %>% select(-c(x5, w, z, t))),
                             method = "glm",
                             family = "binomial",
                             metric = my_class_metric2,
                             preProcess = c("center", "scale"),
                             trControl = my_class_ctrl2)
```

``` r
set.seed(1234)

class_caret_lm_expanded_roc <- train(outcome ~ ., 
                                 data = class_df_caret,
                                 method = "glm",
                                 family = "binomial",
                                 metric = my_class_metric1,
                                 preProcess = c("center", "scale"),
                                 trControl = my_class_ctrl1)

class_caret_lm_expanded_acc <- train(outcome ~ ., 
                                 data = class_df_caret,
                                 method = "glm",
                                 family = "binomial",
                                 metric = my_class_metric2,
                                 preProcess = c("center", "scale"),
                                 trControl = my_class_ctrl2)
```

``` r
set.seed(1234)

class_caret_lm_basisMod3_roc <- caret::train(outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                                         (v1 + v2 + v3 + v4 + v5), 
                                         data = class_df_caret,
                                         method = "glm",
                                         family = "binomial",
                                         metric = my_class_metric1,
                                         preProcess = c("center", "scale"),
                                         trControl = my_class_ctrl1)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
class_caret_lm_basisMod3_acc <- caret::train(outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                                         (v1 + v2 + v3 + v4 + v5), 
                                         data = class_df_caret,
                                         method = "glm",
                                         family = "binomial",
                                         metric = my_class_metric2,
                                         preProcess = c("center", "scale"),
                                         trControl = my_class_ctrl2)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
set.seed(1234)

class_caret_lm_expandedMod3_roc <- train(outcome ~ .^2, 
                                     data = class_df_caret,
                                     method = "glm",
                                     family = "binomial",
                                     metric = my_class_metric1,
                                     preProcess = c("center", "scale"),
                                     trControl = my_class_ctrl1)

class_caret_lm_expandedMod3_acc <- train(outcome ~ .^2, 
                                     data = class_df_caret,
                                     method = "glm",
                                     family = "binomial",
                                     metric = my_class_metric2,
                                     preProcess = c("center", "scale"),
                                     trControl = my_class_ctrl2)
```

Next, let’s train the two elastic net models.

``` r
set.seed(1234)

class_caret_enetMod1_roc <- train(outcome ~ m * (x1 + x2 + x3 + x4 + x5 + v1 + v2 + v3 + v4 + v5 + w + z + t)^2, 
                              data = class_df_caret,
                              method = "glmnet",
                              metric = my_class_metric1,
                              preProcess = c("center", "scale"),
                              trControl = my_class_ctrl1)

class_caret_enetMod1_acc <- train(outcome ~ m * (x1 + x2 + x3 + x4 + x5 + v1 + v2 + v3 + v4 + v5 + w + z + t)^2, 
                              data = class_df_caret,
                              method = "glmnet",
                              metric = my_class_metric2,
                              preProcess = c("center", "scale"),
                              trControl = my_class_ctrl2)
```

``` r
set.seed(1234)

class_caret_enetMod2_roc <- caret::train(outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                                          (v1 + v2 + v3 + v4 + v5), 
                                     data = class_df_caret,
                                     method = "glmnet",
                                     metric = my_class_metric1,
                                     preProcess = c("center", "scale"),
                                     trControl = my_class_ctrl1)
```

    ## Warning: from glmnet C++ code (error code -93); Convergence for 93th lambda
    ## value not reached after maxit=100000 iterations; solutions for larger lambdas
    ## returned

    ## Warning: from glmnet C++ code (error code -95); Convergence for 95th lambda
    ## value not reached after maxit=100000 iterations; solutions for larger lambdas
    ## returned

``` r
class_caret_enetMod2_acc <- caret::train(outcome ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                                          (v1 + v2 + v3 + v4 + v5), 
                                     data = class_df_caret,
                                     method = "glmnet",
                                     metric = my_class_metric2,
                                     preProcess = c("center", "scale"),
                                     trControl = my_class_ctrl2)
```

    ## Warning: from glmnet C++ code (error code -98); Convergence for 98th lambda
    ## value not reached after maxit=100000 iterations; solutions for larger lambdas
    ## returned

Now, let’s train the base and expanded models using neural networks.

``` r
set.seed(1234)

class_caret_neuralNet_base_roc <- train(outcome ~ ., 
                                    data = (class_df_caret %>% select(-c(x5, w, z, t))),
                                    method = "nnet",
                                    metric = my_class_metric1,
                                    preProcess = c("center", "scale"),
                                    trControl = my_class_ctrl1, 
                                    trace = FALSE)

class_caret_neuralNet_base_acc <- train(outcome ~ ., 
                                    data = (class_df_caret %>% select(-c(x5, w, z, t))),
                                    method = "nnet",
                                    metric = my_class_metric2,
                                    preProcess = c("center", "scale"),
                                    trControl = my_class_ctrl2, 
                                    trace = FALSE)
```

``` r
set.seed(1234)

class_caret_neuralNet_expanded_roc <- train(outcome ~ ., 
                                        data = class_df_caret,
                                        method = "nnet",
                                        metric = my_class_metric1,
                                        preProcess = c("center", "scale"),
                                        trControl = my_class_ctrl1, 
                                        trace = FALSE)

class_caret_neuralNet_expanded_acc <- train(outcome ~ ., 
                                        data = class_df_caret,
                                        method = "nnet",
                                        metric = my_class_metric2,
                                        preProcess = c("center", "scale"),
                                        trControl = my_class_ctrl2, 
                                        trace = FALSE)
```

Next, let’s train the base and expanded models using the random forest
method.

``` r
set.seed(1234)

class_caret_rf_base_roc <- train(outcome ~ ., 
                             data = (class_df_caret %>% select(-c(x5, w, z, t))),
                             method = "rf",
                             metric = my_class_metric1,
                             preProcess = c("center", "scale"),
                             trControl = my_class_ctrl1, 
                             trace = FALSE)

class_caret_rf_base_acc <- train(outcome ~ ., 
                             data = (class_df_caret %>% select(-c(x5, w, z, t))),
                             method = "rf",
                             metric = my_class_metric2,
                             preProcess = c("center", "scale"),
                             trControl = my_class_ctrl2, 
                             trace = FALSE)
```

``` r
set.seed(1234)

class_caret_rf_expanded_roc <- train(outcome ~ ., 
                                 data = class_df_caret,
                                 method = "rf",
                                 metric = my_class_metric1,
                                 preProcess = c("center", "scale"),
                                 trControl = my_class_ctrl1, 
                                 trace = FALSE)

class_caret_rf_expanded_acc <- train(outcome ~ ., 
                                 data = class_df_caret,
                                 method = "rf",
                                 metric = my_class_metric2,
                                 preProcess = c("center", "scale"),
                                 trControl = my_class_ctrl2, 
                                 trace = FALSE)
```

Next, let’s train the base and expanded models using the gradient
boosted tree method.

``` r
set.seed(1234)

class_caret_xgb_base_roc <- train(outcome ~ ., 
                              data = (class_df_caret %>% select(-c(x5, w, z, t))),
                              method = "xgbTree",
                              metric = my_class_metric1,
                              preProcess = c("center", "scale"),
                              trControl = my_class_ctrl1, 
                              trace = FALSE, 
                              verbosity = 0)

class_caret_xgb_base_acc <- train(outcome ~ ., 
                              data = (class_df_caret %>% select(-c(x5, w, z, t))),
                              method = "xgbTree",
                              metric = my_class_metric2,
                              preProcess = c("center", "scale"),
                              trControl = my_class_ctrl2, 
                              trace = FALSE, 
                              verbosity = 0)
```

``` r
set.seed(1234)

class_caret_xgb_expanded_roc <- train(outcome ~ ., 
                                  data = class_df_caret,
                                  method = "xgbTree",
                                  metric = my_class_metric1,
                                  preProcess = c("center", "scale"),
                                  trControl = my_class_ctrl1, 
                                  trace = FALSE, 
                                  verbosity = 0)

class_caret_xgb_expanded_acc <- train(outcome ~ ., 
                                  data = class_df_caret,
                                  method = "xgbTree",
                                  metric = my_class_metric2,
                                  preProcess = c("center", "scale"),
                                  trControl = my_class_ctrl2, 
                                  trace = FALSE, 
                                  verbosity = 0)
```

Lastly, we will use two methods that were not discussed in class. First,
let’s fit the base and expanded additive models with Generalized
Additive Model using Splines.

``` r
set.seed(1234)

class_caret_gamSpline_base_roc <- train(outcome ~ ., 
                                    data = (class_df_caret %>% select(-c(x5, w, z, t))),
                                    method = "gamSpline",
                                    metric = my_class_metric1,
                                    preProcess = c("center", "scale"),
                                    trControl = my_class_ctrl1, 
                                    verbosity = 0)

class_caret_gamSpline_base_acc <- train(outcome ~ ., 
                                    data = (class_df_caret %>% select(-c(x5, w, z, t))),
                                    method = "gamSpline",
                                    metric = my_class_metric2,
                                    preProcess = c("center", "scale"),
                                    trControl = my_class_ctrl2, 
                                    verbosity = 0)
```

``` r
set.seed(1234)

class_caret_gamSpline_expanded_roc <- train(outcome ~ ., 
                                        data = class_df_caret,
                                        method = "gamSpline",
                                        metric = my_class_metric1,
                                        preProcess = c("center", "scale"),
                                        trControl = my_class_ctrl1, 
                                        verbosity = 0)

class_caret_gamSpline_expanded_acc <- train(outcome ~ ., 
                                        data = class_df_caret,
                                        method = "gamSpline",
                                        metric = my_class_metric2,
                                        preProcess = c("center", "scale"),
                                        trControl = my_class_ctrl2, 
                                        verbosity = 0)
```

Now, let’s fit the base and expanded additive models using Multi-Layer
Perceptron.

``` r
set.seed(1234)

class_caret_mlp_base_roc <- train(outcome ~ ., 
                              data = (class_df_caret %>% select(-c(x5, w, z, t))),
                              method = "mlp",
                              metric = my_class_metric1,
                              preProcess = c("center", "scale"),
                              trControl = my_class_ctrl1, 
                              verbosity = 0)

class_caret_mlp_base_acc <- train(outcome ~ ., 
                              data = (class_df_caret %>% select(-c(x5, w, z, t))),
                              method = "mlp",
                              metric = my_class_metric2,
                              preProcess = c("center", "scale"),
                              trControl = my_class_ctrl2, 
                              verbosity = 0)
```

``` r
set.seed(1234)

class_caret_mlp_expanded_roc <- train(outcome ~ ., 
                                  data = class_df_caret,
                                  method = "mlp",
                                  metric = my_class_metric1,
                                  preProcess = c("center", "scale"),
                                  trControl = my_class_ctrl1, 
                                  verbosity = 0)

class_caret_mlp_expanded_acc <- train(outcome ~ ., 
                                  data = class_df_caret,
                                  method = "mlp",
                                  metric = my_class_metric2,
                                  preProcess = c("center", "scale"),
                                  trControl = my_class_ctrl2, 
                                  verbosity = 0)
```

Finally, we will visualize a dotplot comparing resampled RMSE for all of
the models trained using caret.

``` r
caret_roc_compare <- resamples(list(GLM_base = class_caret_lm_base_roc,
                                    GLM_expanded = class_caret_lm_expanded_roc,
                                    GLM_basisMod3 = class_caret_lm_basisMod3_roc,
                                    GLM_expandedMod3 = class_caret_lm_expandedMod3_roc,
                                    ENET_interactions = class_caret_enetMod1_roc,
                                    ENET_basisMod3 = class_caret_enetMod2_roc,
                                    NNET_base = class_caret_neuralNet_base_roc,
                                    NNET_expanded = class_caret_neuralNet_expanded_roc, 
                                    RF_base = class_caret_rf_base_roc, 
                                    RF_expanded = class_caret_rf_expanded_roc, 
                                    XGB_base = class_caret_xgb_base_roc, 
                                    XGB_expanded = class_caret_xgb_expanded_roc, 
                                    gamSpline_base = class_caret_gamSpline_base_roc, 
                                    gamSpline_expanded = class_caret_gamSpline_expanded_roc, 
                                    MLP_base = class_caret_mlp_base_roc, 
                                    MLP_expanded = class_caret_mlp_expanded_roc))

dotplot(caret_roc_compare, metric = 'ROC')
```

![](part3_classification_files/figure-gfm/class%20caret%20models%20ROC-1.png)<!-- -->

``` r
caret_acc_compare <- resamples(list(GLM_base = class_caret_lm_base_acc,
                                    GLM_expanded = class_caret_lm_expanded_acc,
                                    GLM_basisMod3 = class_caret_lm_basisMod3_acc,
                                    GLM_expandedMod3 = class_caret_lm_expandedMod3_acc,
                                    ENET_interactions = class_caret_enetMod1_acc,
                                    ENET_basisMod3 = class_caret_enetMod2_acc,
                                    NNET_base = class_caret_neuralNet_base_acc,
                                    NNET_expanded = class_caret_neuralNet_expanded_acc, 
                                    RF_base = class_caret_rf_base_acc, 
                                    RF_expanded = class_caret_rf_expanded_acc, 
                                    XGB_base = class_caret_xgb_base_acc, 
                                    XGB_expanded = class_caret_xgb_expanded_acc, 
                                    gamSpline_base = class_caret_gamSpline_base_acc, 
                                    gamSpline_expanded = class_caret_gamSpline_expanded_acc, 
                                    MLP_base = class_caret_mlp_base_acc, 
                                    MLP_expanded = class_caret_mlp_expanded_acc))

dotplot(caret_acc_compare, metric = 'Accuracy')
```

![](part3_classification_files/figure-gfm/class%20caret%20models%20accuracy-1.png)<!-- -->

The XGB_expanded model was the best at maximizing both AUC and accuracy.
The gamSpline_expanded model was second best in both cases.
GLM_basisMod3 was the third best in terms of AUC, whereas RF_base was
the third best for accuracy.
