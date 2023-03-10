Final Project Part 2: Regression
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

## Part 2: Regression

### iiA) Linear Regression Models

Let’s start by making a simple train/test split of the data. We will
evaluate model performance by computing root mean squared error (RMSE)
on model predictions made using the test set.

``` r
## set seed for reproducibility
set.seed(15213)

## generate samples
sub_sample <- sample(nrow(data), size = nrow(data)*0.80)

train <- data[sub_sample, ]
test <- data[-sub_sample, ]

## subset to base set, 
reg_base_train <- train %>% select(x1:m, y) 
reg_base_test <- test %>% select(x1:m, y) 

## subset to expanded set, 
reg_expanded_train <- train %>% select(x1:m, x5:y) 
reg_expanded_test <- test %>% select(x1:m, x5:y) 
```

First, fit the three models using the base feature set.

``` r
## all linear additive features
reg_baseMod1 <- lm(formula = y ~ ., data = reg_base_train)

## interaction of the categorical input with all continuous inputs
reg_baseMod2 <- lm(formula = y ~ m * ., data = reg_base_train)

## all pair-wise interactions of the continuous inputs
reg_baseMod3 <- lm(formula = y ~ .^2, data = (reg_base_train %>% select(-m)))
```

Next, fit the three models using the expanded feature set.

``` r
## linear additive features
reg_expandedMod1 <- lm(formula = y ~ ., data = reg_expanded_train)

## interaction of the categorical input with continuous features
reg_expandedMod2 <- lm(formula = y ~ m * ., data = reg_expanded_train)

## pair-wise interactions between the continuous features
reg_expandedMod3 <- lm(formula = y ~ .^2, data = (reg_expanded_train %>% select(-m)))
```

Finally, fit the three models using linear basis functions.

UPDATE: It is worth noting, that for this part, I decided not to include
x5. When I was fitting each of the models, I took a look at their
coefficient summaries (even though I am only displaying the coefficient
summaries for the top three models) - I noticed that the coefficient for
x5 was often NA. I would guess that the reason behind this is that x5 is
too highly correlated with some of the other inputs (probably x1 to x4).

``` r
reg_basisMod1 <- lm(formula = y ~ m + (x1 + I(x1^2)) * (x2 + x3 + x4 + z + I(z^2)) + 
                      v1 + v2 + v3 + v4 + v5 + w + I(w^2), data = reg_expanded_train)

reg_basisMod2 <- lm(formula = y ~ m + (x1 + I(x1^2) + I(x1^3)) * (x2 + x3 + x4 + z + I(z^2)) + 
                  w + v1 + v2 + v3 + v4 + v5, data = reg_expanded_train)

reg_basisMod3 <- lm(formula = y ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                      (v1 + v2 + v3 + v4 + v5), data = reg_expanded_train)
```

Next, lets evaluate each of our models by computing test set RMSE.

``` r
reg_base_pred1 <- predict(reg_baseMod1, reg_base_test)
reg_base_rmse1 <- sqrt(mean((reg_base_pred1 - reg_base_test$y)^2))

reg_base_pred2 <- predict(reg_baseMod2, reg_base_test)
reg_base_rmse2 <- sqrt(mean((reg_base_pred2 - reg_base_test$y)^2))

reg_base_pred3 <- predict(reg_baseMod3, reg_base_test)
reg_base_rmse3 <- sqrt(mean((reg_base_pred3 - reg_base_test$y)^2))

reg_expanded_pred1 <- predict(reg_expandedMod1, reg_expanded_test)
```

    ## Warning in predict.lm(reg_expandedMod1, reg_expanded_test): prediction from a
    ## rank-deficient fit may be misleading

``` r
reg_expanded_rmse1 <- sqrt(mean((reg_expanded_pred1 - reg_expanded_test$y)^2))

reg_expanded_pred2 <- predict(reg_expandedMod2, reg_expanded_test)
```

    ## Warning in predict.lm(reg_expandedMod2, reg_expanded_test): prediction from a
    ## rank-deficient fit may be misleading

``` r
reg_expanded_rmse2 <- sqrt(mean((reg_expanded_pred2 - reg_expanded_test$y)^2))

reg_expanded_pred3 <- predict(reg_expandedMod3, reg_expanded_test)
```

    ## Warning in predict.lm(reg_expandedMod3, reg_expanded_test): prediction from a
    ## rank-deficient fit may be misleading

``` r
reg_expanded_rmse3 <- sqrt(mean((reg_expanded_pred3 - reg_expanded_test$y)^2))

reg_basis_pred1 <- predict(reg_basisMod1, reg_expanded_test)
reg_basis_rmse1 <- sqrt(mean((reg_basis_pred1 - reg_expanded_test$y)^2))

reg_basis_pred2 <- predict(reg_basisMod2, reg_expanded_test)
reg_basis_rmse2 <- sqrt(mean((reg_basis_pred2 - reg_expanded_test$y)^2))

reg_basis_pred3 <- predict(reg_basisMod3, reg_expanded_test)
reg_basis_rmse3 <- sqrt(mean((reg_basis_pred3 - reg_expanded_test$y)^2))

reg_rmse_df <- data.frame(model = c("reg_baseMod1", "reg_baseMod2", "reg_baseMod3", 
                                    "reg_expandedMod1", "reg_expandedMod2", "reg_expandedMod3", 
                                    "reg_basisMod1", "reg_basisMod2", "reg_basisMod3"),
                          rmse = c(reg_base_rmse1, reg_base_rmse2, reg_base_rmse3, 
                                   reg_expanded_rmse1, reg_expanded_rmse2, reg_expanded_rmse3, 
                                   reg_basis_rmse1, reg_basis_rmse2, reg_basis_rmse3))

reg_rmse_df %>% ggplot() + 
  geom_point(aes(x = as.factor(model), y = rmse)) + 
  xlab("Model") + 
  ylab("RMSE") + 
  ggtitle("Test Set RMSE for each of the Nine Linear Regression Models") + 
  theme_bw() +
  theme(axis.text.x = element_text(face = "bold", angle = 90))
```

![](part2_regression_files/figure-gfm/Regression%20Model%20Test%20Set%20RMSE-1.png)<!-- -->

As shown in the above figure, the three models with linear basis
functions performed best in terms of test set RMSE. reg_basisMod3
performed the absolute best.

Now, lets take a look at the coefficient summary for these three models.

``` r
summary(reg_basisMod3)
```

    ## 
    ## Call:
    ## lm(formula = y ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + 
    ##     z + I(z^2) + w) + (v1 + v2 + v3 + v4 + v5), data = reg_expanded_train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.0922 -0.6477  0.0526  0.6820  3.5252 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     1.176e+03  9.662e+01  12.175  < 2e-16 ***
    ## mB             -8.566e-02  9.978e-02  -0.859 0.390819    
    ## mC             -6.679e-02  9.816e-02  -0.680 0.496434    
    ## mD             -1.698e-01  9.956e-02  -1.705 0.088449 .  
    ## mE              7.592e-02  9.861e-02   0.770 0.441558    
    ## x1             -4.586e+03  3.708e+02 -12.367  < 2e-16 ***
    ## sin(x1)         4.459e+03  3.613e+02  12.340  < 2e-16 ***
    ## cos(x1)        -1.169e+03  9.618e+01 -12.160  < 2e-16 ***
    ## x2             -1.716e+02  4.630e+02  -0.371 0.710957    
    ## x3             -4.251e+01  2.288e+02  -0.186 0.852610    
    ## x4             -1.004e+03  5.792e+02  -1.734 0.083311 .  
    ## z              -4.341e+01  7.351e+01  -0.591 0.554948    
    ## I(z^2)          1.596e+01  1.317e+01   1.211 0.226120    
    ## w              -1.443e+02  1.318e+02  -1.095 0.273927    
    ## v1             -2.027e-02  1.150e-02  -1.762 0.078356 .  
    ## v2             -2.238e-01  1.128e-01  -1.984 0.047499 *  
    ## v3             -4.231e-02  1.229e-02  -3.444 0.000599 ***
    ## v4              1.814e-01  1.355e-01   1.339 0.180899    
    ## v5             -2.202e-02  9.391e-03  -2.344 0.019257 *  
    ## x1:x2          -3.076e+03  1.423e+03  -2.162 0.030852 *  
    ## x1:x3          -2.042e+03  8.657e+02  -2.359 0.018546 *  
    ## x1:x4           2.790e+03  1.937e+03   1.440 0.150109    
    ## x1:z            8.763e+02  1.974e+02   4.438 1.01e-05 ***
    ## x1:I(z^2)      -1.324e+02  3.863e+01  -3.428 0.000634 ***
    ## x1:w            5.091e+02  4.250e+02   1.198 0.231232    
    ## sin(x1):x2      3.170e+03  1.371e+03   2.312 0.020972 *  
    ## sin(x1):x3      2.066e+03  8.456e+02   2.443 0.014751 *  
    ## sin(x1):x4     -2.644e+03  1.871e+03  -1.413 0.157905    
    ## sin(x1):z      -8.848e+02  1.886e+02  -4.692 3.09e-06 ***
    ## sin(x1):I(z^2)  1.323e+02  3.700e+01   3.576 0.000366 ***
    ## sin(x1):w      -4.892e+02  4.094e+02  -1.195 0.232425    
    ## cos(x1):x2      1.629e+02  4.602e+02   0.354 0.723429    
    ## cos(x1):x3      4.340e+01  2.278e+02   0.191 0.848936    
    ## cos(x1):x4      9.971e+02  5.760e+02   1.731 0.083778 .  
    ## cos(x1):z       4.232e+01  7.294e+01   0.580 0.561960    
    ## cos(x1):I(z^2) -1.554e+01  1.307e+01  -1.189 0.234785    
    ## cos(x1):w       1.443e+02  1.310e+02   1.101 0.271165    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9624 on 964 degrees of freedom
    ## Multiple R-squared:  0.825,  Adjusted R-squared:  0.8185 
    ## F-statistic: 126.3 on 36 and 964 DF,  p-value: < 2.2e-16

``` r
summary(reg_basisMod2)
```

    ## 
    ## Call:
    ## lm(formula = y ~ m + (x1 + I(x1^2) + I(x1^3)) * (x2 + x3 + x4 + 
    ##     z + I(z^2)) + w + v1 + v2 + v3 + v4 + v5, data = reg_expanded_train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.1265 -0.6684  0.0457  0.6736  3.5762 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     6.061e+00  4.334e-01  13.984  < 2e-16 ***
    ## mB             -8.751e-02  9.995e-02  -0.876 0.381479    
    ## mC             -6.293e-02  9.832e-02  -0.640 0.522297    
    ## mD             -1.744e-01  9.942e-02  -1.754 0.079782 .  
    ## mE              8.336e-02  9.864e-02   0.845 0.398290    
    ## x1             -1.186e+02  7.783e+00 -15.239  < 2e-16 ***
    ## I(x1^2)         5.561e+02  3.815e+01  14.574  < 2e-16 ***
    ## I(x1^3)        -7.304e+02  5.368e+01 -13.606  < 2e-16 ***
    ## x2             -1.503e+01  2.805e+00  -5.356 1.06e-07 ***
    ## x3              2.729e+00  9.731e-01   2.805 0.005138 ** 
    ## x4             -3.671e+00  3.936e+00  -0.933 0.351199    
    ## z              -5.655e-01  7.810e-01  -0.724 0.469185    
    ## I(z^2)          3.377e-01  1.360e-01   2.484 0.013164 *  
    ## w               1.714e+00  3.291e-01   5.208 2.34e-07 ***
    ## v1             -2.056e-02  1.151e-02  -1.786 0.074393 .  
    ## v2             -2.209e-01  1.129e-01  -1.957 0.050656 .  
    ## v3             -4.422e-02  1.228e-02  -3.600 0.000334 ***
    ## v4              1.874e-01  1.356e-01   1.382 0.167254    
    ## v5             -2.160e-02  9.403e-03  -2.297 0.021826 *  
    ## x1:x2           1.702e+02  3.584e+01   4.748 2.37e-06 ***
    ## x1:x3          -2.436e+00  1.710e+01  -0.142 0.886781    
    ## x1:x4           9.983e+01  6.545e+01   1.525 0.127508    
    ## x1:z           -1.459e+01  1.058e+01  -1.379 0.168222    
    ## x1:I(z^2)       8.245e-01  1.827e+00   0.451 0.651915    
    ## I(x1^2):x2     -3.664e+02  1.180e+02  -3.105 0.001958 ** 
    ## I(x1^2):x3      9.487e+01  7.041e+01   1.348 0.178135    
    ## I(x1^2):x4     -3.158e+02  2.664e+02  -1.186 0.236098    
    ## I(x1^2):z      -2.362e+00  3.619e+01  -0.065 0.947985    
    ## I(x1^2):I(z^2)  4.744e+00  6.510e+00   0.729 0.466368    
    ## I(x1^3):x2     -1.910e+02  1.099e+02  -1.738 0.082477 .  
    ## I(x1^3):x3     -5.019e+02  9.232e+01  -5.436 6.89e-08 ***
    ## I(x1^3):x4      2.332e+02  2.955e+02   0.789 0.430283    
    ## I(x1^3):z       1.317e+02  3.273e+01   4.024 6.16e-05 ***
    ## I(x1^3):I(z^2) -1.943e+01  6.397e+00  -3.037 0.002453 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9642 on 967 degrees of freedom
    ## Multiple R-squared:  0.8238, Adjusted R-squared:  0.8178 
    ## F-statistic:   137 on 33 and 967 DF,  p-value: < 2.2e-16

``` r
summary(reg_basisMod1)
```

    ## 
    ## Call:
    ## lm(formula = y ~ m + (x1 + I(x1^2)) * (x2 + x3 + x4 + z + I(z^2)) + 
    ##     v1 + v2 + v3 + v4 + v5 + w + I(w^2), data = reg_expanded_train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.2790 -0.7475  0.0162  0.7420  4.5776 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     2.605e+00  4.002e-01   6.509 1.21e-10 ***
    ## mB             -2.207e-01  1.108e-01  -1.991 0.046711 *  
    ## mC             -1.483e-01  1.092e-01  -1.357 0.174983    
    ## mD             -2.046e-01  1.104e-01  -1.852 0.064274 .  
    ## mE             -3.395e-02  1.094e-01  -0.310 0.756304    
    ## x1             -3.158e+01  3.434e+00  -9.196  < 2e-16 ***
    ## I(x1^2)         6.853e+01  7.936e+00   8.636  < 2e-16 ***
    ## x2             -7.596e+00  2.661e+00  -2.854 0.004404 ** 
    ## x3              6.844e+00  8.793e-01   7.784 1.79e-14 ***
    ## x4             -7.916e-01  3.334e+00  -0.237 0.812391    
    ## z              -2.287e+00  6.665e-01  -3.431 0.000627 ***
    ## I(z^2)          5.226e-01  1.135e-01   4.606 4.65e-06 ***
    ## v1             -9.938e-03  1.275e-02  -0.779 0.435959    
    ## v2             -1.573e-01  1.253e-01  -1.255 0.209692    
    ## v3             -3.958e-02  1.360e-02  -2.910 0.003692 ** 
    ## v4              1.400e-01  1.504e-01   0.931 0.352234    
    ## v5             -1.477e-02  1.041e-02  -1.419 0.156333    
    ## w               3.395e+00  7.089e-01   4.789 1.94e-06 ***
    ## I(w^2)         -9.799e-01  4.868e-01  -2.013 0.044414 *  
    ## x1:x2          -3.930e+01  1.399e+01  -2.810 0.005056 ** 
    ## x1:x3          -7.503e+01  6.861e+00 -10.936  < 2e-16 ***
    ## x1:x4           5.268e+01  2.711e+01   1.943 0.052322 .  
    ## x1:z            1.937e+01  4.114e+00   4.709 2.86e-06 ***
    ## x1:I(z^2)      -2.624e+00  7.189e-01  -3.651 0.000276 ***
    ## I(x1^2):x2      1.521e+02  1.892e+01   8.039 2.62e-15 ***
    ## I(x1^2):x3      1.460e+02  1.485e+01   9.829  < 2e-16 ***
    ## I(x1^2):x4     -1.140e+02  4.728e+01  -2.412 0.016066 *  
    ## I(x1^2):z      -4.725e+01  5.299e+00  -8.917  < 2e-16 ***
    ## I(x1^2):I(z^2)  6.926e+00  1.024e+00   6.764 2.31e-11 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.073 on 972 degrees of freedom
    ## Multiple R-squared:  0.7807, Adjusted R-squared:  0.7743 
    ## F-statistic: 123.6 on 28 and 972 DF,  p-value: < 2.2e-16

x1 definitely seems to be important, if not the most important feature
in the entire dataset. The separate interactions of x1 with x2, x3, and
z are also important. Of the v inputs, v3 was the only one which
remained consistently important across the models. w and z were also
found to be important - the best results were obtained when no linear
basis function was applied to w, and when a quadratic (square) was
applied to z (with the EDA in mind, the square on z makes sense).

It is worth noting that the model summaries show very high standard
errors for some of our coefficients. In other words, we are relatively
uncertain about the true (population) value of these coefficients. This
is most certainly due to issues with colinearity in the models, which
makes sense, because I included duplicated inputs (ex: z + I(z^2)) in
all of the basis models.

### iiB) Bayesian Regression Models

First, we must define our functions for using Bayesian regression with
Laplace Approximation.

``` r
lm_logpost <- function(unknowns, my_info)
{
  # specify the number of unknown beta parameters
  length_beta <- ncol(my_info$design_matrix)
  
  # extract the beta parameters from the `unknowns` vector
  beta_v <- unknowns[1:length_beta]
  
  # extract the unbounded noise parameter, varphi
  lik_varphi <- unknowns[length_beta + 1]
  
  # back-transform from varphi to sigma
  lik_sigma <- exp(lik_varphi)
  
  # extract design matrix
  X <- my_info$design_matrix
  
  # calculate the linear predictor
  mu <- as.vector(X %*% as.matrix(beta_v))
  
  # evaluate the log-likelihood
  log_lik <- sum(dnorm(x = my_info$yobs,
                       mean = mu,
                       sd = lik_sigma,
                       log = TRUE))
  
  # evaluate the log-prior
  log_prior_beta <- sum(dnorm(x = beta_v,
                              mean = my_info$mu_beta,
                              sd = my_info$tau_beta,
                              log = TRUE))
  
  log_prior_sigma <- dexp(x = lik_sigma, 
                          rate = my_info$sigma_rate, 
                          log = TRUE)
  
  # add the mean trend prior and noise prior together
  log_prior <- log_prior_beta + log_prior_sigma
  
  # account for the transformation
  log_derive_adjust <- lik_varphi
  
  # sum together
  log_lik + log_prior + log_derive_adjust
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
generate_lm_post_samples <- function(mvn_result, length_beta, num_samples)
{
  MASS::mvrnorm(n = num_samples,
                mu = mvn_result$mode,
                Sigma = mvn_result$var_matrix) %>% 
    as.data.frame() %>% tibble::as_tibble() %>% 
    purrr::set_names(c(sprintf("beta_%02d", 0:(length_beta-1)), "varphi")) %>% 
    mutate(sigma = exp(varphi))
}
```

``` r
post_lm_pred_samples <- function(Xnew, Bmat, sigma_vector)
{
  # number of new prediction locations
  M <- nrow(Xnew)
  # number of posterior samples
  S <- nrow(Bmat)
  
  # matrix of linear predictors
  Umat <- Xnew %*% t(Bmat)
  
  # assmeble matrix of sigma samples, set the number of rows
  Rmat <- matrix(rep(sigma_vector, M), M, byrow = TRUE)
  
  # generate standard normal and assemble into matrix
  # set the number of rows
  Zmat <- matrix(rnorm(M*S), M, byrow = TRUE)
  
  # calculate the random observation predictions
  Ymat <- Umat + Rmat * Zmat
  
  # package together
  list(Umat = Umat, Ymat = Ymat)
  
}
```

``` r
make_post_lm_pred <- function(Xnew, post)
{
  Bmat <- post %>% select(starts_with("beta_")) %>% as.matrix()
  
  sigma_vector <- post %>% pull(sigma)
  
  post_lm_pred_samples(Xnew, Bmat, sigma_vector)
}
```

``` r
summarize_lm_pred_from_laplace <- function(mvn_result, Xtest, num_samples)
{
  # generate posterior samples of the beta parameters
  post <- generate_lm_post_samples(mvn_result, ncol(Xtest), num_samples)
  
  # make posterior predictions on the test set
  pred_test <- make_post_lm_pred(Xtest, post)
  
  # calculate summary statistics on the predicted mean and response
  # summarize over the posterior samples
  
  # posterior mean, should you summarize along rows (rowMeans) or 
  # summarize down columns (colMeans) ???
  mu_avg <- rowMeans(pred_test$Umat)
  y_avg <- rowMeans(pred_test$Ymat)
  
  # posterior quantiles for the middle 95% uncertainty intervals
  mu_lwr <- apply(pred_test$Umat, 1, stats::quantile, probs = 0.025)
  mu_upr <- apply(pred_test$Umat, 1, stats::quantile, probs = 0.975)
  y_lwr <- apply(pred_test$Ymat, 1, stats::quantile, probs = 0.025)
  y_upr <- apply(pred_test$Ymat, 1, stats::quantile, probs = 0.975)
  
  # book keeping
  tibble::tibble(
    mu_avg = mu_avg,
    mu_lwr = mu_lwr,
    mu_upr = mu_upr,
    y_avg = y_avg,
    y_lwr = y_lwr,
    y_upr = y_upr
  ) %>% 
    tibble::rowid_to_column("pred_id")
}
```

For the next part part, I will refit the best model from iA
(reg_basisMod3). I will also refit reg_expandedMod3 - I chose this model
because it seems like the interactions between continuous features
(especially when including those in the expanded feature set) are
important for predicting the outcome. This is evident from both test set
RMSE and the model summaries.

``` r
reg_bayes_X01 <- model.matrix(y ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                      (v1 + v2 + v3 + v4 + v5), data = reg_expanded_train)

reg_bayes_info01 <- list(
  yobs = reg_expanded_train$y,
  design_matrix = reg_bayes_X01,
  mu_beta = 0,
  tau_beta = 5,
  sigma_rate = 1
)
```

``` r
reg_bayes_laplace01 <- my_laplace(rep(0, ncol(reg_bayes_X01) + 1), lm_logpost, reg_bayes_info01)
reg_bayes_laplace01 %>% glimpse()
```

    ## List of 5
    ##  $ mode        : num [1:38] 7.6332 -0.1955 -0.1269 -0.2443 0.0139 ...
    ##  $ var_matrix  : num [1:38, 1:38] 11.53419 -0.00597 -0.00398 -0.00282 -0.00178 ...
    ##  $ log_evidence: num -1735
    ##  $ converge    : chr "YES"
    ##  $ iter_counts : num 132

``` r
reg_bayes_X02 <- model.matrix(y ~ .^2, data = (reg_expanded_train %>% select(-m)))

reg_bayes_info02 <- list(
  yobs = reg_expanded_train$y,
  design_matrix = reg_bayes_X02,
  mu_beta = 0,
  tau_beta = 5,
  sigma_rate = 1
)
```

``` r
reg_bayes_laplace02 <- my_laplace(rep(0, ncol(reg_bayes_X02) + 1), lm_logpost, reg_bayes_info02)
reg_bayes_laplace02 %>% glimpse()
```

    ## List of 5
    ##  $ mode        : num [1:93] -0.453 -5.613 0.144 1.09 0.126 ...
    ##  $ var_matrix  : num [1:93, 1:93] 6.35 -2.67 -1.39 -5.69 -2.72 ...
    ##  $ log_evidence: num -1915
    ##  $ converge    : chr "YES"
    ##  $ iter_counts : num 281

Once again, we will use test set RMSE to evaluate the models. First,
let’s generate our posterior predictions.

``` r
reg_bayes_testX01 <- model.matrix(y ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                      (v1 + v2 + v3 + v4 + v5), data = reg_expanded_test)

reg_bayes_pred01 <- summarize_lm_pred_from_laplace(reg_bayes_laplace01, reg_bayes_testX01, 5000)
reg_bayes_pred01 %>% glimpse()
```

    ## Rows: 251
    ## Columns: 7
    ## $ pred_id <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,…
    ## $ mu_avg  <dbl> 0.3413236, -1.0860630, -0.5372310, 3.5296929, 2.4470987, 0.882…
    ## $ mu_lwr  <dbl> -0.08250572, -1.33940074, -0.82322883, 3.25074952, 2.13224560,…
    ## $ mu_upr  <dbl> 0.76752163, -0.83601457, -0.26000685, 3.81404331, 2.76205462, …
    ## $ y_avg   <dbl> 0.2898910, -1.0485816, -0.5318385, 3.5451589, 2.4345425, 0.866…
    ## $ y_lwr   <dbl> -2.10121410, -3.50838979, -2.95389891, 1.19122334, 0.09711106,…
    ## $ y_upr   <dbl> 2.7795537, 1.3234020, 1.8187628, 5.9278328, 4.8247053, 3.29067…

``` r
reg_bayes_testX02 <- model.matrix(y ~ .^2, data = (reg_expanded_test %>% select(-m)))

reg_bayes_pred02 <- summarize_lm_pred_from_laplace(reg_bayes_laplace02, reg_bayes_testX02, 5000)
reg_bayes_pred02 %>% glimpse()
```

    ## Rows: 251
    ## Columns: 7
    ## $ pred_id <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,…
    ## $ mu_avg  <dbl> -0.3930052, -1.3237997, -0.8568971, 3.3394831, 2.5041324, 0.54…
    ## $ mu_lwr  <dbl> -1.29904215, -1.58195248, -1.17931582, 2.96372051, 1.97857433,…
    ## $ mu_upr  <dbl> 0.48498134, -1.07367405, -0.53853619, 3.72587640, 3.02498959, …
    ## $ y_avg   <dbl> -0.3969790, -1.3105996, -0.8397848, 3.3637124, 2.5186391, 0.53…
    ## $ y_lwr   <dbl> -2.92092205, -3.73963826, -3.39914017, 0.99338521, -0.05721276…
    ## $ y_upr   <dbl> 2.229787, 1.057753, 1.571336, 5.795018, 4.998273, 3.026731, 7.…

Now, calculate RMSE by comparing y_avg to the actual y value.

``` r
reg_bayes_rmse1 <- sqrt(mean((reg_bayes_pred01$y_avg - reg_base_test$y)^2))

reg_bayes_rmse2 <- sqrt(mean((reg_bayes_pred02$y_avg - reg_base_test$y)^2))

reg_bayes_rmse_df <- data.frame(model = c("reg_bayesMod1", "reg_bayesMod2"),
                          rmse = c(reg_bayes_rmse1, reg_bayes_rmse2))

reg_bayes_rmse_df %>% ggplot() + 
  geom_point(aes(x = as.factor(model), y = rmse)) + 
  xlab("Model") + 
  ylab("RMSE") + 
  ggtitle("Test Set RMSE for each of the Two Bayesian Regression Models") + 
  theme_bw() +
  theme(axis.text.x = element_text(face = "bold", angle = 90))
```

![](part2_regression_files/figure-gfm/Bayesian%20Regression%20Model%20RMSE-1.png)<!-- -->

The best of the two models was identified as reg_bayesMod1, as according
to test set RMSE. Next, let’s show the regression coefficient posterior
summary statistics for our best model.

``` r
reg_bayes_summary <- function(laplace_object, design_matrix){
  
  names <- c(colnames(design_matrix), "varphi")
  modes <- laplace_object$mode
  sd <- sqrt(diag(laplace_object$var_matrix))
  
  df <- data.frame(Parameter = names, Estimate = modes, sdev = sd)
  
  df2 <- data.frame(Parameter = "sigma", Estimate = exp(df[nrow(df), 2]), sdev = exp(df[nrow(df), 3]))
  
  df %>% rbind(df2)
    
}
```

``` r
reg_bayes_summary01 <- reg_bayes_summary(reg_bayes_laplace01, reg_bayes_X01)
reg_bayes_summary01
```

    ##         Parameter      Estimate       sdev
    ## 1     (Intercept)   7.633194427 3.39620220
    ## 2              mB  -0.195526341 0.12598189
    ## 3              mC  -0.126864358 0.12427106
    ## 4              mD  -0.244253458 0.12549195
    ## 5              mE   0.013882132 0.12420701
    ## 6              x1  -5.856789402 3.54536377
    ## 7         sin(x1)  -3.267535955 3.62960376
    ## 8         cos(x1)  -5.406942821 3.35642658
    ## 9              x2   0.875714711 3.66603276
    ## 10             x3   5.945814240 3.53944867
    ## 11             x4   0.965160183 3.64373009
    ## 12              z  -4.157481390 3.26675943
    ## 13         I(z^2)   5.297327112 1.28347718
    ## 14              w   4.586162778 3.35683768
    ## 15             v1  -0.005488556 0.01454719
    ## 16             v2  -0.120428325 0.14265584
    ## 17             v3  -0.041883997 0.01550184
    ## 18             v4   0.081938227 0.17143754
    ## 19             v5  -0.023908227 0.01185096
    ## 20          x1:x2  -9.825301512 4.34348603
    ## 21          x1:x3 -11.903143538 3.89734503
    ## 22          x1:x4  -0.585976447 4.45593727
    ## 23           x1:z  -2.045232282 3.47955513
    ## 24      x1:I(z^2) -13.108605949 3.21211311
    ## 25           x1:w   1.437828992 3.55250220
    ## 26     sin(x1):x2 -10.339575798 4.36368885
    ## 27     sin(x1):x3 -13.408835868 3.94575189
    ## 28     sin(x1):x4  -0.281704932 4.50417831
    ## 29      sin(x1):z  10.010700666 3.65489892
    ## 30 sin(x1):I(z^2)  11.925921796 3.09185412
    ## 31      sin(x1):w   1.950393998 3.69427306
    ## 32     cos(x1):x2  -2.524066095 3.71895179
    ## 33     cos(x1):x3  -2.905479655 3.53103143
    ## 34     cos(x1):x4   0.907857184 3.70448091
    ## 35      cos(x1):z   1.723012039 3.15037438
    ## 36 cos(x1):I(z^2)  -4.788571036 1.25473444
    ## 37      cos(x1):w  -3.854112467 3.28840422
    ## 38         varphi   0.202996723 0.02295482
    ## 39          sigma   1.225068453 1.02322031

Let’s extract our posterior sigma term…

``` r
reg_bayes_summary01[nrow(reg_bayes_summary01), ]
```

    ##    Parameter Estimate    sdev
    ## 39     sigma 1.225068 1.02322

… and compare it to sigma from the corresponding lm() model.

``` r
summary(reg_basisMod3)$sigma
```

    ## [1] 0.9623927

The Bayesian and MLE estimates on sigma are relatively similar (1.23
vs. 0.96, respectively). Overall, the posterior uncertainty on sigma is
not very precise, considering the standard deviation on sigma was
estimated to be 1.02 (almost as large as the sigma term itself)
according to our Bayesian analyses.

### iiC) Linear Model Predictions

Even though I have the means to easily visualize the confidence and
prediction intervals for the Bayesian models, I will instead use the
linear models for this part (I’d like to practice working with
prediction intervals for lm objects!).

Start by defining the prediction grid. Considering x1 and z seem to be
the most important inputs, we will define our grid in terms of these
parameters. The rest of the variables will be assigned their mean (if
continuous) or mode (if categorical) (UPDATE: I had to use at least two
values for the categorical variable, so I picked the two most common
ones).

``` r
reg_pred_viz_grid <- expand.grid(x1 = seq(min(data$x1), max(data$x1), length.out = 101),
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

Make the predictions, and retrieve the confidence/prediction intervals.

``` r
reg_linear_pred01_plusConfInt <- data.frame(predict(reg_basisMod3, newdata = reg_pred_viz_grid, 
                                                    interval = "confidence", level = 0.95)) %>%
  rename(ci_lwr = lwr, ci_upr = upr)
reg_linear_pred01_plusPredInt <- data.frame(predict(reg_basisMod3, newdata = reg_pred_viz_grid, 
                                                    interval = "prediction", level = 0.95)) %>%
  rename(pred_lwr = lwr, pred_upr = upr) %>% select(-fit)

reg_linear_pred01 <- cbind(reg_pred_viz_grid, reg_linear_pred01_plusConfInt, reg_linear_pred01_plusPredInt)
```

``` r
reg_linear_pred02_plusConfInt <- data.frame(predict(reg_expandedMod3, newdata = reg_pred_viz_grid, 
                                                    interval = "confidence", level = 0.95)) %>% 
  rename(ci_lwr = lwr, ci_upr = upr)
```

    ## Warning in predict.lm(reg_expandedMod3, newdata = reg_pred_viz_grid, interval =
    ## "confidence", : prediction from a rank-deficient fit may be misleading

``` r
reg_linear_pred02_plusPredInt <- data.frame(predict(reg_expandedMod3, newdata = reg_pred_viz_grid, 
                                                    interval = "prediction", level = 0.95)) %>% 
  rename(pred_lwr = lwr, pred_upr = upr) %>% select(-fit)
```

    ## Warning in predict.lm(reg_expandedMod3, newdata = reg_pred_viz_grid, interval =
    ## "prediction", : prediction from a rank-deficient fit may be misleading

``` r
reg_linear_pred02 <- cbind(reg_pred_viz_grid, reg_linear_pred02_plusConfInt, reg_linear_pred02_plusPredInt)
```

Finally, visualize the results.

``` r
reg_linear_pred01 %>% ggplot(aes(x = x1)) + 
  geom_ribbon(aes(ymin = pred_lwr, ymax = pred_upr), fill = 'orange') + 
  geom_ribbon(aes(ymin = ci_lwr, ymax = ci_upr), fill = 'grey') + 
  geom_line(aes(y = fit)) + 
  facet_wrap(facets = ~z) + 
  coord_cartesian(ylim = c(-7, 7))
```

![](part2_regression_files/figure-gfm/reg_linear_pred01%20viz-1.png)<!-- -->

``` r
reg_linear_pred02 %>% ggplot(aes(x = x1)) + 
  geom_ribbon(aes(ymin = pred_lwr, ymax = pred_upr), fill = 'orange') + 
  geom_ribbon(aes(ymin = ci_lwr, ymax = ci_upr), fill = 'grey') + 
  geom_line(aes(y = fit)) + 
  facet_wrap(facets = ~z) + 
  coord_cartesian(ylim = c(-7, 7))
```

![](part2_regression_files/figure-gfm/reg_linear_pred02%20viz-1.png)<!-- -->

The predictive trends are nowhere near the same across the two models.
This makes sense, considering that we are using relatively complex basis
functions in reg_basisMod3 and simple linear interaction terms in
reg_expandedMod3. Surprisingly, reg_basisMod3 did not show signs of
overfitting when being evaluated using the test set… we will see if this
holds up through the rest of our analysis.

### iiD) Train/Tune with Resampling

First, we will prepare the caret dataset to fit our models.

``` r
reg_df_caret <- data %>% 
  select(-c(output, outcome))

reg_df_caret %>% glimpse()
```

    ## Rows: 1,252
    ## Columns: 15
    ## $ x1 <dbl> 0.025878, 0.030768, 0.019325, 0.306212, 0.031296, 0.031073, 0.02440…
    ## $ x2 <dbl> 0.255934, 0.261575, 0.020877, 0.033379, 0.259342, 0.027119, 0.03183…
    ## $ x3 <dbl> 0.492830, 0.498460, 0.258360, 0.255385, 0.264387, 0.260915, 0.02205…
    ## $ x4 <dbl> 0.012770, 0.055779, 0.012424, 0.056190, 0.056594, 0.055192, 0.05575…
    ## $ v1 <dbl> 0.275651, 0.343204, 4.998508, 5.090153, 5.031107, 9.977407, 0.23012…
    ## $ v2 <dbl> 0.033657, 0.027082, 0.030259, 0.052342, 0.517705, 0.532436, 1.00521…
    ## $ v3 <dbl> 1.166214, 1.260579, 1.298285, 1.322005, 1.368195, 1.298797, 1.16544…
    ## $ v4 <dbl> 0.408402, 0.664248, 0.412870, 0.652111, 0.533701, 0.857509, 0.69071…
    ## $ v5 <dbl> 0.525226, 2.866343, 0.409007, 0.861594, 6.451933, 0.958574, 0.20876…
    ## $ m  <chr> "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A…
    ## $ x5 <dbl> 0.212588, 0.153418, 0.689014, 0.348834, 0.388381, 0.625701, 0.86595…
    ## $ w  <dbl> 0.50619858, 0.47195344, 0.07709835, 0.10712990, 0.80796683, 0.08579…
    ## $ z  <dbl> 1.25050808, 1.39745312, 0.05731369, 0.83844661, 0.65315580, 0.08546…
    ## $ t  <dbl> 0.009277586, 0.009294651, 0.151249854, 0.266428788, 2.604629249, 5.…
    ## $ y  <dbl> 1.3009808, 0.9946226, 5.5174529, -0.7263327, 1.0201407, 3.0320223, …

Next, let’s define our metric and resampling method. We will perform
10-fold 5-repeat cross validation.

``` r
my_ctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 5)

my_metric <- 'RMSE'
```

Now, let’s start fitting models. We will start by fitting the four
linear models. The seed will be set in each chunk to ensure
reproducibility.

``` r
set.seed(1234)

reg_caret_lm_base <- train(y ~ ., 
                           data = (reg_df_caret %>% select(-c(x5, w, z, t))),
                           method = "lm",
                           metric = my_metric,
                           preProcess = c("center", "scale"),
                           trControl = my_ctrl)
```

``` r
set.seed(1234)

reg_caret_lm_expanded <- train(y ~ ., 
                               data = reg_df_caret,
                               method = "lm",
                               metric = my_metric,
                               preProcess = c("center", "scale"),
                               trControl = my_ctrl)
```

``` r
set.seed(1234)

reg_caret_lm_basisMod3 <- caret::train(y ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                                         (v1 + v2 + v3 + v4 + v5), 
                                data = reg_df_caret,
                                method = "lm",
                                metric = my_metric,
                                preProcess = c("center", "scale"),
                                trControl = my_ctrl)
```

``` r
set.seed(1234)

reg_caret_lm_expandedMod3 <- train(y ~ .^2, 
                                   data = reg_df_caret,
                                   method = "lm",
                                   metric = my_metric,
                                   preProcess = c("center", "scale"),
                                   trControl = my_ctrl)
```

Next, let’s train the two elastic net models.

``` r
set.seed(1234)

reg_caret_enetMod1 <- train(y ~ m * (x1 + x2 + x3 + x4 + x5 + v1 + v2 + v3 + v4 + v5 + w + z + t)^2, 
                               data = reg_df_caret,
                               method = "glmnet",
                               metric = my_metric,
                               preProcess = c("center", "scale"),
                               trControl = my_ctrl)
```

``` r
set.seed(1234)

reg_caret_enetMod2 <- caret::train(y ~ m + (x1 + sin(x1) + cos(x1)) * (x2 + x3 + x4 + z + I(z^2) + w) + 
                                          (v1 + v2 + v3 + v4 + v5), 
                                      data = reg_df_caret,
                                      method = "glmnet",
                                      metric = my_metric,
                                      preProcess = c("center", "scale"),
                                      trControl = my_ctrl)
```

Now, let’s train the base and expanded models using neural networks.

``` r
set.seed(1234)

reg_caret_neuralNet_base <- train(y ~ ., 
                                  data = (reg_df_caret %>% select(-c(x5, w, z, t))),
                                  method = "nnet",
                                  metric = my_metric,
                                  preProcess = c("center", "scale"),
                                  trControl = my_ctrl, 
                                  trace = FALSE)
```

``` r
set.seed(1234)

reg_caret_neuralNet_expanded <- train(y ~ ., 
                                      data = reg_df_caret,
                                      method = "nnet",
                                      metric = my_metric,
                                      preProcess = c("center", "scale"),
                                      trControl = my_ctrl, 
                                      trace = FALSE)
```

Next, let’s train the base and expanded models using the random forest
method.

``` r
set.seed(1234)

reg_caret_rf_base <- train(y ~ ., 
                           data = (reg_df_caret %>% select(-c(x5, w, z, t))),
                           method = "rf",
                           metric = my_metric,
                           preProcess = c("center", "scale"),
                           trControl = my_ctrl, 
                           trace = FALSE)
```

``` r
set.seed(1234)

reg_caret_rf_expanded <- train(y ~ ., 
                               data = reg_df_caret,
                               method = "rf",
                               metric = my_metric,
                               preProcess = c("center", "scale"),
                               trControl = my_ctrl, 
                               trace = FALSE)
```

Next, let’s train the base and expanded models using the gradient
boosted tree method.

``` r
set.seed(1234)

reg_caret_xgb_base <- train(y ~ ., 
                            data = (reg_df_caret %>% select(-c(x5, w, z, t))),
                            method = "xgbTree",
                            metric = my_metric,
                            preProcess = c("center", "scale"),
                            trControl = my_ctrl, 
                            trace = FALSE, 
                            verbosity = 0)
```

``` r
set.seed(1234)

reg_caret_xgb_expanded <- train(y ~ ., 
                                data = reg_df_caret,
                                method = "xgbTree",
                                metric = my_metric,
                                preProcess = c("center", "scale"),
                                trControl = my_ctrl, 
                                trace = FALSE, 
                                verbosity = 0)
```

Lastly, we will use two methods that were not discussed in class. First,
let’s fit the base and expanded additive models with Generalized
Additive Modeling using Splines.

``` r
set.seed(1234)

reg_caret_gamSpline_base <- train(y ~ ., 
                                  data = (reg_df_caret %>% select(-c(x5, w, z, t))),
                                  method = "gamSpline",
                                  metric = my_metric,
                                  preProcess = c("center", "scale"),
                                  trControl = my_ctrl, 
                                  verbosity = 0)
```

    ## Loading required package: gam

    ## Loading required package: splines

    ## Loading required package: foreach

    ## 
    ## Attaching package: 'foreach'

    ## The following objects are masked from 'package:purrr':
    ## 
    ##     accumulate, when

    ## Loaded gam 1.22

``` r
set.seed(1234)

reg_caret_gamSpline_expanded <- train(y ~ ., 
                                      data = reg_df_caret,
                                      method = "gamSpline",
                                      metric = my_metric,
                                      preProcess = c("center", "scale"),
                                      trControl = my_ctrl, 
                                      verbosity = 0)
```

Now, let’s fit the base and expanded additive models using Multi-Layer
Perceptron.

``` r
set.seed(1234)

reg_caret_mlp_base <- train(y ~ ., 
                            data = (reg_df_caret %>% select(-c(x5, w, z, t))),
                            method = "mlp",
                            metric = my_metric,
                            preProcess = c("center", "scale"),
                            trControl = my_ctrl, 
                            verbosity = 0)
```

``` r
set.seed(1234)

reg_caret_mlp_expanded <- train(y ~ ., 
                                data = reg_df_caret,
                                method = "mlp",
                                metric = my_metric,
                                preProcess = c("center", "scale"),
                                trControl = my_ctrl,
                                verbosity = 0)
```

Finally, we will visualize a dotplot comparing resampled RMSE for all of
the models trained using caret.

``` r
caret_rmse_compare <- resamples(list(LM_base = reg_caret_lm_base,
                                     LM_expanded = reg_caret_lm_expanded,
                                     LM_basisMod3 = reg_caret_lm_basisMod3,
                                     LM_expandedMod3 = reg_caret_lm_expandedMod3,
                                     ENET_interactions = reg_caret_enetMod1,
                                     ENET_basisMod3 = reg_caret_enetMod2,
                                     NNET_base = reg_caret_neuralNet_base,
                                     NNET_expanded = reg_caret_neuralNet_expanded, 
                                     RF_base = reg_caret_rf_base, 
                                     RF_expanded = reg_caret_rf_expanded, 
                                     XGB_base = reg_caret_xgb_base, 
                                     XGB_expanded = reg_caret_xgb_expanded, 
                                     gamSpline_base = reg_caret_gamSpline_base, 
                                     gamSpline_expanded = reg_caret_gamSpline_expanded, 
                                     MLP_base = reg_caret_mlp_base, 
                                     MLP_expanded = reg_caret_mlp_expanded))

dotplot(caret_rmse_compare, metric = 'RMSE')
```

![](part2_regression_files/figure-gfm/caret%20models%20RMSE-1.png)<!-- -->

As shown by the above plot of resampled RMSE, the xgBoost method with
the expanded set of features had the best results by a considerable
margin.
