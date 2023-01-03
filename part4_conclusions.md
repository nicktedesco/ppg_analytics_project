Final Project Part 4: Interpretation
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

## Retraining Best Models

We have to retrain our best models from regression and classification in
order to have access to them in this file.

### Regression

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

Finally, let’s retrain our best regression model:
reg_caret_xgb_expanded.

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

### Classification

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
my_class_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                        returnData = FALSE, classProbs = TRUE,
                        summaryFunction = twoClassSummary)

my_class_metric <- 'ROC'
```

Finally, let’s retrain our classification best model:
class_caret_xgb_expanded.

``` r
set.seed(1234)

class_caret_xgb_expanded_roc <- train(outcome ~ ., 
                                  data = class_df_caret,
                                  method = "xgbTree",
                                  metric = my_class_metric,
                                  preProcess = c("center", "scale"),
                                  trControl = my_class_ctrl, 
                                  trace = FALSE, 
                                  verbosity = 0)
```

## Part 4: Interpretation

As shown by the RMSE, accuracy, and AUC figures, the expanded versions
of the models tended to perform better than their base counterparts.
This would suggest that features in the expanded dataset are important
for predicting our outcome.

Let’s take a look at the most important variables in our best performing
models for regression and classification. For regression, xgb_expanded
performed the best.

``` r
plot(varImp(reg_caret_xgb_expanded))
```

![](part4_conclusions_files/figure-gfm/varImp%20regression%20XGB_expanded-1.png)<!-- -->

The best regression model suggests that x1 and z are the most important
inputs. Now, let’s look at the best classification model: xgb_expanded.

``` r
plot(varImp(class_caret_xgb_expanded_roc))
```

![](part4_conclusions_files/figure-gfm/varImp%20classification%20xgb_expanded-1.png)<!-- -->

The best classification model suggests that x1 and w are the most
important inputs.

Now, let’s visualize the two most important variables for each case
(regression and classification) with respect to the predicted
logit-transformed response (regression) or predicted probability
(classification). In other words, we will create two contour plots: one
for the important inputs from regression (x1 and z) with respect to the
predicted logit-transformed response, and one for the important inputs
from classification (x1 and w) with respect to the predicted
probability.

First, we need to define two visualization grids to make predictions on:
one for the best regression model and its corresponding inputs (x1 and
z), and one for the classification model and its corresponding inputs
(x1 and w). Later in this section, we are asked to see if the optimal
variable combinations differ over different levels of m - therefore, we
will also include all levels of m in the visualization grids.

``` r
reg_viz_grid <- expand.grid(x1 = seq(min(data$x1), max(data$x1), length.out=75),
                            z = seq(min(data$z), max(data$z), length.out=75),
                            m = c("A", "B", "C", "D", "E"),
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
                            KEEP.OUT.ATTRS = FALSE, 
                            stringsAsFactors = FALSE) %>% 
  as.data.frame() %>% tibble::as_tibble()
```

``` r
class_viz_grid <- expand.grid(x1 = seq(min(data$x1), max(data$x1), length.out=75),
                              w = seq(min(data$w), max(data$w), length.out=75),
                              m = c("A", "B", "C", "D", "E"),
                              x2 = mean(data$x2),
                              x3 = mean(data$x3),
                              x4 = mean(data$x4),
                              x5 = mean(data$x5),
                              v1 = mean(data$v1),
                              v2 = mean(data$v2),
                              v3 = mean(data$v3),
                              v4 = mean(data$v4),
                              v5 = mean(data$v5), 
                              z = mean(data$z),
                              t = mean(data$t), 
                              KEEP.OUT.ATTRS = FALSE, 
                              stringsAsFactors = FALSE) %>% 
  as.data.frame() %>% tibble::as_tibble()
```

Now, make our predictions using the two visualization grids.

``` r
reg_pred <- predict(reg_caret_xgb_expanded, newdata = reg_viz_grid)

reg_viz_grid_df <- cbind(reg_viz_grid, reg_pred)
```

``` r
class_pred <- predict(class_caret_xgb_expanded_roc, newdata = class_viz_grid, type = 'prob')

class_viz_grid_df <- cbind(class_viz_grid, class_pred)
```

``` r
reg_viz_grid_df %>%
  ggplot(aes(x = x1, y = z)) + 
  geom_raster(aes(fill = reg_pred)) + 
  facet_wrap(~m) + 
  scale_fill_gradient2(low = 'blue', mid = 'white', high = 'red',
                       midpoint = max(reg_pred) - (max(reg_pred) - min(reg_pred)) / 2,
                       limits = c(min(reg_pred), max(reg_pred)))
```

![](part4_conclusions_files/figure-gfm/important%20regression%20inputs-1.png)<!-- -->

``` r
class_viz_grid_df %>%
  ggplot(aes(x = x1, y = w)) + 
  geom_raster(aes(fill = event)) + 
  facet_wrap(~m) + 
  scale_fill_gradient2(low = 'blue', mid = 'white', high = 'red',
                       midpoint = 0.5,
                       limits = c(0, 1))
```

![](part4_conclusions_files/figure-gfm/important%20classification%20inputs-1.png)<!-- -->

As shown by the regression figure, low to median values of x1 (0.1 to
0.3) in conjunction with low to median values of z (0 to 3) minimize the
logit-transformed response. This trend does not seem to differ very much
across the different machines.

As shown by the classification figure, very low values of z1 and any
value of w results in the lowest event probability. However, we can also
see that relatively higher values of x1 (0.35 to 0.6) in conjunction
with relatively higher values of w (0.4 to 1.0) also result in low event
probability. This trend does not differ for most of the machines -
however, we see a very slightly different trend for m = E, where
moderately lower values of x1 (0.1 to 0.2) in conjunction with median
(0.50) or high (0.90) values of w result in low event probability.

With all of this information in mind, we might shoot for x1 \~ 0.35, z
\~ 2, and w \~ 0.50 as the most optimal input values for reducing
surface corrosion. If we are using machine E, we might shoot for x1 \~
0.1 to 0.2, z \~ 2, and w \~ 0.50. However, since these inputs depend on
one another, it might be difficult able to achieve this sort of balance
in practice.

## Holdout Set Predictions

First, load the holdout set and initialize the dataframe to store our
results

``` r
holdout <- read.csv('/Users/nick/Downloads/fall2022_holdout_inputs.csv')

holdout <- holdout %>% 
  mutate(
    x5 = 1 - (x1 + x2 + x3 + x4), 
    w = x2 / (x3 + x4), 
    z = (x1 + x2) / (x4 + x5), 
    t = v1 * v2
  )
```

Next, make predictions for the continuous outcome.

``` r
y = predict(reg_caret_xgb_expanded, newdata = holdout)
```

Now, for the categorical outcome.

``` r
probability = predict(class_caret_xgb_expanded_roc, newdata = holdout, type = "prob")
outcome <- ifelse(probability$event >= 0.5, "event", "nonevent")
```

Finally, organize the results into the final dataframe for download.

``` r
holdout_predictions <- data.frame(y = y, outcome = outcome, probability = probability$event) %>%
  tibble::rowid_to_column() %>% rename(id = rowid)
```

``` r
write.csv(holdout_predictions, '/Users/nick/Documents/GSPH/INFSCI 2595/holdout_predictions.csv')
```
