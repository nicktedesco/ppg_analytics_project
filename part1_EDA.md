Final Project Part 1: Exploratory Data Analysis
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

## Part 1: Exploration

### Summary Statistics

Let’s start by calculating some summary statistics for each of our
variables.

``` r
summary(data)
```

    ##        x1                 x2                 x3                 x4          
    ##  Min.   :0.003117   Min.   :0.001173   Min.   :0.003344   Min.   :0.001447  
    ##  1st Qu.:0.144825   1st Qu.:0.059488   1st Qu.:0.180333   1st Qu.:0.034732  
    ##  Median :0.278602   Median :0.170552   Median :0.263551   Median :0.055822  
    ##  Mean   :0.265192   Mean   :0.159282   Mean   :0.262674   Mean   :0.053189  
    ##  3rd Qu.:0.352812   3rd Qu.:0.238338   3rd Qu.:0.343556   3rd Qu.:0.072108  
    ##  Max.   :0.609092   Max.   :0.446306   Max.   :0.509710   Max.   :0.101868  
    ##        v1                  v2                 v3               v4         
    ##  Min.   : 0.003474   Min.   :0.002281   Min.   : 1.003   Min.   :0.01867  
    ##  1st Qu.: 3.335833   1st Qu.:0.334104   1st Qu.: 3.945   1st Qu.:0.30184  
    ##  Median : 5.137150   Median :0.515154   Median : 5.632   Median :0.48923  
    ##  Mean   : 5.079560   Mean   :0.503186   Mean   : 5.569   Mean   :0.49132  
    ##  3rd Qu.: 6.850576   3rd Qu.:0.684780   3rd Qu.: 7.189   3rd Qu.:0.67959  
    ##  Max.   :10.133807   Max.   :1.018897   Max.   :10.177   Max.   :0.97913  
    ##        v5                 m                 output             x5          
    ##  Min.   : 0.006831   Length:1252        Min.   :0.0070   Min.   :0.000718  
    ##  1st Qu.: 2.439787   Class :character   1st Qu.:0.2517   1st Qu.:0.115705  
    ##  Median : 6.496589   Mode  :character   Median :0.4835   Median :0.185718  
    ##  Mean   : 5.867330                      Mean   :0.5311   Mean   :0.259663  
    ##  3rd Qu.: 9.328919                      3rd Qu.:0.8430   3rd Qu.:0.362748  
    ##  Max.   : 9.999845                      Max.   :0.9990   Max.   :0.931574  
    ##        w                  z                 t                   y           
    ##  Min.   :0.002642   Min.   :0.03436   Min.   : 0.000443   Min.   :-4.95482  
    ##  1st Qu.:0.277468   1st Qu.:0.82703   1st Qu.: 0.797858   1st Qu.:-1.08930  
    ##  Median :0.537113   Median :1.95938   Median : 2.364547   Median :-0.06602  
    ##  Mean   :0.523756   Mean   :2.06518   Mean   : 2.538723   Mean   : 0.53007  
    ##  3rd Qu.:0.770949   3rd Qu.:3.07785   3rd Qu.: 3.419533   3rd Qu.: 1.68072  
    ##  Max.   :0.998165   Max.   :5.08727   Max.   :10.136498   Max.   : 6.90676  
    ##       outcome   
    ##  event    :436  
    ##  non_event:816  
    ##                 
    ##                 
    ##                 
    ## 

### Univariate Distributions

Now, let’s visualize each of the univariate distributions in the
dataset.

``` r
ggplot(data = data,
       mapping = aes(x = x1)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x1-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x2)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x2-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x3)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x3-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x4)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x4-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x5)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x5-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v1)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v1-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v2)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v2-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v3)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v3-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v4)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v4-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v5)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v5-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = m)) + 
  geom_bar()
```

![](part1_EDA_files/figure-gfm/Bar%20Chart:%20M-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = w)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20w-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = z)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20z-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = t)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20t-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = output)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20output-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = y)) + 
  geom_histogram(bins = 20)
```

![](part1_EDA_files/figure-gfm/Histogram:%20y-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = outcome)) + 
  geom_bar()
```

![](part1_EDA_files/figure-gfm/Bar%20Chart:%20Outcome-1.png)<!-- -->

### Univariate Distributions Facetted by Categorical Input (M)

Now, let’s visualize each of the univariate distributions again, by
facetted by the categorical input (m).

``` r
ggplot(data = data,
       mapping = aes(x = x1)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x1%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x2)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x2%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x3)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x3%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x4)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x4%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = x5)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20x5%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v1)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v1%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v2)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v2%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v3)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v3%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v4)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v4%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = v5)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20v5%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = w)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20w%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = z)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20z%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = t)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20t%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = output)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20output%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = y)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Histogram:%20y%20(Facetted%20by%20M)-1.png)<!-- -->

``` r
ggplot(data = data,
       mapping = aes(x = outcome)) + 
  geom_bar() + 
  facet_wrap(~m)
```

![](part1_EDA_files/figure-gfm/Bar%20Chart:%20Outcome%20(Facetted%20by%20M)-1.png)<!-- -->

For each of the input variables, their respective univariate
distributions show no extreme change with different values of m. The
output and logit-transformed output (y) both show similar distributions
across the different values of m. The event:nonevent proportion for the
outcome variable does not drastically change with m, but it does appear
that m == D results in a more even distribution of event vs non-event,
as compared to the rest of the values of m. Also, perhaps m == E results
in a higher proportion of the non-event.

These findings suggest that the machine (m) may not greatly impact the
output/outcome.

### Relationships Between Inputs

We can look at the relationships between inputs by making a correlation
matrix for all inputs in the dataset.

``` r
cor_matrix <- cor((data %>% select(-c(m, output, y, outcome))))

corrplot::corrplot(cor_matrix, type = 'upper', method = 'square')
```

![](part1_EDA_files/figure-gfm/corrplot%20of%20inputs-1.png)<!-- -->

Some of the base / derived features are very correlated, as illustrated
by the above correlation plot. This may be somewhat problematic when
fitting certain types of regression models. We will look at some of
these relationships more in-depth when visualizating the binary outcome
with respect to 2D combinations of the inputs (see “Relationships
Between Binary Outcome and Inputs / Derived Features” Chunk).

### Relationships Between Outputs and Inputs

Next, let’s look at the relationships between the different outputs and
inputs by making scatter plots for each combination of input and
output/y.

``` r
data %>% pivot_longer(cols = c(x1, x2, x3, x4, x5, v1, v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = value, y = output)) + 
  geom_point(alpha = 0.3) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/Output%20vs%20Inputs%20and%20Derived%20Features-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(x1, x2, x3, x4, x5, v1, v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = value, y = y)) + 
  geom_point(alpha = 0.3) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/y%20vs%20Inputs%20and%20Derived%20Features-1.png)<!-- -->

We can see clear patterns between some of the inputs and the
output/transformed output. More specifically, look at z vs y
(quadratic?) and x1 vs y (wave-like? quadratic? polynomial?), and w
(wave-like? polynomial?) - these appear to be the strongest
relationships.

### Relationships Between Binary Outcome and Inputs

Finally, we can visualize the relationships between the binary outcome
and the inputs by making 2D scatterplots for each combination of the
inputs, and coloring the points by the value of the outcome
(event/non-event).

``` r
data %>% pivot_longer(cols = c(x2, x3, x4, x5, v1, v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = x1, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/x1%20vs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

There seem to be a few important interactions between x1 and some of the
other inputs… (ex: x1 vs w, z, x2, x3).

``` r
data %>% pivot_longer(cols = c(x3, x4, x5, v1, v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = x2, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/x2%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

The interaction between x2 and w seems to show some patterns for the
binary outcome variable…

``` r
data %>% pivot_longer(cols = c(x4, x5, v1, v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = x3, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/x3%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(x5, v1, v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = x4, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/x4%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(v1, v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = x5, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/x5%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(v2, v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = v1, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/v1%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(v3, v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = v2, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/v2%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(v4, v5, w, z, t)) %>%
  ggplot(mapping = aes(x = v3, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/v3%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(v5, w, z, t)) %>%
  ggplot(mapping = aes(x = v4, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/v4%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(w, z, t)) %>%
  ggplot(mapping = aes(x = v5, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/v5%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(z, t)) %>%
  ggplot(mapping = aes(x = w, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/w%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->

``` r
data %>% pivot_longer(cols = c(t)) %>%
  ggplot(mapping = aes(x = z, y = value)) + 
  geom_point(aes(color = as.factor(outcome), shape = as.factor(outcome)), size = 1) + 
  facet_wrap(~name, scales = 'free')
```

![](part1_EDA_files/figure-gfm/z%20vs%20Inputs%20Colored%20by%20Binary%20Outcome-1.png)<!-- -->
