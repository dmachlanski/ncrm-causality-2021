% A quick introduction to machine learning   
  Spyros Samothrakis \
  Senior Lecturer, IADS \
  University of Essex \
  MiSoC
% June 22, 2022


#

\usebackgroundtemplate{

}


<!-- ------------------------------------------------------ Intro ------------------------------------- -->

# Introduction


## Welcome/course contents
* What will this course cover?
    * Day 1: An intro to machine learning (ML)
    * Day 1: ML labs
    * Day 2: An intro to causal inference
    * Day 2: ML and causal inference labs
* Textbooks?
    * [Mitchell, T. M. (1997). Machine learning.](http://www.cs.cmu.edu/~tom/mlbook.html)
    * [Bishop, C. M. (2006). Pattern recognition and machine learning. springer.](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

    * [Wasserman, L. (2013). All of statistics: a concise course in statistical inference. Springer Science & Business Media.](http://www.stat.cmu.edu/~larry/all-of-statistics/index.html)


## Better science through data

[Hey, Tony, Stewart Tansley, and Kristin M. Tolle. "Jim Gray on eScience: a transformed scientific method." (2009).](http://languagelog.ldc.upenn.edu/myl/JimGrayOnE-Science.pdf)

* Thousand years ago: empirical branch
	* You observed stuff and you wrote down about it
* Last few hundred years: theoretical branch
	*  Equations of gravity, equations of electromagnetism
* Last few decades: computational branch
	* Modelling at the micro level, observing at the macro level
* Today: data exploration
	* Let machines create models using vast amounts of data

## Better business through data

* There was a report by Mckinsey

[Manyika, J., Chui, M., Brown, B., Bughin, J., Dobbs, R., Roxburgh, C., & Hung Byers, A. (2011). Big data: The next frontier for innovation, competition, and productivity. McKinsey Global Institute.](http://www.mckinsey.com/business-functions/digital-mckinsey/our-insights/big-data-the-next-frontier-for-innovation)

* Urges everyone to monetise "Big Data"
* Use the data provided within your organisation to gain insights
* Has some numbers as to how much this is worth
* Proposes a number of methods, most of them associated with machine learning and databases

## Why is it popular now?

* **Algorithms + data + tools**

* [Breiman, L. (2001). Statistical modeling: The two cultures (with comments and a rejoinder by the author). Statistical science, 16(3), 199-231.](http://projecteuclid.org/download/pdf_1/euclid.ss/1009213726%20)
* [Anderson, P. W. (1972). More is different. Science, 177(4047), 393-396.](https://www.tkm.kit.edu/downloads/TKM1_2011_more_is_different_PWA.pdf)
* [Pedregosa, et.al. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830.](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)


## So this course covers tools

* ML theory  
    * *Supervised learning*
        *Regression*
        *Classification*
    * Understanding basic modelling
    * Confirming your model is sane
    * Tuning your model
    * **All within a very applied setting**

* Tools
    * Numpy
    * Scikit-learn

## What is supervised learning?

* Imagine someone gives you a group of smokers
    * And asks the question -- what is their life expectancy?
* **Completely made up imaginary data**


## Some abstraction

* We are given inputs $x_0, x_1...x_n$ and we are looking to predict $y$
* Let's plot!


## Regression - link the dots (1)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_50-crop}.jpg}

## Regression - link the dots (2)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_100-crop}.jpg}

## Regression - link the dots (3)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_200-crop}.jpg}

## Regression - link the dots (4)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_500-crop}.jpg}


## Regression - link the dots (5)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_1000-crop}.jpg}

## Regression - link the dots (6)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_10000-crop}.jpg}


## Classification - draw a boundary (1)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_50_class-crop}.jpg}

## Classification - draw a boundary (2)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_100_class-crop}.jpg}

## Classification - draw a boundary (3)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_200_class-crop}.jpg}

## Classification - draw a boundary (4)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_500_class-crop}.jpg}


## Classification - draw a boundary (5)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_1000_class-crop}.jpg}

## Classification - draw a boundary (6)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{intra_spiral_10000_class-crop}.jpg}

## Full data

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{extra_spiral-crop}.pdf}

## Intuition

* That's it - we are given data, and we need to come up with an algorithm to join it up -- but in high dimensions
    * Can can be binary, categorical, real-valued
* How well well a function joins the data is called the "loss"
* Very low loss is not good, it might not generalise that well to unseen data points -- you can learn to memorise data instances

# Classic algorithms for joining those dots

## Linear regression

* Linear and logistic regression
    * Logistic regression does classification
* You just assume everything is a line


## Example (Linear regression)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_200_LinearRegression-crop}.jpg}


## Example (Linear regression)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_2000_LinearRegression-crop}.jpg}


## Example (Decision tree)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_200_DecisionTreeRegressor-crop}.jpg}


## Example (Decision tree)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_2000_DecisionTreeRegressor-crop}.jpg}

## Example (Decision tree --- internal)

\includegraphics[trim={0 0 0 0},clip,width = 0.3\textwidth]{./code/cropped/{reg_vis_tree_intra_spiral_2000_DecisionTreeRegressor-crop}.jpg}


## Example (Random forest)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_2000_RandomForestRegressor-crop}.jpg}

## Example (Random forest)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_2000_RandomForestRegressor-crop}.jpg}

## Example (Random forest)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_2000_CatBoostRegressor-crop}.jpg}

## Example (Gradient boosting)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{reg_intra_spiral_2000_CatBoostRegressor-crop}.jpg}

## Classification (Logistic regression)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{class_spiral_LogisticRegression}.png}

## Classification (Decision trees)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{class_spiral_DecisionTreeClassifier}.png}


## Classification (Random forests)

\includegraphics[trim={0 0 0 0},clip,width = 0.9\textwidth]{./code/cropped/{class_spiral_RandomForestClassifier}.png}


# Higher dimensions



## Data dimensionality

* Until now we have seen input data of 1 (for regression) or two (for classification) dimensions
* How about higher dimensional data?   
    * Some times data can have millions of features
* Let's examine more high dimensional dataset
* Visualisation becomes harder

## Diabetes data

[Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. Annals of statistics, 32(2), 407-499.](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

\tiny

| Feature  | Description  |
|---|---|
|$X_0$   | age in years  |
|$X_1$   |sex   |
|$X_2$   |bmi body mass index   |
|$X_3$   | bp average blood pressure  |
|$X_4$   | s1 tc, total serum cholesterol  |
|$X_5$   | s2 ldl, low-density lipoproteins  |
|$X_6$   | s3 hdl, high-density lipoproteins  |
|$X_7$   | s4 tch, total cholesterol / HDL  |
|$X_8$   |s5 ltg, possibly log of serum triglycerides level   |
|$X_9$   | s6 glu, blood sugar level  |
|$y$   |  disease progression one year after baseline |

## Let's see the real data values

## Plotting?




# Testin'

## Quality assessment

* In lower dimensions, the visualisations we did provided some insights
to the quality of our methods
    * This is impossible in higher dimensions
* We need to measure some kind of metric that denotes quality of fit


## Metrics

* For regression,
    * Mean Squared Error
    * Mean Absolute Error
* For classification
    * Accuracy
    * Mean Squared Error
    * Cross-entropy loss
    * AUC
* Each one has different benefits, e.g. absolute errors tend to be more robust to outliers


## Accuracy

* Each row is now assigned to a class of ${y_i} \in{0..20}$

* Accuracy is the obvious one
	* $\mathit{accuracy} = \frac{1} {N} \sum\limits_{i=0}^{N-1} {y_i = \hat{f}(x) }$
	* The higher the accuracy the better
* What if the dataset is unbalanced - how informative is accuracy then?
* There are multiple metric functions
	* Use the one appropriate for your problem

## Mean Squared Error (MSE)

* Reality is $f(x)$
* Our model is $\hat{f}(x)$ (e.g. a decision tree)
* Sample from the model are $\{y_{0}... y_{N}\}$

	* $MSE = \frac{1} {N} \sum\limits_{i = 1}^N \left( y_{i} - \hat{f}(x_{i}) \right)^2$
* For every possible sample
	* $E\left[\left(y-\hat{f}(x)\right)^2\right]$





## Train/validation/test split

* Basic idea: split your data into three portions
* (a) train, you used that to train your classifier/regressor
* (b) validation, you use that to assess the quality of your method, retraining as you see fit
* (c) test, you report results on this
* Common split is 60%/20%/20%

## Cross validation

* How about we split our data into multiple validation sets and find the mean?
* Colliqualyy goes by names like 5-fold CV, 10-fold CV
*


## Pictorial depiction of 5-fold CV
[Copied from SKlearns website](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

\includegraphics[trim={0 0 0 0},clip,width = 0.8\textwidth]{./graphics/{grid_search_cross_validation}.png}



# Tuning


## Hyperparameters

* How many trees?
* Tree depth?
* Maximum tree size
* l2 regularisation?

## Effects of hyperparameters

## We need to look for optimal parameters
* Computationally expensive
* We can do this either by searching both the classifier/regressor space and their parameters

# Wrapping up
## Wrapping up
* You get data from somewhere
* ML will help you predict certain targets
* Data can be noisy
* You might need to pre-process it
* The more data the better
* Choosing the right classifier/regressor is important
    *
