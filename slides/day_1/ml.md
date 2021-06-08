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
    * *Supervised learning, (non)-linear regression*
    * Understanding basic modelling
    * Confirming your model is sane
    * Tuning your model
    * **All within a very applied setting**
* Tools
    * Numpy
    * Scikit-learn

## What is supervised learning?

* Imagine someone gives you a group of smokers
    * And asks the question -- what is there life expectancy?
* **Completely made up imaginary data**


## Some abstraction

* We are given inputs $x_0, x_1...x_n$ and we are looking to predict $y$
* Let's plot!
