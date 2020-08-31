@def title = "MLJ Time Tutorials"
@def Benchmark = ["syntax", "code"]

# How to use MLJTime

\tableofcontents <!-- you can use \toc as well -->


## Installing

This package requires Julia v1.0 or later.
Refer to the [official documentation](https://julialang.org/downloads) on how to
install and run Julia on your system.

Depending on your needs, choose an appropriate command from the following list
and enter it in Julia's REPL.
To activate the `pkg` mode, type `]` (and to leave it, type `<backspace>`).

### Clone the package for development
```julia
pkg> dev https://github.com/alan-turing-institute/MLJTime.jl.git
```
# MLJ GSoC 2020 Weekly Reports
Contributors: @aa25desh, @vollmersj, @mloning 

## Contents
[TOC]

## Week 1
17 May - 25 June
*intro, onboarding*

## Week 2
25 May - 1 June

### Actions
- [x] Julia implementation of tsf is complete. 
- [x] Ipython notebook of tsf over gunpoint dataset.
- [x] Functionality of ready to load data in MLJtime.

### Questions
- Use of MLJ vs small pkgs for basic algorithms.
  - Package particularities, e.g. predicting single instance vs multiple instance at same time in decesion trees from Decision Tree package in Julia
  - check out MLJ model registry: https://alan-turing-institute.github.io/MLJ.jl/dev/model_search/ and extend for MLJTime algorithms
  - add scitypes for time series https://github.com/alan-turing-institute/MLJScientificTypes.jl/
  - How do we tun a mix of julia and Python objects. 
- Tables can't handel ND objects, they only support 2d tables
  - We have to create Array(matrix) every time as input to algorithms. 
  - options:
      - 3d array
      - storing 2d array in table column
      - using alternative data container like IndexedTable.jl
- train and test splits.
  - Do we use sktime ts way to store data or csv.
  - Can we put in train and test in single file and finally split before fit or predict.
  - combine train and test data into single file and single load function

### Reflection


## Week 3
1 June - 8 June

### Actions
- [x] try out passing data container to sktime via PyCall, add to Jupyter notebook
- [x] use MLJ instead of DecisionTree package
- [x] make sure our models work in MLJ composition classes like pipelines and tuning (e.g. tuning the number of estimators), add to Jupyter notebook
- [x] combine train and test data into single matrix and single load function
- [ ] try out 3 options: 
    1. 3d array (matrix),
    2. 2d array in column in Tables,
    3. IndexedTables.jl
- [ ] @vollmersj, think about data container and make decision
- [ ] ask Anthony for feedback on data container
- [ ] write wrapper for sktime
- [ ] add introduction and explanatory notes to example notebook

### Questions
* PyCall does not work, because of `pd.DataFrame` requirement in sktime, would work if sktime accepted 3d numpy container
* IndexTable.jl is working, but does not unequal-length data (only via nan values)
* @vollmersj How do you make models compatible with `TunedModel` in MLJ?

### Reflection 

### Notes

#### work-around for pycall with nested pd.DataFrame
1. 3d matrix in Julia, let's say `X`
2. pycall and we pass `X` 
3. from `sktime.utils.data_container import detabularize`
4. `X = detabularize(X)`
5. call `fit(X)`


## Week 4
8 June - 15 June

### Actions
- [x] write tutorial notebook similar to this one: https://alan-turing-institute.github.io/sktime/examples/02_classification_univariate.html ("tutorial-driven development"), we can build website later, the notebook will also be good for a blog posts
- [x] write data loading function that works with univariate data sets from timeseriesclassification.com


milestone: 2-5 to be completed by Mon, 15th of June

### Questions
- How to identify univariate datasets from        http://timeseriesclassification.com/dataset.php

### Reflection 
- Some of code is unix based (TSdataset), find their parallel in win. 

## Week 4
15 June - 22 June

### Actions
- [x] write `load_ts_file()` function to load `.ts` files into Julia, load `.ts` file from hard drive and return IndexedTable without intermediate step of writing csv file
- [x] please push notebook online into examples folder, Jupyter notebook is fine for now
- [x] make sure our models work in MLJ composition classes like pipelines and tuning (e.g. tuning the number of estimators), and `evaluate!` function
- [x] write tutorial notebook similar to this one: https://alan-turing-institute.github.io/sktime/examples/02_classification_univariate.html ("tutorial-driven development"), we can build website later, the notebook will also be good for a blog posts

milestone: 1-2 (possibly also 3) to be completed by Mon, 22nd of June

### Questions

### Reflection 
- [ ] start writing daily progress in #jsco-stand-up channel
- [x] open draft PR on GitHub to give feedback

## Week 5
22 June - 29 June

### Actions

- [x] write TimeSeriesForestClassifier into model so that we can pass it to machine
- [ ] write script to benchmark MLJTime algorithms on univariate time series classification data sets
    - data sets: http://www.timeseriesclassification.com/dataset.php
    - script from sktime:https://github.com/alan-turing-institute/sktime/blob/master/sktime/contrib/experiments.py
- [x] benchmark TSF on univariate data (accuracy and run time)
- [ ] build website based on example notebook and Franklin, chat to Thibaut on slack

### Notes
```julia
using MLJTime
using MLJBase:partition
using StableRNGs

# we almost have this one, this one loads .ts files and returns IndexedTables
X, y = ts_dataset("datasets", "arrow_head")

# we have to make tsf a model so that we can pass it to machine()
tsf = RandomForestClassifierTS()
machine = machine(tsf, X, y)

# we've got this
rng = StableRNG(566)
train, test = partition(eachindex(y), 0.7, shuffle=true, rng=rng)

# once we have the machine, this should work
fit!(machine, rows=train)
ŷ = predict(machine, rows=test)
```

### Reflections
- [x] daily 5-min call to show progress
- [ ] improve written progress reporting
- [ ] summarize progress on #ml-jsoc channel

## Week 6
29 June - 6 July

### Actions

- [x] write TimeSeriesForestClassifier into model so that we can pass it to machine
- [ ] write script to benchmark MLJTime algorithms on univariate time series classification data sets
    - data sets: http://www.timeseriesclassification.com/dataset.php
    - script from sktime:https://github.com/alan-turing-institute/sktime/blob/master/sktime/contrib/experiments.py
- [ ] benchmark TSF on univariate data (accuracy and run time)
- [ ] build website based on example notebook and Franklin, chat to Thibaut on slack

### Notes

- [ ] conversion/promotion for indexed tables
- [x] Tuning 
- [ ] Piplines - where is input output - https://github.com/alan-turing-institute/MLJBase.jl/blob/d93ef2f1095d9037af116b5d9d861e8ae7e6821b/src/composition/models/pipelines.jl#L498 
- [ ] boxcox transform or log transform MLJ transformers and apply to time series before classification
- [ ] plug and play like in https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/learning-networks/index.html should work 
- [x] warning message `WARNING: Method definition (::Type{MLJTime.RandomForestClassifierTS})() in module MLJTime overwritten.` -- incremental compilation may be fatally broken for this module *
- [ ] http://docs.junolab.org/stable/man/debugging/ step through functions to see what gets called where
- [x] resampling 
- [ ] apply transformations e.g. `X, y = @load_iris
pca = @load PCA
mach = machine(pca, X)
fit!(mach)`
- [x] Notebook with machine, fit! and predict. 
- [x] y should be an array.

https://github.com/Evovest/EvoTrees.jl/blob/master/src/MLJ.jl 
From Anthony Blaom to Everyone: (9:39 am)
 https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/ https://github.com/alan-turing-institute/MLJFlux.jl 
 
 
 From Anthony Blaom to Everyone: (9:34 am)
 https://github.com/Evovest/EvoTrees.jl/blob/master/src/MLJ.jl 
From Anthony Blaom to Everyone: (9:39 am)
 https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/ https://github.com/alan-turing-institute/MLJFlux.jl 
From aadeshdeshmukh to Everyone: (10:06 am)
 ###################Transformation Algorithm  Tabularizer SFA, PAA, SAX PCA ColumnTransformer, RowwiseTransformer,  ColumnConcatenator, IntervalSegmenter RandomIntervalSegmenter, ShapeletTransform,  PlateauFinder, DerivativeSlopeTransformer, RandomIntervalFeatureExtractor  ####################Classification Algorithm   ColumnEnsembleClassifier, HomogeneousColumnEnsembleClassifier BOSSEnsemble ElasticEnsemble, CachedTransformer  (ProximityStump, Proximity Tree) Proximity Forest RandomIntervalSpectralForest, TimeSeriesForest ShapeletTransformClassifier  ####################Prediction Algorithm BaseForecaster BaseUpdateableForecaster BaseSingleSeriesForecaster TransformedTargetForecaster ReducedRegressionForecaster ARIMAForecaster DummyForecaster
 
 ## Examples given by the A.Bloam
 
#### EvoTrees 
- Does not have the MLJ kind of Abstration eg. fit predict, At least that is what is get form examples
- here they don't disturb the `machine` and `evaluate` form MLJBase

#### MLJFlux
- This pkg does not use MLJbase at all
- It severs as the helper to MLJModles(I guess)

#### MLJFair
- As AB said in this pkg we have networking part figured out, i coun't find where it is implemented, Will be great if Sebastian can point it to
- I have seen how this pkg useas the network, should look at it while developing pipeline 

#### We still have warning massage!
- It went away for some time, But then other part of Pkg was not working  eg. matrix.
- They are useing Paramerts Pkg, should we consider it?

#### Predict form machine is haveing problem
- iterator for DecisionTree.DecisionTreeClassifier is missing 
- Debugger do not load the `julia code from pkg` many times
- We have to ceck how Array of model is put into the predict from the `MLJBase.predict`

#### For our website/Blog
- can't use nbconvert as anacoda doesn't work on Big sur 
- finding way to in-corporate the unregistered pkg in Franklin

#### Once we have basic things, Making the pkg clean and readable is important
- adding commants and filing the issues what doesn't work.
- `julia> predict(forest, X_new)
ERROR: MethodError: no method matching iterate(::DecisionTree.DecisionTreeClassifier)
Closest candidates are:
  iterate(::Core.SimpleVector) at essentials.jl:603
  iterate(::Core.SimpleVector, ::Any) at essentials.jl:603
  iterate(::ExponentialBackOff) at error.jl:253
  ...
Stacktrace:
 [1] proba_predict(::DecisionTree.DecisionTreeClassifier, ::Array{Float64,1}) at /Users/aadeshdeshmukh/.julia/dev/MLJTime/src/IntervalBasedForest.jl:87`
 
- [ ] Is there way to build RandomForestClassifier() with different data set rather than single? 

- [x] I am not 100% sure how sktime utilises the RandomForestClassifier from scikit learn, may be we should talk about it (Only in case if tsf algorithm)

#### MLJ can directly operates with IndexedTables only Matrix <-> Table conversion is required (We have function avilable for the same). 

#### We have to add the MLJMeasure for tuning to work and export resampling for CVs.

#### Other wise most of the MLJTuning is working with our algoritham.

#### CategoricalArrays will be added by today.


## Week 7
6 July - 13 July

### Actions
- [x] y should be array, not IndexedTable when loading from `ts_dataset()`, fix `fit!(machine)` to work with y as array
 - [x] rename `RandomForestClassifierTS` to `TimeSeriesForestClassifier` 
- [x] fix TimeSeriesForestClassifier
- [x] add China Town data set (smallest data set, great for notebooks and unit testing): http://www.timeseriesclassification.com/description.php?Dataset=Chinatown
- [x] deprecation warning, perhaps talk to Thibault what the new intended workflow is
- [ ] Refactoring MLJTime, cleaning the code and making it more readable, adding comments and docstrings, and meaningful variable names
- [ ] Start writing blog post and example notebook
- [ ] Give short summary in #jsoc-standup

### Notes

## Week 8
13 July - 20 July
*Exam period: Aadesh starts his exams on 15 July*

## Week 9
20 - 27 July
*Exam period*

## Week 10
27 - 3 August

### Actions
- [ ] benchmarking MLJTime's TimeSeriesForestClassifier (TSF) on all univariate data sets from timeseriesclassification.com, validating our implementation against the Java reference implementation, speed comparison
    * accuracy
    * timing for fit and predict
    * make comparable to sktime (number of trees, extracted features, number of intervals)
    * no tuning for now
    * see https://github.com/mloning/sktime-benchmarking
- [x] writing unit test to make sure published TSF results are reproducible
- [x] adding support for multivariate time series classification algorithms
- [ ] composition classes: pipelines (see e.g. `Tabularizer()` in sktime)
- [ ] adding algorithms other than TimeSeriesForest, factoring out common data preprocessing steps (e.g. interval segmentation, feature extration, etc)


## Week 11
3 - 10 August

### Actions
- [x] benchmarking MLJTime's TimeSeriesForestClassifier (TSF) on all univariate data sets from timeseriesclassification.com, validating our implementation against the Java reference implementation, speed comparison
    * accuracy
    * timing for fit and predict
    * make comparable to sktime (number of trees, extracted features, number of intervals)
    * no tuning for now
    * see https://github.com/mloning/sktime-benchmarking
- [x] writing unit test to make sure published TSF results are reproducible
- [ ] adding support for multivariate time series classification algorithms
- [ ] composition classes: pipelines (see e.g. `Tabularizer()`) in sktime
- [x] investigate why results still vary in TSF even when setting the random state
- [ ] adding algorithms other than TimeSeriesForest, factoring out common data preprocessing steps (e.g. interval segmentation, feature extration, etc)
- [x] implement KNN with DTW

## Week 12
10 - 17 August

### Actions
- [ ] composition classes: pipelines (see e.g. `Tabularizer()`) in sktime
- [x] benchmarking MLJTime's TimeSeriesForestClassifier (TSF) on all univariate data sets from timeseriesclassification.com, validating our implementation against the Java reference implementation, speed comparison
- [ ] forecasting: documentation-driven development, start with real-world example, online/streaming data, continuously updating and making forecasting
- [ ] data simulator/generator


### Notes
* The data series part of the Metric is not clear in lib we are using
* Making KDTree, Balltree; There is some problem with KDTree as it doesn’t accepts the generic metric
* Making predictions using the threes, Which package is good for such thing in Julia for this?
* Have not to fully read through the sklearn algorithm  
* Markus makes concrete example form specific data set to the Accuracy in Python so, I can compare my Julia code in parallel
* Variable W in the algorithm is not specific please can anyone explain it. - ML: window for warping

#### Run benchmarking
* change path in [`benchmark/tsf.jl`](https://github.com/alan-turing-institute/MLJTime.jl/blob/master/benchmark/tsf.jl)
* to install julia: `wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.0-linux-x86_64.tar.gz | tar -xzv`
* to add packages `] add link` 
* to run script: `julia-1.5.0/bin/julia -p 32 script.jl`
* to get data `wget http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip`

#### Tabularizer
```python
classifier = TabularClassifier(DecisionTreeClassifier())
classifier.fit(X_train, y_train) # transform IndexedTable to plain Table
y_pred = classifier.predict(X_test)
```

## Week 13
17 - 24 August

### Actions
- [ ] composition classes: pipelines (see e.g. `Tabularizer()`) in sktime
- [x] benchmarking MLJTime's TimeSeriesForestClassifier (TSF) on all univariate data sets from timeseriesclassification.com, validating our implementation against the Java reference implementation, speed comparison

## Notes
BOSS with SFA
1> What is threshold 
2> Randomised_ensemble what gets shuffled?
      either through randomising over the para
      space to make a fixed size ensemble quickly or by creating a
      variable size ensemble of those within a threshold
      of the best
3> The length of window is based on count of points or length on time axis 
4> Concept of the weights in the predicting?
5> Should we keep all of our algorithms classification probabilistic ?
BOSS ->  BOSSindividual  ->  SFA (BitWord, sklearn import preprocessing)


## Week 14
24 - 31 August 

- boxplot for visualising the distribution of timing ratio, summarize in words, e.g. "Julia is x times faster than Python", first plot: combine fit and predict timings, second plot: distinguish between fit and predict timings, finally discuss limitation ("it may not be a fair comparison because we haven't optimised the Julia code ...")
- time comparison with ordered datasets on x-axis (ordered by size or avg train time)
- scatter plot with diagonal line, x-axis is Python, y-axis is Julia accuracy
- hypothesis testing: wilcoxon signed rank test, binomial test (see https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)


## 2nd Report (blog post)
*26th of August 2020*
* notebook
* comparative benchmarking for TSF, optionally KNN
* statistical analysis of results

## Final milestone: work product
*30th of August 2020*
https://developers.google.com/open-source/gsoc/help/work-product

- [x] TimeSeriesForest 
- [x] KNN with dynamic time warping
- [ ] Tabularizer to use MLJTime with MLJ (see above)
- [x] Benchmarking against Python implementions (accuracy and speed)
- [ ] Dictionary-based or shapelets
- [ ] update README with links to blog posts and any notes for users/contributors 
- [ ] Box plot
- [ ] Wilconxon dist
- [ ] two types of 
- [x] add README with links to blog posts, MLJ parent project, acknowledgements, install instructions, how to contribute etc
- [ ] package manager 

## 3rd Report (blog post)
*7th of September*
* dictionary-based classification
* KNN
* Tabularizer
* overall summary of work
* outlook of future work

## Future work
- [ ] extend KNN to multivariate/unequal data
- [ ] add KDTree and BallTree to KNN
- [ ] forecasting




