@def title = "Time Series Forest"
@def hascode = true
@def rss = "A short description of the page which would serve as **blurb** in a `RSS` feed; you can use basic markdown here but the whole description string must be a single line (not a multiline string). Like this one for instance. Keep in mind that styling is minimal in RSS so for instance don't expect maths or fancy styling to work; images should be ok though: ![](https://upload.wikimedia.org/wikipedia/en/b/b0/Rick_and_Morty_characters.jpg)"
@def rss_title = "Univariate time series classification "
@def rss_pubdate = Date(2019, 5, 1)

@def Benchmark = ["syntax", "code", "image"]

# Univariate time series classification

\toc

## example

```julia:./univariat.jl
using MLJTime

X, y = ts_dataset("Chinatown")
unique(y)

train, test = partition(eachindex(y), 0.7, shuffle=true, rng=1234) #70:30 split
X_train, y_train = X[train], y[train];
X_test, y_test = X[test], y[test];

model = TimeSeriesForestClassifier(n_trees=3)
mach = machine(model, X_train, y_train)
fit!(mach)

y_pred = predict(mach, X_test)
```

\output{./univariat.jl}
