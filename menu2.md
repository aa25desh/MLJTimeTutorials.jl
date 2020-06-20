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
train, test = load_gunpoint()
X_train, y_train = X_y_split(train)
X_test, y_test = X_y_split(test)
forest = RandomForestClassifierTS(X_train, y_train)
@show y_predict = predict_new(X_test, forest)
@show y_test
@show L1(y_predict, y_test)
```

\output{./univariat.jl}
