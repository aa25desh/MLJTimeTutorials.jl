@def title = "Data Loading"
@def hascode = true
@def date = Date(2019, 3, 22)
@def rss = "A short description of the page which would serve as **blurb** in a `RSS` feed; you can use basic markdown here but the whole description string must be a single line (not a multiline string). Like this one for instance. Keep in mind that styling is minimal in RSS so for instance don't expect maths or fancy styling to work; images should be ok though: ![](https://upload.wikimedia.org/wikipedia/en/3/32/Rick_and_Morty_opening_credits.jpeg)"

@def Benchmark = ["syntax", "code"]

# Working with Data Loading
`univariate_datasets` is the list of all dataset available.
```julia:./data.jl
using MLJTime
@show univariate_datasets
```

\output{./data.jl}
\toc


### Online Download
`TSdataset(dataset::Array)` function adds new data set from [timeseriesclassification.com.](http://timeseriesclassification.com) to "data" folder for the MLJTime directory.
Make sure that your path must be in the `MLJTime` directory.
For example take `ACSF1`
```julia:./data.jl
TSdataset(["ACSF1"])
```

### With TS Files
`TSdataset(filepath::String)` fucntion converts `.ts` files to `.csv` files
& add in same folder so one create julia data. `Path to the Dir` will be inpute
in this case.
```julia:./data.jl
TSdataset(""/Users/abc/MLJtime_pdf_data/GunPoint")
```

## Julia Matrix & IndexedTables as Data Container.

### Inbuilt Data sets to play
The famous dataset in TimeSeries are made available to pluge and play direcaly
