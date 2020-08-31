# This file was generated, do not modify it. # hide
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