# This file was generated, do not modify it. # hide
using MLJTime
train, test = load_gunpoint()
X_train, y_train = X_y_split(train)
X_test, y_test = X_y_split(test)
forest = RandomForestClassifierTS(X_train, y_train)
@show predict_new(X_test, forest)