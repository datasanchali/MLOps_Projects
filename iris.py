import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# train a random forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# save the trained model to a file
joblib.dump(model, "model.joblib")
