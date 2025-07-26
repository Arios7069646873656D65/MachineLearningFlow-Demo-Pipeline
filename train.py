import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

mlflow.set_experiment("Iris LogisticRegression")

for C in [0.01, 0.1, 1.0, 10.0]:
    with mlflow.start_run():

        mlflow.log_param("C", C)

        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(x_train,y_train)

        y_pred =model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

    print("C=", C, "accuracy=", acc)