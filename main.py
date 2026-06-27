import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

def avaliar_modelo(modelo, X, y, c):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for train_index, test_index in c.split(X, y):

        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    print(f"\n=== {modelo.__class__.__name__} ===")
    print(f"Accuracy : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall   : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1-Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

df = pd.read_csv("bank-full.csv", sep=";")

X = df.drop("y", axis=1)
y = df["y"]


y = y.map({
    "no":0,
    "yes":1
})

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)


c = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

accuracies = []
precisions = []
recalls = []
f1s = []

print("Formato de X:")
print(X.shape)

#print("\nPrimeiras linhas:")
#print(X.head())

print("\nPrimeiros valores da classe:")
print(y.head())

modelo = DecisionTreeClassifier(random_state=42)
avaliar_modelo(modelo, X, y, c)
