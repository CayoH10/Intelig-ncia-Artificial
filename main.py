import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

def buscar_melhor_modelo(nome_modelo, modelo, parametros, X, y):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", clone(modelo))
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=10,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X, y)

    print(f"\n=== {nome_modelo} ===")
    print("Melhores parâmetros:")
    print(grid.best_params_)
    print(f"Melhor F1: {grid.best_score_:.4f}")

    resultados = pd.DataFrame(grid.cv_results_)

    colunas = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]

    resultados = resultados[colunas]

    print("\nTop 5 resultados do GridSearch:")
    print(resultados.sort_values("rank_test_score").head())

    resultados.to_csv(
        f"gridsearch_{nome_modelo}.csv",
        index=False
    )

    return grid.best_estimator_

def avaliar_modelo(modelo, X, y, c):

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for train_index, test_index in c.split(X, y):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        pipeline = clone(modelo)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    accuracy = np.mean(accuracies)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    std_accuracy = np.std(accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1s)

    print(f"Accuracy : {accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision: {precision:.4f} ± {std_precision:.4f}")
    print(f"Recall   : {recall:.4f} ± {std_recall:.4f}")
    print(f"F1-Score : {f1:.4f} ± {std_f1:.4f}")

    return {
        "Accuracy": accuracy,
        "Accuracy_std": std_accuracy,
        "Precision": precision,
        "Precision_std": std_precision,
        "Recall": recall,
        "Recall_std": std_recall,
        "F1": f1,
        "F1_std": std_f1
    }



df = pd.read_csv("bank-full.csv", sep=";")

X = df.drop("y", axis=1)
y = df["y"]


y = y.map({
    "no":0,
    "yes":1
})

X = pd.get_dummies(X, drop_first=True)



c = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)



print("Formato de X:")
print(X.shape)



print("\nPrimeiros valores da classe:")
print(y.head())

parametros = {

    "DecisionTree": {
        "modelo__criterion": ["gini", "entropy"],
        "modelo__max_depth": [3, 5, 10]
    },

    "KNN": {
        "modelo__n_neighbors": [3, 5, 7]
    },

    "NaiveBayes": {
       
        "modelo__var_smoothing": [1e-9, 1e-8, 1e-7]
    },

    "LogisticRegression": {
        "modelo__C": [0.1, 1, 10]
    },

    "SVM": {
        "modelo__C": [0.1, 1, 10],
        "modelo__kernel": ["linear", "rbf"]
    },

    "MLP": {
        "modelo__hidden_layer_sizes": [(50,), (100,), (150,)],
        "modelo__alpha": [0.0001, 0.001]
    }
}

modelos = [

    ("DecisionTree",
     DecisionTreeClassifier(random_state=42)),

    ("KNN",
     KNeighborsClassifier()),

    ("NaiveBayes",
     GaussianNB()),

    ("LogisticRegression",
     LogisticRegression(max_iter=1000,
                        random_state=42)),

    ("SVM",
     SVC(random_state=42)),

    ("MLP",
     MLPClassifier(max_iter=200,
                   random_state=42))
]


resultados_finais = []

for nome, modelo in modelos:

    print(f"\n{'='*50}")
    print(nome)
    print('='*50)

    melhor_modelo = buscar_melhor_modelo(
        nome,
        modelo,
        parametros[nome],
        X,
        y
    )

    resultado = avaliar_modelo(
        melhor_modelo,
        X,
        y,
        c
    )

    resultado["Algoritmo"] = nome

    resultados_finais.append(resultado)

df_resultados = pd.DataFrame(resultados_finais)

print("\n===== RESULTADOS FINAIS =====")
print(df_resultados)

df_resultados.to_csv(
    "resultados_finais.csv",
    index=False
)