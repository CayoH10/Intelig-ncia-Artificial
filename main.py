import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("bank-full.csv", sep=";")

X = df.drop("y", axis=1)
y = df["y"]


y = y.map({
    "no":0,
    "yes":1
})

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()


c = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

print("Formato de X:")
print(X.shape)

#print("\nPrimeiras linhas:")
#print(X.head())

print("\nPrimeiros valores da classe:")
print(y.head())

for i, (train, test) in enumerate(c.split(X, y), start=1):
    print(f"Fold {i}")
    print(f"Treino: {len(train)} exemplos")
    print(f"Teste : {len(test)} exemplos")
    print("-" * 30)