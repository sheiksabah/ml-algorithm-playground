import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

iris = load_iris()
X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

def evaluate_model(name, hidden_layers, activation="relu",
                   solver="adam", max_iter=3000):
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n{name}")
    print("Hidden layers:", hidden_layers)
    print(f"Training accuracy: {train_acc * 100:.2f}%")
    print(f"Testing accuracy:  {test_acc * 100:.2f}%")

    return {
        "name": name,
        "layers": hidden_layers,
        "train_acc": train_acc,
        "test_acc": test_acc
    }

baseline = evaluate_model(
    "Baseline MLP",
    hidden_layers=(3,),
    activation="relu",
    solver="adam"
)

tuned = evaluate_model(
    "Tuned MLP",
    hidden_layers=(3,),
    activation="tanh",
    solver="lbfgs"
)

architectures = [
    (5,),
    (10,),
    (10, 5),
    (15, 10),
    (20, 10)
]

complex_results = []

for architecture in architectures:
    result = evaluate_model(
        f"Complex MLP {architecture}",
        hidden_layers=architecture,
        activation="tanh",
        solver="lbfgs"
    )
    complex_results.append(result)
print("\nTest accuracy comparison:")
print(f"Baseline (3 neurons): {baseline['test_acc'] * 100:.2f}%")
print(f"Tuned (3 neurons):    {tuned['test_acc'] * 100:.2f}%")

for result in complex_results:
    print(
        f"{result['name']}: "
        f"{result['test_acc'] * 100:.2f}%"
    )

best_complex = max(
    complex_results,
    key=lambda result: result["test_acc"]
)

print("\nBest complex architecture:", best_complex["layers"])
print(f"Best complex test accuracy: {best_complex['test_acc'] * 100:.2f}%")

if best_complex["test_acc"] > baseline["test_acc"]:
    print("The best complex model improved over the baseline.")
else:
    print("No tested complex model improved over the baseline.")
