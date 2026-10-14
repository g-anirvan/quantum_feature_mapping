
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Generate a chaotic time series using the logistic map
n_steps = 600
x = np.zeros(n_steps)
x[0] = 0.2
r = 3.9

for t in range(1, n_steps):
    x[t] = r * x[t - 1] * (1 - x[t - 1])

plt.figure(figsize=(10, 3))
plt.plot(x[:100], color="green")
plt.title("Logistic Map: Chaotic Time Series")
plt.xlabel("Time Step")
plt.ylabel("x(t)")
plt.grid(alpha=0.3)
plt.show()

# 2. Build a fixed quantum feature circuit
n_qubits = 4
input_param = Parameter("input_val")

qc = QuantumCircuit(n_qubits)

# Encode the input in the qubits
for i in range(n_qubits):
    qc.ry(input_param * (i + 1), i)

# Entangling layer
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.cx(3, 0)

# 3. Extract probability features using statevector simulation
features = []

for val in x:
    bound_qc = qc.assign_parameters({input_param: float(val)})
    sv = Statevector.from_instruction(bound_qc)
    features.append(sv.probabilities())

features = np.array(features)

print("Quantum feature matrix:", features.shape)

# 4. Prepare one-step-ahead prediction data
X = features[:-1]
y = x[1:]

train_size = 400

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Quantum-feature regression model
quantum_model = Ridge(alpha=1.0)
quantum_model.fit(X_train, y_train)
y_quantum = quantum_model.predict(X_test)

# 6. Classical polynomial baseline
# The logistic map is a quadratic function of x(t).
X_classical = x[:-1].reshape(-1, 1)

Xc_train = X_classical[:train_size]
Xc_test = X_classical[train_size:]

classical_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1e-8)
)

classical_model.fit(Xc_train, y_train)
y_classical = classical_model.predict(Xc_test)

# 7. Naive persistence baseline
y_naive = x[train_size:-1]


# classical Ridge regression on the raw input \(x_t\) as an additional baseline.

classical_linear = Ridge(alpha=1.0)

X_linear = x[:-1].reshape(-1, 1)

classical_linear.fit(X_linear[:train_size],y_train)

y_linear = classical_linear.predict(X_linear[train_size:])

print("Classical linear RMSE:", f"{rmse(y_test, y_linear):.6f}")

# 8. Evaluate all models
def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

print(f"Quantum-feature model RMSE: {rmse(y_test, y_quantum):.6f}")
print(f"Classical quadratic RMSE:   {rmse(y_test, y_classical):.6f}")
print(f"Naive baseline RMSE:        {rmse(y_test, y_naive):.6f}")

# 9. Plot the first 100 test predictions
n_plot = 100

plt.figure(figsize=(11, 4))
plt.plot(y_test[:n_plot], label="Actual", color="black")
plt.plot(y_quantum[:n_plot], "--", label="Quantum features")
plt.plot(y_classical[:n_plot], ":", label="Classical quadratic")
plt.plot(y_naive[:n_plot], alpha=0.6, label="Naive baseline")

plt.xlabel("Test Time Step")
plt.ylabel("Predicted x(t+1)")
plt.title("One-Step-Ahead Prediction: Model Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()  
