from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print("Linear Kernel Accuracy:",
      accuracy_score(y_test, y_pred_linear))
print("Linear Kernel Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_linear))
rbf_model = SVC(kernel='rbf')
rbf_model.fit(X_train, y_train)
y_pred_rbf = rbf_model.predict(X_test)
print("\nRBF Kernel Accuracy:",
      accuracy_score(y_test, y_pred_rbf))
print("RBF Kernel Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))