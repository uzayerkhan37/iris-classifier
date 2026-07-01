import os
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def train_model():
    # 1. Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")
    
    # 5. Save model artifact
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/iris_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {model_path}")
    
    # 6. Generate and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap=plt.cm.Blues)
    
    matrix_path = "outputs/confusion_matrix.png"
    plt.savefig(matrix_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Success: Confusion matrix saved to {matrix_path}")

if __name__ == "__main__":
    train_model()
