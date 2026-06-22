import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

def train_model(n_estimators):
    print("--- Starting Iris Model Training ---")
    
    # 1. Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Train the Model
    print(f"Training RandomForestClassifier with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # 5. Ensure outputs/ directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # 6. Save the trained model using joblib
    model_path = 'outputs/iris_model.joblib'
    joblib.dump(model, model_path)
    print(f"Successfully saved model to: {model_path}")
    
    # 7. Generate and save the Confusion Matrix as a PNG
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    
    # Create a nice visual heatmap for the confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=iris.target_names, 
        yticklabels=iris.target_names
    )
    plt.title('Iris Classifier Confusion Matrix')
    plt.ylabel('True Species')
    plt.xlabel('Predicted Species')
    plt.tight_layout()
    
    # Save the chart image
    matrix_path = 'outputs/confusion_matrix.png'
    plt.savefig(matrix_path)
    plt.close() # Closes the figure plot to free up memory
    print(f"Successfully saved confusion matrix plot to: {matrix_path}")
    print("--- Training Process Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris Classifier Script")
    parser.add_argument('--estimators', type=int, default=100, help="Number of trees")
    args = parser.parse_args()
    
    train_model(args.estimators)