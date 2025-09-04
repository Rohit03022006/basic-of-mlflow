import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine dataset
data = load_wine()
X = data.data
y = data.target

# Split test and split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
print(X_train.shape,  X_test.shape, y_train.shape, y_test.shape)

max_depth = 5 # Maximum depth of the tree for base learners in the forest
n_estimators = 1 # Number of trees in the forest

print("Training RandomForestClassifier...")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("Wine_Quality_Classification") # Ek experiment banaya jiska naam hai

# Start an MLflow run
with mlflow.start_run():
    # Create and train the model
    crf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    crf.fit(X_train, y_train)

    # Make predictions
    y_pred = crf.predict(X_test)
    print("Predictions:", y_pred)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=data.target_names, yticklabels=data.target_names)
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # Log parameters and metrics to MLflow
    mlflow.log_metric("accuracy", accuracy) 
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)


    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    # Log the model to MLflow

    # tags
    mlflow.set_tag("model", "RandomForestClassifier")
    mlflow.set_tag("dataset", "Wine Quality")
    mlflow.set_tag("type", "classification")
    mlflow.set_tag("developer", "Rohit")

    # Log the model to MLflow
    mlflow.sklearn.log_model(crf, "random_forest_model")