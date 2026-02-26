from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import os


# Define a function to train the model using GridSearchCV and StratifiedKFold
def train_model(model, param_grid, X_train, y_train):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=skf, verbose=2, n_jobs=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search


# Define a function to evaluate the model on the test set
def evaluate_model(model, X_test, y_test, classes, model_name, artifacts_dir="artifacts"):
    y_pred = model.predict(X_test)
 
    report = classification_report(y_test, y_pred, target_names=classes, digits=3, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=classes, digits=3))
  

    acc = accuracy_score(y_test, y_pred)
    cf = confusion_matrix(y_test, y_pred)

    os.makedirs(artifacts_dir, exist_ok=True)
    confmat_path = os.path.join(artifacts_dir, f"{model_name}_confusion_matrix.png")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cf, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(confmat_path, bbox_inches='tight')
    plt.close()
    

    metrics = {
        'accuracy': acc,
    }
    if 'macro avg' in report:
        metrics['f1_macro'] = report['macro avg'].get('f1-score', None)
        metrics['precision_macro'] = report['macro avg'].get('precision', None)
        metrics['recall_macro'] = report['macro avg'].get('recall', None)
    if 'weighted avg' in report:
        metrics['f1_weighted'] = report['weighted avg'].get('f1-score', None)

    return metrics, confmat_path

