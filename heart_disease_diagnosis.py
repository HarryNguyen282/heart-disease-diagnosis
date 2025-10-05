
import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)


def read_csv(file_path):
    df = pd.read_csv(file_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def evaluate_val(model_name, X_train, y_train, X_val, y_val, k=0, depth=0):
    if model_name == "NB":
        model = GaussianNB()
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_name == "Tree":
        model = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    elif model_name == "KMeans":
        model = KMeans(n_clusters=2, random_state=SEED)
    elif model_name == "ensemble":
        ensemble_model = StackingClassifier(
            estimators=[
                ('knn', KNeighborsClassifier(n_neighbors=k)),
                ('dt', DecisionTreeClassifier(max_depth=depth, random_state=SEED)),
                ('nb', GaussianNB())
            ],
            final_estimator=KNeighborsClassifier(n_neighbors=k),
            stack_method='predict_proba',
            passthrough=False,
            cv=5,
        )
        model = ensemble_model

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    return model, val_acc

def evaluate_test(model, X_test, y_test):
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    classification_rep = classification_report(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    return test_acc, classification_rep, cm


def find_optimal_k(X_train, y_train, X_val=None, y_val=None, k_range=range(1,21), cv_splits=5):

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_score = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        k_scores.append(np.mean(cv_score))

    optimal_k = k_range[np.argmax(k_scores)]
    return optimal_k

def find_optimal_depth(X_train, y_train, X_val=None, y_val=None, depth_range=range(1,11), min_depth=2, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    depth_scores = []
    for depth in range(min_depth, depth_range.stop):
        tree = DecisionTreeClassifier(max_depth=depth)  
        cv_score = cross_val_score(
            tree, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )

        depth_scores.append(np.mean(cv_score))
    
    optimal_depth = depth_range[np.argmax(depth_scores)]
    return optimal_depth


def display_results(accuracy, accuracy_fe, test_accuracy, test_accuracy_fe):
    print(f"training accuracy on Original Dataset: {accuracy}")
    print(f"training accuracy on Feature Engineered Dataset: {accuracy_fe}")
    print(f"testing accuracy on Original Dataset: {test_accuracy}")
    print(f"testing accuracy on Feature Engineered Dataset: {test_accuracy_fe}")
    labels = ["Original Dataset", "Feature Engineered Dataset"]
    val_acc = [accuracy, accuracy_fe]
    test_acc = [test_accuracy, test_accuracy_fe]
    x = np.arange(len(labels)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, val_acc, width, label='Validation Accuracy', color='skyblue', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='lightgreen', edgecolor='black', linewidth=1)

    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def add_data_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_data_labels(rects1)
    add_data_labels(rects2)

    plt.tight_layout()
    plt.show()

def main():
    X_train, y_train = read_csv('splits/raw_train.csv')
    X_val, y_val = read_csv('splits/raw_val.csv')
    X_test, y_test = read_csv('splits/raw_test.csv')

    X_train_fe, y_train_fe = read_csv('splits/fe_train.csv')
    X_val_fe, y_val_fe = read_csv('splits/fe_val.csv')
    X_test_fe, y_test_fe = read_csv('splits/fe_test.csv')

    optimal_k = find_optimal_k(X_train, y_train, X_val, y_val)
    optimal_k_fe = find_optimal_k(X_train_fe, y_train_fe, X_val_fe, y_val_fe)

    optimal_depth = find_optimal_depth(X_train, y_train, X_val, y_val)
    optimal_depth_fe = find_optimal_depth(X_train_fe, y_train_fe, X_val_fe, y_val_fe)

    models = ['knn', 'NB', 'Tree', 'KMeans', 'ensemble']
    results = defaultdict(dict)
    for m in models:
        LM_model, accuracy = evaluate_val(m, X_train, y_train, X_val, y_val, optimal_k, optimal_depth)
        test_acc, classification_rep, cm = evaluate_test(LM_model, X_test, y_test)
        
        model_fe, accuracy_fe = evaluate_val(m, X_train_fe, y_train_fe, X_val_fe, y_val_fe, optimal_k_fe, optimal_depth_fe)
        test_acc_fe, classification_rep_fe, cm_fe = evaluate_test(model_fe, X_test_fe, y_test_fe)

        results[m]['Original'] = test_acc
        results[m]['Feature_Engineered'] = test_acc_fe

        print()
        print("-"*50)
        print(f"Result for model: {m}")
        display_results(accuracy, accuracy_fe, test_acc, test_acc_fe)

    results_df = pd.DataFrame(results).T
    results_df.columns = ['Test Accuracy on Original Dataset', 'Test Accuracy on Feature Engineered Dataset']
    results_df = results_df.sort_values(by='Test Accuracy on Feature Engineered Dataset', ascending=False)
    print(results_df)

if __name__ == "__main__":
    main()
