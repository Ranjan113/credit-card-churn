import pandas as pd
import numpy as np
import json
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc, 
    confusion_matrix, ConfusionMatrixDisplay
)
     

TARGET_COLUMN = 'Attrition_Flag'

# create a logger
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.INFO)

# create a handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error('the file to load does not exist')
        raise

def calculate_metrics(y_true, y_pred, y_pred_proba):
    prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = float(auc(rec, prec))
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'pr_auc': pr_auc
    }
    return metrics

def save_metrics(metrics_dict: dict, save_path: Path):
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def make_X_and_y(data: pd.DataFrame, target_column:str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X,y

def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


def plot_confusion_matrix(y_true, y_pred, save_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path: Path, title: str):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC-AUC Score ({title}):", roc_auc)

def plot_pr_curve_and_print(y_true, y_pred_proba, save_path: Path, title: str):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {title}")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR-AUC Score ({title}):", pr_auc)




if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    # data and model paths
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    model_path = root_path / "models" / "model.joblib"

    # output directory
    figures_dir = root_path / "reports" / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    # load test data
    test_data = load_data(test_data_path)
    logger.info('test data loaded successfully')

    # split X/y
    X_test, y_test = make_X_and_y(test_data, TARGET_COLUMN)
    logger.info('test data splitting completed')

    # load model
    model = load_model(model_path)
    logger.info('model loaded successfully')

    # get predictions and positive class probabilities
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1] 
    logger.info('test predictions completed')

    # calculate metrics (test only)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    logger.info('test metrics calculated successfully')

    # save metrics
    metrics_path = root_path / "metrics.json"
    save_metrics({"test_metrics": test_metrics}, metrics_path)
    logger.info('test metrics saved successfully')

    # test confusion matrix
    test_cm_path = figures_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_test_pred, test_cm_path, "Test Set")
    logger.info(f'Test confusion matrix saved to {test_cm_path}')

    # test ROC & PR curves
    test_roc_path = figures_dir / "test_roc_curve.png"
    plot_roc_curve(y_test, y_test_pred_proba, test_roc_path, "Test Set")
    test_pr_path = figures_dir / "test_pr_curve.png"
    plot_pr_curve_and_print(y_test, y_test_pred_proba, test_pr_path, "Test Set")

    logger.info("model evaluation (test only) completed successfully")