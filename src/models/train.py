import pandas as pd
import yaml
import logging
from pathlib import Path
import joblib
from xgboost import XGBClassifier


TARGET_COLUMN = 'Attrition_Flag'

# create a logger
logger = logging.getLogger('model_training')
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
        logger.error('file to load does not exist')
        raise

def read_params(file_path: Path):
    with open(file_path,'r') as f:
        params_file = yaml.safe_load(f)
    return params_file

def save_model(model, save_dir:Path, model_name: str):
    # form the save location
    save_location = save_dir / model_name
    # save the model
    joblib.dump(model,save_location)

def train_model(model, X_train,y_train):
    # fit the model
    model.fit(X_train,y_train)
    return model

def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data load path
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    # params file
    params_file_path = root_path / "params.yaml"

    # load the training data
    training_data = load_data(data_path)
    logger.info('Training data loaded successfully')

    # split the data into X and y
    X_train, y_train = make_X_and_y(training_data,TARGET_COLUMN)
    logger.info('Data splitting completed')

    # model parameters
    model_params = read_params(params_file_path)['Train']

    # xgboost params
    xgb_params = model_params['XG_boost']
    logger.info('xg boost params read successfully')

    # build the xg boost model
    xgb = XGBClassifier(**xgb_params)
    logger.info('built xg boost model')

    # fit the model
    xgb_model = train_model(xgb,X_train,y_train)
    logger.info('model training completed')

    # model name
    model_filename = 'model.joblib'
    # directory to save model
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True,parents=True)

    # save the model
    save_model(xgb_model,model_save_dir,model_filename)
    logger.info("trained model saved to location")

