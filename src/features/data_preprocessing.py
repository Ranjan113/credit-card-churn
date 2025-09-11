import yaml
import joblib
import logging
import pandas as pd
from pathlib import Path
from sklearn import set_config
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder


TARGET_COLUMN = 'Attrition_Flag'

CATEGORICAL_NOMINAL = ['Gender', 'Marital_Status', 'Income_Category', 'Card_Category']
CATEGORICAL_ORDINAL = ['Education_Level']
NUMERICAL_CONTINUOUS = ['Customer_Age', 'Dependent_count', 'Total_Relationship_Count', 
                       'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                       'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio','Avg_Transaction_Value']
NUMERICAL_IMPUTE_MEDIAN = ['Avg_Transaction_Value']
CATEGORICAL_IMPUTE_MODE = ['Education_Level', 'Marital_Status', 'Income_Category']


# set the transformer outptus to pandas
set_config(transform_output='pandas')

# create a logger
logger = logging.getLogger('data_preprocessing')
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
        logger.error('The file to load does not exist')
        raise


def make_X_and_y(data: pd.DataFrame,target_column: str):
    X = data.drop(columns = target_column)
    y = data[target_column]
    return X,y


def join_X_and_y_concat(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return pd.concat([X, y], axis=1, join='inner')


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)


def save_transformer(transformer, save_dir: Path, transformer_name: str):
    # form the save location
    save_location = save_dir / transformer_name
    # save the transformer
    joblib.dump(value=transformer,filename=save_location)


def train_preprocessor(preprocessor, data:pd.DataFrame):
    # fit on the data
    preprocessor.fit(data)
    return preprocessor


def perform_transformation(preprocessor, data: pd.DataFrame):
    # transform the data
    transformed_data = preprocessor.transform(data)
    return transformed_data


def read_params(file_path: Path):
    try:
        with open(file_path,'r') as f:
            params_file = yaml.safe_load(f)
        random_state = params_file['Data_Preprocessing']['smote_random_state']
        neighbors = params_file['Data_Preprocessing']['smote_k_neighbors']
        return random_state,neighbors
    except FileNotFoundError:
        logger.error('file to load does not exist, switching to default values')
        random_state = 42
        neighbors = 5
        return random_state,neighbors


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int,neighbours: int) -> tuple[pd.DataFrame, pd.Series]:
    """ apply smote to handle class imbalance"""
    try:
        sm = SMOTE(random_state= random_state,k_neighbors=neighbours)
        X_train_res_np, y_train_res_np = sm.fit_resample(X_train,y_train)

        X_train_res = pd.DataFrame(X_train_res_np, columns=X_train.columns)
        y_train_res = pd.Series(y_train_res_np, name=TARGET_COLUMN)

        logger.info(f'Original training set shape: {X_train.shape}')
        logger.info(f'Resampled training set shape: {X_train_res.shape}')
        logger.info(f'Original class distribution: {y_train.value_counts().to_dict()}')
        logger.info(f'Resampled class distribution: {pd.Series(y_train_res).value_counts().to_dict()}')

        return X_train_res, y_train_res
    
    except Exception as e:
        logger.error(f'SMOTE failed: {str(e)}')
        raise



if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data load path
    train_data_path = root_path / "data" / "interim" / "train.csv"
    test_data_path = root_path / "data" / "interim" / "test.csv"
    # save data directory
    save_data_dir = root_path / "data" / "processed"
    # make directory if not exists
    save_data_dir.mkdir(exist_ok=True,parents=True)
    # train and test data save paths
    train_trans_filename = 'train_trans.csv'
    test_trans_filename = 'test_trans.csv'
    save_train_trans_path = save_data_dir / train_trans_filename
    save_test_trans_path = save_data_dir / test_trans_filename
    # params file path
    params_file_path = root_path / "params.yaml"


    simple_imputer = ColumnTransformer(transformers=[
        ('mode_imputing',SimpleImputer(strategy='most_frequent'),CATEGORICAL_IMPUTE_MODE),
        ('median_imputing',SimpleImputer(strategy='median'),NUMERICAL_IMPUTE_MEDIAN)
        ],remainder='passthrough',n_jobs=-1,verbose_feature_names_out=False)
    
    encode = ColumnTransformer(transformers=[
        ('ordinal_encoding',OrdinalEncoder(categories=[['Uneducated','High School','College','Graduate','Advanced_Degree']]),CATEGORICAL_ORDINAL),
        ('nominal_encoding',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),CATEGORICAL_NOMINAL)
        ],remainder='passthrough',n_jobs=-1,verbose_feature_names_out=False)
    
    power_transform = ColumnTransformer(transformers=[
        ('power_transform',PowerTransformer(),NUMERICAL_CONTINUOUS)
        ],remainder='passthrough',n_jobs=-1,verbose_feature_names_out=False)
    
    preprocessor = Pipeline(steps=[
        ("simple_imputer",simple_imputer),
        ('encoding',encode),
        ('pt',power_transform)
        ])


    # load the train and test data
    train_df = load_data(train_data_path)
    logger.info('Train data loaded successfully')
    test_df = load_data(test_data_path)
    logger.info('Test data loaded successfully')


    # split the train and test data
    X_train, y_train = make_X_and_y(data=train_df, target_column=TARGET_COLUMN)
    X_test, y_test = make_X_and_y(data=test_df, target_column=TARGET_COLUMN)
    logger.info('Data splitting completed')


    # encode the target column
    attrition_mapping = {'Existing Customer': 0, 'Attrited Customer': 1}
    y_train_enc = y_train.map(attrition_mapping)
    logger.info('encoded train data target columns')
    y_test_enc = y_test.map(attrition_mapping)
    logger.info('encoded test data target columns')


    # fit the preprocessor on X_train
    train_preprocessor(preprocessor=preprocessor, data=X_train)
    logger.info('Preprocessor is trained')

    # ensure pipeline outputs pandas DataFrame
    preprocessor.set_output(transform="pandas")

    # transform the data
    X_train_trans = perform_transformation(preprocessor=preprocessor, data=X_train)
    logger.info('Train data is transformed')
    X_test_trans = perform_transformation(preprocessor= preprocessor, data=X_test)
    logger.info('Test data is transformed')


    # read params
    random_state,neighbors = read_params(params_file_path)
    logger.info('parameters read successfully')
    logger.info(f'Using random_state: {random_state} and k_neighbors: {neighbors}')


    # apply smote
    X_train_res, y_train_res = apply_smote(X_train_trans,y_train_enc,random_state,neighbors)
    logger.info('smote applied successfully')


    # join back X and y
    train_trans_df = join_X_and_y_concat(X_train_res,y_train_res)
    test_trans_df = join_X_and_y_concat(X_test_trans,y_test_enc)
    logger.info("Datasets joined")


    # save the transformed data
    save_data(train_trans_df,save_train_trans_path)
    logger.info('train data saved to location')

    save_data(test_trans_df,save_test_trans_path)
    logger.info('test data saved to location')


    # save preprocessor to location
    transformer_filename = "preprocessor.joblib"
    transformer_save_dir = root_path / "models"
    transformer_save_dir.mkdir(exist_ok=True)

    save_transformer(transformer=preprocessor, save_dir=transformer_save_dir, transformer_name=transformer_filename)
    logger.info("Preprocessor saved to location")

