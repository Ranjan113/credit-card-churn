import logging
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# create a logger
logger = logging.getLogger('data_preparation')
logger.setLevel(logging.INFO)

# create a handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error('The file to load does not exist')
        raise


def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data,test_size = test_size, random_state = random_state)

    return train_data, test_data


def read_params(file_path: Path):
    try:
        with open(file_path,'r') as f:
            params_file = yaml.safe_load(f)
        test_size = params_file['Data_Preparation']['test_size']
        random_state = params_file['Data_Preparation']['random_state']
        return test_size,random_state
    except FileNotFoundError:
        logger.error('file to load does not exist, switching to default values')
        default_dict = {'test_size' : 0.2, 'random_state': 42}
        # read the default dict
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']
        return test_size,random_state


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path,index=False)


if __name__ == '__main__':
    # root path
    root_path = Path(__file__).parent.parent.parent
    print(root_path)
    # data load path
    data_load_path = root_path / "data" / "cleaned" / "bankchurners_cleaned.csv"
    # save data directory
    save_data_dir = root_path / "data" / "interim"
    # make directory if not present
    save_data_dir.mkdir(exist_ok=True,parents=True)
    # train and test data save paths
    train_filename = 'train.csv'
    test_filename = 'test.csv'
    save_train_data_path = save_data_dir / train_filename
    save_test_data_path = save_data_dir / test_filename
    # read params file path
    params_file_path = root_path / "params.yaml"

    # load the cleaned data
    df = load_data(data_load_path)
    logger.info('Data read successfully')

    # read params
    test_size, random_state = read_params(params_file_path)
    logger.info('parameters read successfully')
    logger.info(f'Using test_size: {test_size}, random_state: {random_state}')

    # split data
    train_data, test_data = split_data(df,test_size = test_size,random_state = random_state)
    logger.info('succesfully split the data into train and test')

    # save train data
    save_data(train_data,save_train_data_path)
    logger.info(f'train data saved to location')

    # save test data
    save_data(test_data,save_test_data_path)
    logger.info(f'test data saved to location')

