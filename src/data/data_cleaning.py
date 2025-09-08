import logging
import numpy as np
import pandas as pd
from pathlib import Path

# create a logger
logger = logging.getLogger('data_cleaning')
logger.setLevel(logging.INFO)

# create a handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a formatter
formatter = logging.Formatter(fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

cols_to_drop = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                'CLIENTNUM',
                'Months_on_book',
                'Total_Revolving_Bal',
                'Avg_Open_To_Buy',
                'Total_Trans_Amt',
                'Total_Trans_Ct'
]


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error('The file to load does not exist')


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
      return (
      data
      .assign(
          Education_Level = data['Education_Level'].replace({'Unknown':np.nan,'Post-Graduate':'Advanced_Degree','Doctorate':'Advanced_Degree'}),
          Marital_Status = data['Marital_Status'].replace({'Unknown':np.nan}),
          Income_Category = data['Income_Category'].replace({'Unknown':np.nan})
        )
    )

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .assign(
            Avg_Transaction_Value = np.where(
                data['Total_Trans_Ct'] == 0, 
                np.nan,
                data['Total_Trans_Amt'] / data['Total_Trans_Ct']
            )
        )
    )


def drop_columns(data: pd.DataFrame, columns:list) -> pd.DataFrame:
    df = data.drop(columns=columns)
    return df


def perform_data_cleaning(data: pd.DataFrame, save_data_path: Path) -> None:
    cleaned_data = (data
    .pipe(data_cleaning)
    .pipe(engineer_features)
    .pipe(drop_columns,columns=cols_to_drop)
    )
    # save the data
    cleaned_data.to_csv(save_data_path,index=False)

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # clean data save direcotry
    cleaned_data_save_dir = root_path / "data" / "cleaned"
    # make directory if not exists
    cleaned_data_save_dir.mkdir(exist_ok=True,parents=True)
    # clean data file name
    cleaned_data_filename = 'bankchurners_cleaned.csv'
    # data save path
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename

    # data load path
    data_load_path = root_path / "data" / "raw" / "BankChurners.csv"

    # load the data
    df = load_data(data_load_path)
    logger.info('Data read successfully')

    # clean data and save
    perform_data_cleaning(data=df, save_data_path=cleaned_data_save_path)
    logger.info('Data cleaned and saved')




