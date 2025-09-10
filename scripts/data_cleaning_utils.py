import numpy as np
import pandas as pd

cols_to_drop = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                'CLIENTNUM',
                'Months_on_book',
                'Total_Revolving_Bal',
                'Avg_Open_To_Buy',
                'Total_Trans_Amt',
                'Total_Trans_Ct'
]


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
    df = data.drop(columns=columns, errors="ignore")
    return df


def perform_data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = (data
    .pipe(data_cleaning)
    .pipe(engineer_features)
    .pipe(drop_columns,columns=cols_to_drop)
    )
    # save the data
    return cleaned_data
