import pandas as pd
import streamlit as st
import warnings
def validate_obj(column):
    if column.isna().all():
        return 'All rows are null'
    unique_values = column.nunique()
    total_values = len(column)
    threshold = 0.5
    return "categorical" if  unique_values< (total_values*threshold) else "textual"

def sample(column,dt):
    unique_values = column.nunique()
    total_values = len(column)
    threshold = 0.5
    if((dt != 'categorical') and (dt !='textual')):
        return column[:6].tolist()
    if(dt == 'categorical' and (unique_values< (total_values*threshold)) and (unique_values<20 )):
        return list(column.unique())
    else:
        return column[:6].tolist()


def extract_data(data: pd.DataFrame) -> dict:
    num_rows, num_columns = data.shape

    column_data_types = {
        column: validate_obj(data[column]) if data[column].dtype == 'O' else data[column].dtype
        for column in data.columns
    }

    mean = {
        column:round(data[column].mean(),2)  if data[column].dtype in ['int64','float64',int,float] else None
        for column in data.columns
    }

    no_null = {
        column: data[column].isnull().sum() for column in data.columns
    }

    sample_elements = {
        column:sample(data[column],column_data_types[column]) for column in data.columns
    }

    rules = {
        'num_rows': num_rows,
        'num_columns': num_columns,
        'column_names_data_types': column_data_types,
        'mean' : mean,
        'num_of_null' : no_null,
        'sample_elements':sample_elements
        
    }

    return rules

def check_type(dtype: str, value):
    if "float" in str(dtype):
        return round(float(value), 2)
    elif "int" in str(dtype):
        return int(value)
    else:
        return value

def kurtosis(df):
    kurt_value = df.kurt()
    if(kurt_value > 3):
        return 'Leptokurtic'
    elif(kurt_value < 3):
        return 'Platykurtic'
    else:
        return 'Mesokurtic'

def skewness(df):
    skew_value = df.skew()
    if -0.5 < skew_value < 0.5:
        return 'fairly symmetrical'
    if -1 < skew_value < -0.5 or 0.5 < skew_value < 1:
        return 'Moderately Skewed'
    elif skew_value <= -1 or skew_value >= 1:
        return 'Highly Skewed'
    else:
        return 'Approximately Symmetric'

def get_column_properties(df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
  """Get properties of each column in a pandas DataFrame"""
  properties_list = []
  for column in df.columns:
      dtype = df[column].dtype
      properties = {}
      if dtype in [int, float, complex]:
          properties["dtype"] = "number"
          properties["std"] = check_type(dtype, df[column].std())
          properties["mean"] = check_type(dtype, df[column].mean())
          #properties["var"] = check_type(dtype, df[column].var())
          properties["min"] = check_type(dtype, df[column].min())
          properties["max"] = check_type(dtype, df[column].max())
          properties["skewness"] = skewness(df[column])
          #properties["kurtosis"] = kurtosis(df[column])

      elif dtype == bool:
          properties["dtype"] = "boolean"
      elif dtype == object:
          # Check if the string column can be cast to a valid datetime
          try:
              with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  pd.to_datetime(df[column], errors='raise')
                  properties["dtype"] = "date"
          except ValueError:
              # Check if the string column has a limited number of values
              if df[column].nunique() / len(df[column]) < 0.5:
                  properties["dtype"] = "category"
              else:
                  properties["dtype"] = "string"
      elif pd.api.types.is_categorical_dtype(df[column]):
          properties["dtype"] = "category"
      elif pd.api.types.is_datetime64_any_dtype(df[column]):
          properties["dtype"] = "date"
      else:
          properties["dtype"] = str(dtype)

      # add min max if dtype is date
      if properties["dtype"] == "date":
          try:
              properties["min"] = df[column].min()
              properties["max"] = df[column].max()
          except TypeError:
              cast_date_col = pd.to_datetime(df[column], errors='coerce')
              properties["min"] = cast_date_col.min()
              properties["max"] = cast_date_col.max()
      # Add additional properties to the output dictionary
      nunique = df[column].nunique()
      if "samples" not in properties:
          non_null_values = df[column][df[column].notnull()].unique()
          n_samples = min(n_samples, len(non_null_values))
          samples = pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
          if(nunique < 30):
              properties["Unique_elements"] = df[column].unique()
          else:
            properties["samples"] = samples
      properties["num_unique_values"] = nunique
      properties["num_of_nulls"] = df[column].isnull().sum()
      properties["semantic_type"] = ""
      properties["description"] = ""
      properties["type_of_measurement"]=""
      properties_list.append({"column": column, "properties": properties})
  
  return properties_list
