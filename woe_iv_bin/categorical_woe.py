import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def woe_discrete(df, cat_variable_name, y_df):
    """
    Assumes the categorical variable is already in a binned/categorized form.
    Calculates Weight of Evidence (WoE) and Information Value (IV) for a categorical variable.
    """
    # Combine the feature and target variable for easier manipulation
    data = pd.concat([df[[cat_variable_name]], y_df], axis=1)
    data.columns = ['feature', 'target']

    # Calculate the necessary statistics
    grouped = data.groupby('feature')['target'].agg(['count', 'mean'])
    grouped.columns = ['n_obs', 'prop_good']
    grouped['prop_n_obs'] = grouped['n_obs'] / grouped['n_obs'].sum()
    grouped['n_good'] = grouped['prop_good'] * grouped['n_obs']
    grouped['n_bad'] = grouped['n_obs'] - grouped['n_good']
    grouped['prop_n_good'] = grouped['n_good'] / grouped['n_good'].sum()
    grouped['prop_n_bad'] = grouped['n_bad'] / grouped['n_bad'].sum()

    # Calculate WoE
    grouped['WoE'] = np.log((grouped['prop_n_good'] + 0.0001) / (grouped['prop_n_bad'] + 0.0001))
    grouped['IV'] = (grouped['prop_n_good'] - grouped['prop_n_bad']) * grouped['WoE']
    total_IV = grouped['IV'].sum()

    # Return the WoE and IV values
    grouped['total_IV'] = total_IV
    return grouped[['WoE', 'IV', 'total_IV']]