# IMPORTS
import sys
import logging
import gdown
import os
import rich
import numpy as np
import pandas as pd
from pathlib import Path
from varclushi import VarClusHi
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tabpfn import TabPFNClassifier
import torch
from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt



version_tag = "dev"

# initialize logger.
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))
log.setLevel(logging.INFO)

log.info("Log initialized.")
##########################################################################################

# PATHS
DATA_DIR = Path("../_data").resolve()
OUTPUT_DIR = Path("../output").resolve()

# LOAD DATA
model_data = pd.read_csv(DATA_DIR / "train.csv")
inference_data = pd.read_csv(DATA_DIR / "test.csv")

###############################################################################################
#DATA CLEANING
# Cap the heavy right tailed 'veh_value' at the 99th percentile for outlier control
veh_value_cap = round(np.nanpercentile(model_data['veh_value'], 99), 3)
print(f"veh_value cap at 99th percentile: {veh_value_cap}")
model_data['veh_value'] = model_data['veh_value'].clip(upper=veh_value_cap)

# Check category distribution of 'veh_body'
rich.print( model_data['veh_body'].value_counts() )
# Group 'MCARA', 'CONVT', 'BUS', and 'RDSTR' 'veh_body' as 'Other'
model_data.loc[model_data['veh_body'].isin(['MCARA','CONVT','BUS','RDSTR']), 'veh_body'] = 'Other'

# Cap the heavy right tailed 'veh_value' at 722
credit_score_cap = 722
model_data['credit_score'] = model_data['credit_score'].clip(upper=credit_score_cap)

# Assume single vehicle policy and create a vehicle count variable
model_data['veh_cnt'] = 1

# Add policy year 
model_data['data_segment'] = "1|model"

# Rename columns for clarity and consistency
model_data = model_data.rename(columns={'numclaims': 'claim_cnt'})
model_data = model_data.rename(columns={'claimcst0': 'claim_amt'})

# Create a new column 'claim_sev' (claim severity) as claim_amt divided by claim_cnt
# If claim_cnt is zero, set claim_sev to NaN to avoid division by zero
model_data['claim_sev'] = model_data.apply(
    lambda row: row['claim_amt'] / row['claim_cnt'] if row['claim_cnt'] != 0 else np.nan,
    axis=1
)

## PREDICTOR LIST
# Create predictor list
veh_pred_lst = ['veh_value', 'veh_body', 'veh_age', 'engine_type', 'max_power', 'veh_color', ]
policy_pred_lst = ['gender', 'agecat', 'e_bill' ]
driving_behavior_pred_lst = ['area', 'time_of_week_driven', 'time_driven']
demo_pred_lst = ['marital_status', 'low_education_ind', 'credit_score', 'driving_history_score']
pred_lst = veh_pred_lst + policy_pred_lst + driving_behavior_pred_lst + demo_pred_lst
cols = ['data_segment', 'exposure'] + pred_lst

# Concatenate model_data[cols] and inference_data[cols] vertically
combined_expo_pred_data = pd.concat(
    [model_data[cols], 
     inference_data[cols]], 
     axis=0, 
     ignore_index=True
     )
print('Combined data shape:', combined_expo_pred_data.shape)

# One-hot encode categorical variables in pred_lst
categorical_cols = [col for col in pred_lst if model_data[col].dtype == 'object' or str(model_data[col].dtype).startswith('category')]
train_data_encoded = pd.get_dummies(model_data, columns=categorical_cols, drop_first=False)

# Update pred_lst to include new dummy variable columns
new_pred_lst = []
for col in pred_lst:
    if col in categorical_cols:
        new_pred_lst.extend([c for c in train_data_encoded.columns if c.startswith(col + '_')])
    else:
        new_pred_lst.append(col)

# Create a new DataFrame with 'var_1', all numerical variables in pred_lst and new_pred_lst
# Identify numerical columns in pred_lst and new_pred_lst
num_pred_cols = [col for col in pred_lst if pd.api.types.is_numeric_dtype(train_data[col])]
selected_cols = new_pred_lst + num_pred_cols
train_data_num = train_data_encoded[selected_cols].copy()

###################################################################################

# ANALYSIS

# Quick Variable Reduction
# VarclusHi

real_numeric = train_data_num.select_dtypes(include=['float64','int64'])
# Remove binary dummy columns -> VarClusHi is for continuos numeric variables
numeric_only = real_numeric.loc[:, ~real_numeric.columns.duplicated()]

vch2 = VarClusHi(numeric_only)
vch2.varclus()
vch2.info
vch2.rsquare

# DATA PREP FOR TABPFN

model_data1 = model_data.copy()

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_data1[col] = le.fit_transform(model_data1[col].astype(str))
    label_encoders[col] = le

target_freq = "claim_cnt"
target_sev = "claim_sev"

num_reps = ['veh_age', 'max_power', 'credit_score']
predictors = num_reps + categorical_cols

# Training / validation split
train = model_data1[model_data1['sample'] == '1|bld'].copy()
test  = model_data1[model_data1['sample'] == '2|val'].copy()

X_train = train[predictors]
X_test  = test[predictors]
##########################################
import huggingface_hub
huggingface_hub.login()
### Put token to access TABPFN
############################################

# Frequency Model

y_train_freq = train[target_freq]

model_freq = TabPFNClassifier(
    device='cpu',
    ignore_pretraining_limits=True
)

model_freq.fit(X_train.values, y_train_freq.values)

pred_freq = model_freq.predict(X_test.values)

print(set(pred_freq))
print(train[target_freq].value_counts())
probs = model_freq.predict_proba(X_test.values)
classes = model_freq.classes_
P0 = probs[:, list(classes).index(0)]
P1 = probs[:, list(classes).index(1)]
P2 = probs[:, list(classes).index(2)]
expected_count = 0*P0 + 1*P1 + 2*P2

# Severity Model
# Drop severity NaN
sev_train = train.dropna(subset=[target_sev]).copy()

X_train_sev = sev_train[X_train.columns]
y_train_sev = sev_train[target_sev]

model_sev = TabPFNRegressor(
    device='cpu', 
    ignore_pretraining_limits=True
)

model_sev.fit(X_train_sev.values, y_train_sev.values)


# Predict severity for test set
pred_severity = model_sev.predict(X_test.values)

##################################################################

# Combined Model

# Pure premium = expected count * expected severity * exposure
pure_premium = expected_count * pred_severity * test['exposure'].values

# Add results to test DataFrame
test['expected_count'] = expected_count
test['expected_severity'] = pred_severity
test['pure_premium'] = pure_premium

# Preview first few rows
print(test[['expected_count','expected_severity','pure_premium']].head())

# MODEL EVALUATION:

y_true = test[target_freq].values
y_pred = model_freq.predict(X_test.values)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

sev_test = test[test[target_freq] > 0]
X_test_sev = sev_test[X_train.columns]
y_true_sev = sev_test[target_sev]

y_pred_sev = model_sev.predict(X_test_sev.values)

from sklearn.metrics import mean_squared_error, mean_absolute_error
print("Severity MSE:", mean_squared_error(y_true_sev, y_pred_sev))
print("Severity MAE:", mean_absolute_error(y_true_sev, y_pred_sev))

plt.scatter(y_true_sev, y_pred_sev, alpha=0.5)
plt.xlabel("Actual severity")
plt.ylabel("Predicted severity")
plt.title("Predicted vs Actual Severity")
plt.show()

actual_loss = test['claim_amt']  # total loss per policy
predicted_loss = test['pure_premium']

print("Pure premium MSE:", mean_squared_error(actual_loss, predicted_loss))
print("Pure premium MAE:", mean_absolute_error(actual_loss, predicted_loss))

plt.scatter(actual_loss, predicted_loss, alpha=0.3)
plt.xlabel("Actual total loss")
plt.ylabel("Predicted pure premium")
plt.title("Predicted vs Actual Loss")
plt.show()

