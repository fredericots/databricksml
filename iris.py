# Databricks notebook source
configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope='learningscope', key='learningdatalakeclientid'),
           "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope='learningscope', key='learningdatalakesecret'),
           "fs.azure.account.oauth2.client.endpoint": \
               "https://login.microsoftonline.com/" + \
               dbutils.secrets.get(scope='learningscope', key='learningdatalakedirid') + \
               "/oauth2/token"}

dbutils.fs.mount(
  source = "abfss://data@learning12345.dfs.core.windows.net/",
  mount_point = "/mnt/datalearning",
  extra_configs = configs)


# COMMAND ----------

# dbutils.fs.unmount("/mnt/datalearning")

# COMMAND ----------



# COMMAND ----------

import pandas as pd

DATALAKE_PATH = '/dbfs/mnt/datalearning/'

# COMMAND ----------

train = pd.read_csv(DATALAKE_PATH + 'iris-data-treino.csv', sep=';')

# COMMAND ----------

train.isnull().sum()

# COMMAND ----------

train = train.dropna().reset_index(drop=True)

# COMMAND ----------

train

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import sklearn

# COMMAND ----------

mlflow.autolog()

# COMMAND ----------

train_y = train.loc[:,'classe'].copy()
train_x = train.drop('classe',axis=1)

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
    model = sklearn.tree.DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth = 3)
    model.fit(train_x, train_y)
    
    predicted = model.predict(train_x)
    acc = sklearn.metrics.accuracy_score(train_y, predicted)
    # The AUC score on test data is not automatically logged, so log it manually
    mlflow.log_metric("test_acc", acc)
    print("Test AUC of: {}".format(acc))

# COMMAND ----------



# COMMAND ----------

logged_model = 'runs:/735e21c1d86348e9b3ef9175fb965a19/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

test = pd.read_csv(DATALAKE_PATH + 'iris-data-teste.csv', sep=';')
test_y = test.loc[:,'classe'].copy()
test_x = test.drop('classe',axis=1)

# COMMAND ----------

list(loaded_model.predict(test_x))

# COMMAND ----------

test_y

# COMMAND ----------



# COMMAND ----------


