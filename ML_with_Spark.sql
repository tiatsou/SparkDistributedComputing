-- Databricks notebook source
-- MAGIC 
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px; height: 163px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC This is the module 4 final assignment for [Distributed Computing with Spark SQL](https://www.coursera.org/learn/spark-sql). The editor is Der-Hsuan Tsou.  
-- MAGIC # Logistic Regression Classifier
-- MAGIC 
-- MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this notebook:
-- MAGIC * Preprocess data for use in a machine learning model
-- MAGIC * Step through creating a sklearn logistic regression model for classification
-- MAGIC * Predict the `Call_Type_Group` for incidents in a SQL table  
-- MAGIC 
-- MAGIC By the end of this notebook, we would like to train a logistic regression model to predict 2 of the most common `Call_Type_Group` given information from the rest of the table.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Environment Setup and Load Data

-- COMMAND ----------

-- MAGIC %run ../Includes/Classroom-Setup

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check the number of partitions and the size of each parquet by `%fs ls data_path`  
-- MAGIC --> There are 8 partitions and each parquet size is about 6.3 MB

-- COMMAND ----------

-- MAGIC %fs ls /mnt/davis/fire-calls/fire-calls-clean.parquet

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Load the `/mnt/davis/fire-calls/fire-calls-clean.parquet` data as `fireCallsClean` table.

-- COMMAND ----------

USE DATABRICKS;

CREATE TABLE IF NOT EXISTS fireCallsClean
USING parquet
OPTIONS (
path "/mnt/davis/fire-calls/fire-calls-clean.parquet"
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check that the data is loaded in properly.

-- COMMAND ----------

SELECT * FROM fireCallsClean LIMIT 10

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Data  Preparetion

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check what the different `Call_Type_Group` values are and their respective counts.

-- COMMAND ----------

SELECT `Call_Type_Group`, COUNT(*) AS cnt
FROM fireCallsClean
GROUP BY `Call_Type_Group`
ORDER BY cnt DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Drop all the rows where `Call_Type_Group = null`. Since we don't have a lot of `Call_Type_Group` with the value `Alarm` and `Fire`, also drop these calls from the table. Call this new view `fireCallsGroupCleaned`.

-- COMMAND ----------

CREATE OR REPLACE VIEW fireCallsGroupCleaned AS 
(
  SELECT *
  FROM fireCallsClean
  WHERE `Call_Type_Group` IN ('Potentially Life-Threatening', 'Non Life-threatening')
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check that every entry in `fireCallsGroupCleaned`  has a `Call_Type_Group` of either `Potentially Life-Threatening` or `Non Life-threatening`.

-- COMMAND ----------

SELECT `Call_Type_Group`, COUNT(*) AS cnt
FROM fireCallsGroupCleaned
GROUP BY `Call_Type_Group`;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check the number of rows in `fireCallsGroupCleaned`.  
-- MAGIC --> 134198 rows

-- COMMAND ----------

SELECT COUNT(*) 
FROM fireCallsGroupCleaned;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC To make the call type prediction, the following variables will be used from `fireCallsGroupCleaned` and create a view called `fireCallsDF` so the table can be accessed in Python:
-- MAGIC 
-- MAGIC * "Call_Type"
-- MAGIC * "Fire_Prevention_District"
-- MAGIC * "Neighborhooods_-\_Analysis_Boundaries" 
-- MAGIC * "Number_of_Alarms"
-- MAGIC * "Original_Priority" 
-- MAGIC * "Unit_Type" 
-- MAGIC * "Battalion"
-- MAGIC * "Call_Type_Group"

-- COMMAND ----------

CREATE OR REPLACE VIEW fireCallsDF AS 
(
  SELECT Call_Type, Fire_Prevention_District, `Neighborhooods_-_Analysis_Boundaries`, Number_of_Alarms, Original_Priority, Unit_Type, Battalion, Call_Type_Group
  FROM fireCallsGroupCleaned
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Convert to Pandas DataFrame  
-- MAGIC Load the `fireCallsDF` table just created into python.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = sql("SELECT * FROM fireCallsDF")
-- MAGIC pdDF = df.toPandas()
-- MAGIC display(pdDF)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Train-Test Split
-- MAGIC 
-- MAGIC First, convert the Spark DataFrame to pandas so we can use sklearn to preprocess the data into numbers.  
-- MAGIC By doing so, it will be compatible with the logistic regression algorithm with a [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). 
-- MAGIC 
-- MAGIC Then perform a train test split on the pandas DataFrame. The target variable is `Call_Type_Group`.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from sklearn.model_selection import train_test_split
-- MAGIC from sklearn.preprocessing import LabelEncoder
-- MAGIC 
-- MAGIC le = LabelEncoder()
-- MAGIC numerical_pdDF = pdDF.apply(le.fit_transform)
-- MAGIC 
-- MAGIC X = numerical_pdDF.drop("Call_Type_Group", axis=1)
-- MAGIC y = numerical_pdDF["Call_Type_Group"].values
-- MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check the encoded target labels with the corresponding `Call_Type_group`.  
-- MAGIC --> 0: Non Life-threatening; 1: Potentially Life-Threatening

-- COMMAND ----------

-- MAGIC %python
-- MAGIC y[0:8]

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check the training data `X_train` & `y_train` which should only have numerical values now.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(X_train)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC y_train

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Building a Logistic Regression Model in Sklearn
-- MAGIC 
-- MAGIC Create a pipeline with 2 steps. 
-- MAGIC 
-- MAGIC 0. [One Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder): Converts the  features into vectorized features by creating a dummy column for each value in that category. 
-- MAGIC 
-- MAGIC 0. [Logistic Regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): Although the name includes "regression", it is used for classification by predicting the probability that the `Call Type Group` is one label and not the other.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from sklearn.linear_model import LogisticRegression
-- MAGIC from sklearn.preprocessing import OneHotEncoder
-- MAGIC from sklearn.pipeline import Pipeline
-- MAGIC 
-- MAGIC ohe = ("ohe", OneHotEncoder(handle_unknown="ignore"))
-- MAGIC lr = ("lr", LogisticRegression())
-- MAGIC 
-- MAGIC pipeline = Pipeline(steps = [ohe, lr]).fit(X_train, y_train)
-- MAGIC y_pred = pipeline.predict(X_test)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Evaluate the Model Performance
-- MAGIC Use the accurancy to evaluate the model on test data.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from sklearn.metrics import accuracy_score
-- MAGIC print(f"Accuracy of model: {accuracy_score(y_pred, y_test)}")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Save Model  
-- MAGIC Save pipeline (with both stages) to disk.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import mlflow
-- MAGIC from mlflow.sklearn import save_model
-- MAGIC 
-- MAGIC model_path = "/dbfs/" + username + "/Call_Type_Group_lr"
-- MAGIC dbutils.fs.rm(username + "/Call_Type_Group_lr", recurse=True)
-- MAGIC save_model(pipeline, model_path)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## UDF  
-- MAGIC Now a machine learning pipeline is created and well-trained, [MLflow](https://mlflow.org/) will be used to register the `.predict` function of the sklearn pipeline as a UDF which can be applied in parallel later. The function will be referred as`predictUDF` in SQL.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import mlflow
-- MAGIC from mlflow.pyfunc import spark_udf
-- MAGIC 
-- MAGIC predict = spark_udf(spark, model_path, result_type="int")
-- MAGIC spark.udf.register("predictUDF", predict)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Create a view called `testTable` of the test data `X_test` so that we can see this table in SQL.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark_df = spark.createDataFrame(X_test)
-- MAGIC spark_df.createOrReplaceTempView("testTable")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Save Predictions  
-- MAGIC Create a table called `predictions` using the `predictUDF` function that registered beforehand. Apply the `predictUDF` to every row of `testTable` in parallel so that each row of `testTable` has a `Call_Type_Group` prediction.

-- COMMAND ----------

USE DATABRICKS;
DROP TABLE IF EXISTS predictions;

CREATE TEMPORARY VIEW predictions AS (
  SELECT CAST(predictUDF(Call_Type, Fire_Prevention_District, `Neighborhooods_-_Analysis_Boundaries`, Number_of_Alarms, Original_Priority, Unit_Type, Battalion) AS double) AS Call_Type_Prediction, *
  FROM testTable
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check the table and see what my model predicted for each call entry!

-- COMMAND ----------

SELECT * FROM predictions LIMIT 10

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Check the unique values for `Call_Type_Prediction`.  
-- MAGIC --> 0: Non Life-threatening; 1: Potentially Life-Threatening

-- COMMAND ----------

SELECT DISTINCT(Call_Type_Prediction)
FROM predictions;

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
