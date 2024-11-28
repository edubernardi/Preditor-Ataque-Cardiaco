from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, row_number
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler, StandardScaler, RobustScaler

from pyspark.sql.functions import col, rand, floor
from pyspark.sql import Window
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier

from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import kagglehub
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType

trained_models = {}
scalers = []

def train():
    # Download latest version
    path = kagglehub.dataset_download("rashikrahmanpritom/heart-attack-analysis-prediction-dataset")

    # cria um schema para ler o dataset
    schema = StructType([
        StructField("age", FloatType(), nullable=True),
        StructField("sex", FloatType(), nullable=True),
        StructField("cp", FloatType(), nullable=True),
        StructField("trtbps", FloatType(), nullable=True),
        StructField("chol", FloatType(), nullable=True),
        StructField("fbs", FloatType(), nullable=True),
        StructField("restecg", FloatType(), nullable=True),
        StructField("thalachh", FloatType(), nullable=True),
        StructField("exng", FloatType(), nullable=True),
        StructField("oldpeak", FloatType(), nullable=True),
        StructField("slp", FloatType(), nullable=True),
        StructField("caa", FloatType(), nullable=True),
        StructField("thall", FloatType(), nullable=True),
        StructField("output", FloatType(), nullable=True)
    ])

    import findspark
    findspark.init()

    spark = SparkSession.builder.appName('PySpark-Get-Started').getOrCreate()
    spark.conf.set('spark.sql.repl.eagerEval.enabled', True)

    data = spark.read.csv(path + "/heart.csv", header=True, schema=schema)

    input_columns = [col for col in data.columns if col != "output"]

    assembler = VectorAssembler(inputCols=input_columns, outputCol="features")
    assembled_df = assembler.transform(data)
    
    standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
    scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)    
    scalers.append(standardScaler)

    models = {
        "RandomForestClassifier": RandomForestClassifier(featuresCol='features_scaled', labelCol='output'),
        "LogisiticRegression": LogisticRegression(featuresCol='features_scaled', labelCol='output'),
        "DecisionTreeClassifier": DecisionTreeClassifier(featuresCol='features_scaled', labelCol='output')
    }

    stats = []
    k = 5
    for model_name, model in models.items():
        scaled_df = scaled_df.withColumn("rand", rand())
        window = Window.partitionBy("output").orderBy("rand")
        stratified_df = scaled_df.withColumn("fold", floor((row_number().over(window) - 1) % k))

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        trained_model = None

        for fold in range(k):
            train_data = stratified_df.filter(col("fold") != fold)
            test_data = stratified_df.filter(col("fold") == fold)

            trained_model = model.fit(train_data)

            predictions = trained_model.transform(test_data)

            classifications = predictions.select("prediction", "output")
            metrics = MulticlassMetrics(classifications.rdd.map(tuple))

            accuracies.append(metrics.accuracy)
            precisions.append(metrics.precision(1.0))
            recalls.append(metrics.recall(1.0))
            f1_scores.append(metrics.fMeasure(1.0))

        trained_models[model_name] = trained_model

        model_stats = {}
        model_stats["Name"] = model_name
        model_stats["Average Accuracy"] = sum(accuracies) / k
        model_stats["Average Precision"] = sum(precisions) / k
        model_stats["Average Recall"] = sum(recalls) / k
        model_stats["Average F1 Measure"] = sum(f1_scores) / k
        
        stats.append(model_stats)
    return stats

def predict(patient):
    if trained_models is None:
        return None
    
    spark = SparkSession.builder.appName('PySpark-Get-Started').getOrCreate()

    columns = list(patient.keys())
    values = list(patient.values())
    
    patient_df = spark.createDataFrame([values], columns)
    
    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    assembled_df = assembler.transform(patient_df)
    
    scaled_df = scalers[0].fit(assembled_df).transform(assembled_df)    
    predictions_dict = {}

    for model_name, model in trained_models.items():
        predictions = model.transform(scaled_df)   
        prediction_value = predictions.select("prediction").collect()[0][0]
        
        predictions_dict[model_name] = prediction_value
    
    return predictions_dict
    
