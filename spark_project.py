from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("ML-Logistic").getOrCreate()

data = [
    (1.2, 3.4, 0),
    (2.3, 1.5, 1),
    (0.5, 2.2, 0),
    (3.5, 4.5, 1),
    (1.5, 0.7, 0),
    (4.2, 3.9, 1)
]

columns = ["feature1", "feature2", "label"]
df = spark.createDataFrame(data, columns)

assembler = VectorAssembler(
    inputCols=["feature1", "feature2"],
    outputCol="features"
)

train_df, test_df = df.randomSplit([0.7, 0.3], seed=1)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=10
)

pipeline = Pipeline(stages=[assembler, lr])

model = pipeline.fit(train_df)

predictions = model.transform(test_df)

predictions.select(
    "feature1", "feature2", "label",
    "prediction", "probability"
).show()

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print("AUC:", auc)
