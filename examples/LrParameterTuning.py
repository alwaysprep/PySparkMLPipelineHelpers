import sys
sys.path.extend(['../transformers', '../evaluators'])

from pprint import pprint

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext, StorageLevel

from FeatureHashing import FeatureHashing
from LogLossEvaluator import BinaryLogLossEvaluator

con = SparkConf().setAppName('LogisticRegression')

sc = SparkContext(conf=con)
sqlContext = SQLContext(sc)

dataFile = "../data/criteo.txt"
train, test = filledRDD = sc.textFile(dataFile, 4)\
                .map(lambda x: x.split("\t"))\
                .map(lambda row:
                        (
                            float(row[0]),
                            [float(num) if num else 0.0 for num in row[1:14]],
                            [str(cat) if cat else "" for cat in row[14:]]
                        )
                     )\
                .randomSplit([66, 33])

print("#######"*10 + "NUM OF PARTITION" + "#######"*10 )
print(train.getNumPartitions())
print("#######"*22)

training = train.toDF(["label", "numerical", "categorical"])
testing = test.toDF(["label", "numerical", "categorical"])

training.persist(StorageLevel.MEMORY_AND_DISK_SER)
testing.persist(StorageLevel.MEMORY_AND_DISK_SER)

feature_hasher = FeatureHashing(numericalCol="numerical",
                                categoricalCol="categorical",
                                outputCol="features")
lr = LogisticRegression(maxIter=5, regParam=1, elasticNetParam=0)
pipeline = Pipeline(stages=[feature_hasher, lr])

paramGrid = ParamGridBuilder()\
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryLogLossEvaluator(),
    numFolds=5
)

cvModel = crossval.fit(training)

prediction = cvModel.bestModel.transform(testing)

evaluator = BinaryLogLossEvaluator()
logloss = evaluator.evaluate(prediction)

pprint(logloss)
pprint(cvModel.bestModel)
pprint(cvModel.avgMetrics)
pprint(cvModel.getEstimatorParamMaps())