import sys
sys.path.append('../transformers')

from pprint import pprint

from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark import StorageLevel

from FeatureHashing import FeatureHashing

con = SparkConf()
sc = SparkContext(conf=con)
sqlContext = SQLContext(sc)

dataFile = "../data/criteo.txt"
train, test = filledRDD = sc.textFile(dataFile)\
                .map(lambda x: x.split("\t"))\
                .map(lambda row:
                        (
                            float(row[0]),
                            [float(num) if num else 0.0 for num in row[1:14]],
                            [str(cat) if cat else "" for cat in row[14:]]
                        )
                     )\
                .sample(False, 0.1)\
                .randomSplit([9, 1])


print("#######"*10 + "NUM OF PARTITION" + "#######"*10 )
print(train.getNumPartitions())
print("#######"*22)

training = train.toDF(["label", "numerical", "categorical"])
testing = test.toDF(["label", "numerical", "categorical"])

training.persist(StorageLevel.MEMORY_AND_DISK_SER)
testing.persist(StorageLevel.MEMORY_AND_DISK_SER)

feature_hasher = FeatureHashing(numericalCol="numerical",
                                categoricalCol="categorical",
                                outputCol="features", numFeatures=1024)

mlp = MultilayerPerceptronClassifier(layers=[1024, 512, 2])
pipeline = Pipeline(stages=[feature_hasher, mlp])

paramGrid = ParamGridBuilder() \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=5
)

cvModel = crossval.fit(training)

prediction = cvModel.bestModel.transform(testing)

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
accuracy = evaluator.evaluate(prediction)

pprint(accuracy)
pprint(cvModel.avgMetrics)
pprint(cvModel.getEstimatorParamMaps())