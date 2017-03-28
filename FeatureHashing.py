from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark import keyword_only
from collections import defaultdict, OrderedDict
from pyspark.sql.functions import udf
from pyspark.ml.linalg import SparseVector, VectorUDT


class FeatureHashing(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, numericalCol=None, categoricalCol=None, outputCol=None, numFeatures=2**20):
        super(FeatureHashing, self).__init__()
        self.numericalCol = Param(self, "numericalCol", "")
        self.categoricalCol = Param(self, "categoricalCol", "")
        self.numFeatures = Param(self, "numFeatures", "")
        self._setDefault(numericalCol=numericalCol)
        self._setDefault(categoricalCol=categoricalCol)
        self._setDefault(numFeatures=numFeatures)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, numericalCol=None, categoricalCol=None, outputCol=None, numFeatures=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def getNumericalCol(self):
        return self.getOrDefault(self.numericalCol)

    def getCategoricalCol(self):
        return self.getOrDefault(self.categoricalCol)

    def getNumFeatures(self):
        return self.getOrDefault(self.numFeatures)

    def setNumFeatures(self, value):
        self._paramMap[self.numFeatures] = value
        return self

    def _transform(self, dataset):
        numFeatures = self.getNumFeatures()

        def f(numerical, categorical):
            dic = defaultdict(float)
            numCatFeatures = (numFeatures - len(numerical))
            for categoricalFeature in categorical:
                index = hash(categoricalFeature) % numCatFeatures
                dic[index] += 1.0
            for index, numericalFeature in enumerate(numerical):
                dic[numCatFeatures + index] = numericalFeature
            sorted_dic = OrderedDict(sorted(dic.items()))
            return SparseVector(numFeatures, sorted_dic.keys(), sorted_dic.values())

        t = VectorUDT()

        out_col = self.getOutputCol()
        num_col = dataset[self.getNumericalCol()]
        cat_col = dataset[self.getCategoricalCol()]
        return dataset.withColumn(out_col, udf(f, t)(num_col, cat_col))

