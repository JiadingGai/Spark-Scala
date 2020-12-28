import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val project_path = "/Users/jiadinggai/dev/SPARK/Spark-Scala/"

val df0 = spark.read.format("csv").option("header", "true").
            option("inferSchema", "true").
            load(project_path + "demo_rf/data/demo.csv")

/* Fill in missing value */
val df = df0.na.fill("missing").na.fill(-1.0)
df.printSchema()
df.show()

val CatVarList = spark.read.textFile(project_path + "demo_rf/cat_var.list").collect()
val CatVarListEncoded = CatVarList.map(cname => s"${cname}_index")
val TargetVariableList = Array("TARGET", "TARGET_index")

val transformers = df.columns.filter(cname => CatVarList.contains(cname)).
                     map(cname => new StringIndexer().setInputCol(cname).setOutputCol(s"${cname}_index"))

/*
  // Neat trick to find the type of any variable:
  // The compiler will raise an error, providing you information about the actual return type it found.
  // For example, the code will show that transformers has Array[org.apache.spark.ml.feature.StringIndexer].
  val transformersType: Nothing = transformers
*/
var StringIndexerModels: Array[org.apache.spark.ml.feature.StringIndexerModel] = Array()
for (i <- 0 until transformers.length) {
  StringIndexerModels = StringIndexerModels :+ transformers(i).fit(df)
}

var dfIndexed = df
for (i <- 0 until StringIndexerModels.length) {
  dfIndexed = StringIndexerModels(i).transform(dfIndexed)
}
dfIndexed.show()

val FeatureCols = dfIndexed.columns.filter(cname => !TargetVariableList.contains(cname) && !CatVarList.contains(cname))
val assembler = new VectorAssembler().setInputCols(FeatureCols).setOutputCol("features")
println("FeatureCols.size = " + FeatureCols.size)
FeatureCols.foreach(println)

//val clf = new GBTClassifier().setLabelCol("TARGET_index").setFeaturesCol("features").setMaxIter(3)
//val clf = new DecisionTreeClassifier().setLabelCol("TARGET_index").setFeaturesCol("features").setImpurity("gini")
val clf = new RandomForestClassifier().setLabelCol("TARGET_index").
                                       setFeaturesCol("features").
                                       setNumTrees(66).
                                       setImpurity("gini")

val pipeline = new Pipeline().setStages(Array(assembler, clf))
val model = pipeline.fit(dfIndexed)
model.transform(dfIndexed).show()

// Load validation datasets.
val dfv0 = spark.read.format("csv").option("header", "true").
            option("inferSchema", "true").
            load(project_path + "demo_rf/data/demo_val.csv")
var dfv = dfv0.na.fill("missing").na.fill(-1.0)

/* Detect unseen values and replace with the most frequent level */
for (i <- 0 until StringIndexerModels.length) {
  val mapudf = udf { label: String =>
                 if (StringIndexerModels(i).labels.contains(label))
                   label
                 else
                   StringIndexerModels(i).labels(0)
               }

  val ColumnName = StringIndexerModels(i).getInputCol
  println("[Jiading Gai] " + ColumnName)
  dfv = dfv.withColumn(ColumnName, mapudf(dfv(ColumnName)))
}

/* Apply transformers separately */
var dfvIndexed = dfv
for (i <- 0 until StringIndexerModels.length) {
  dfvIndexed = StringIndexerModels(i).transform(dfvIndexed)
}
dfvIndexed.show()

val predictions = model.transform(dfvIndexed)
println("[Model predictions]")

/* Manipulate scores and GetTopXPR */
predictions.show()

//Obtain scoreAndLabels and cumsum
val ScoreAndLabels = predictions.map(row => (row.getAs[DenseVector]("probability")(1), row.getAs[Double]("TARGET_index"))).
                                 orderBy(desc("_1"))
val SortedLabels = ScoreAndLabels.toDF.select("_2").rdd.map(r => r(0).asInstanceOf[Double]).collect()

var SortedWeights = new Array[Double](SortedLabels.length)
for (i <- 0 until SortedLabels.length) {
  if (SortedLabels(i) == 1.0) {
    SortedWeights(i) = 1.0
  } else {
    SortedWeights(i) = 40.0
  }
}

var CumTotal = SortedWeights.map(_.toInt).scanLeft(0)(_ + _).tail.map(_.toDouble)
var CumBad = SortedLabels.map(_.toInt).scanLeft(0)(_ + _).tail.map(_.toDouble)
var CumGood = new Array[Double](SortedLabels.length)
for (i <- 0 until SortedLabels.length) {
  CumGood(i) = CumTotal(i) - CumBad(i)
}

var CumTotalNormalized = new Array[Double](SortedLabels.length)
for (i <- 0 until SortedLabels.length) {
  CumTotalNormalized(i) = CumTotal(i) / CumTotal(CumTotal.length - 1)
}

val TDRpct = Array[Double](0.0025, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
var TDR = new Array[Double](TDRpct.length)
for (i <- 0 until TDRpct.length) {
  val pct = TDRpct(i)
  var Idx = 0
  for (j <- 1 until CumTotalNormalized.length) {
      if (CumTotalNormalized(j) >= pct && CumTotalNormalized(j-1) <= pct) {
        Idx = j
      }
  }
  println("TDR pct " + pct + " at " + Idx + " = " + CumBad(Idx) / CumBad(CumBad.length - 1))
}

/* Compute Metrics */
val metrics = new BinaryClassificationMetrics(ScoreAndLabels.rdd)
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

