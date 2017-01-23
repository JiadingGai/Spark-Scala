package com.vario.demo

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext

// Jiading GAI
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.DenseVector

import org.apache.spark.sql.functions.udf

object Demo {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Demo")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val df0 = sqlContext.read
                        .format("com.databricks.spark.csv")
                        .option("header", "true") // Use first line of all files as header
                        .option("inferSchema", "true") // Automatically infer data types
                        .load("data/demo.csv")

    /* - Hard-coded a demo dataset.
    if (false) {
      val df = sqlContext.createDataFrame(
        Seq((1.0,9.10,"WINK",0.01,"yes"),
            (1.0,9.20,"WINK",0.0346,"yes"),
            (0.0,9.10,"PAST",0.1,"yes"),
            (1.0,9.30,"WINK",0.0347,"yes"),
            (0.0,9.70,"PAST",0.05,"no"),
            (1.0,9.80,"WINK",0.1,"no"),
            (0.0,9.90,"WINK",0.0347,"no"),
            (0.0,8.10,"LOCK",0.0347,"no"),
            (0.0,8.52,"PAST",0.0347,"yes"),
            (0.0,8.66,"QUIK",0.0347,"no"),
            (0.0,7.84,"PAST",0.1,"yes"),
            (0.0,6.92,"LOCK",0.0347,"no"),
            (0.0,5.52,"PAST",0.05,"yes"),
            (0.0,5.91,"LOCK",0.08,"no"),
            (0.0,6.11,"PAST",0.1,"yes"),
            (1.0,4.12,"PAST",0.1,"no"),
            (1.0,3.54,"PAST",0.0347,"yes"),
            (0.0,3.96,"LOCK",0.0346,"no"),
            (1.0,2.01,"PAST",0.05,"yes"),
            (0.0,1.55,"PAST",0.08,"no"))
        ).toDF("TARGET","RATING","STYLE","FAILURE_RATE","SATISFACTION")
      df.show()
      df.printSchema()
    }
    */

    /* Fill in missing value */
    val df = df0.na.fill("missing").na.fill(-1.0)
    df.printSchema()
    df.show()

    val CatVarList = sc.textFile("cat_var.list").collect()
    val CatVarListEncoded = CatVarList.map(cname => s"${cname}_index")
    val TargetVariableList = Array("TARGET", "TARGET_index")

    val transformers = df.columns.filter(cname => CatVarList.contains(cname))
                                 .map(cname => new StringIndexer().setInputCol(cname).setOutputCol(s"${cname}_index"))

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

    val dfModelReady = assembler.transform(dfIndexed)

    //val clf = new GBTClassifier().setLabelCol("TARGET_index").setFeaturesCol("features").setMaxIter(3)

    //val clf = new DecisionTreeClassifier().setLabelCol("TARGET_index")
    //                                      .setFeaturesCol("features")
    //                                      .setImpurity("gini")

    val clf = new RandomForestClassifier().setLabelCol("TARGET_index")
                                          .setFeaturesCol("features")
                                          .setNumTrees(66)
                                          .setImpurity("gini")

    //val layers = Array[Int](4, 3, 2)
    //val clf = new MultilayerPerceptronClassifier().setLayers(layers)
    //                                              .setBlockSize(128)
    //                                              .setSeed(1234L)
    //                                              .setMaxIter(100)
    //                                              .setLabelCol("TARGET_index")
    //                                              .setFeaturesCol("features")

    //val stages: Array[org.apache.spark.ml.PipelineStage] = assembler
    //val pipeline = new Pipeline().setStages(Array(assembler, clf))
    val clfModel = clf.fit(dfModelReady)
    clfModel.transform(dfModelReady).show()

    // Make validations
    val dfv0 = sqlContext.read
                         .format("com.databricks.spark.csv")
                         .option("header", "true") // Use first line of all files as header
                         .option("inferSchema", "true") // Automatically infer data types
                         .load("data/demo_val.csv")
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

    var dfvModelReady = assembler.transform(dfvIndexed)

    val predictions = clfModel.transform(dfvModelReady)
    println("[Model predictions]")

    /* Manipulate scores and GetTopXPR */
    predictions.show()

    //Obtain scoreAndLabels and cumsum
    val ScoreAndLabels = predictions.map(row => (row.getAs[DenseVector]("probability")(1), row.getAs[Double]("TARGET_index"))).sortBy(-_._1)
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
    val metrics = new BinaryClassificationMetrics(ScoreAndLabels)
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)

    // Select example rows to display
    predictions.select("TARGET_index", "prediction").show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("TARGET_index")
                                                           .setPredictionCol("prediction")
                                                           .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    if (true) {
      println("Learned classification clf model:\n" + clfModel.toDebugString)
    }
  }
}
