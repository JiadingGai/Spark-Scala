package com.vario.demo

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

// Jiading GAI
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object Demo {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Demo")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

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
          (0.0,8.66,"PAST",0.0347,"no"),
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
    //System.exit(1)

    val CatVarList = sc.textFile("cat_var.list").collect()
    val CatVarListEncoded = CatVarList.map(cname => s"${cname}_index")
    val TargetVariableList = Array("TARGET")

    val transformers = df.columns.filter(cname => CatVarList.contains(cname))
                                 .map(cname => new StringIndexer().setInputCol(cname).setOutputCol(s"${cname}_index"))
    val assembler = new VectorAssembler().setInputCols(df.columns.filter(cname => !TargetVariableList.contains(cname) && !CatVarList.contains(cname)))
                                         .setOutputCol("features")
    val gbt = new GBTClassifier().setLabelCol("TARGET_index").setFeaturesCol("features").setMaxIter(3)

    val stages: Array[org.apache.spark.ml.PipelineStage] = transformers :+ assembler :+ gbt
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(df)

    // Make predictions
    val predictions = model.transform(df)

    /* Compute Metrics */
    val labelAndPreds = predictions.map(row => (row.getAs[Double]("TARGET_index"), row.getAs[Double]("prediction")))
    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)

    // Select example rows to display
    predictions.select("TARGET_index", "prediction").show()
 
    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("TARGET_index").setPredictionCol("prediction").setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    if (true) {
      val gbtModel = model.stages(4).asInstanceOf[GBTClassificationModel]
      println("Learned classification GBT model:\n" + gbtModel.toDebugString)
    }
  }
}
