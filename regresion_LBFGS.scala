import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val data = sc.textFile("day.csv")

val filterRDD = data.zipWithIndex().collect { case (r, i) if i != 0 => r }

val labeledPoints = filterRDD.map {line =>
 val valores = line.split(',').map(_.toDouble)
 val features = Vectors.dense(valores.init)
 val label = valores.last
 LabeledPoint(label, features)
}	

labeledPoints.cache

val modelBFGS = new LogisticRegressionWithLBFGS()
    modelBFGS.setNumClasses(5000)
    modelBFGS.setIntercept(true)

val modelo = modelBFGS.run(labeledPoints)


//Predicciendo el modelo 
val predAndObs = labeledPoints.map{ case LabeledPoint(label, features) => val prediction = modelo.predict(features)
(prediction, label)
}

modelo.clearThreshold

val scorAndLabels = labeledPoints.map{ case LabeledPoint(label, features) => val score = modelo.predict(features)
(score, label) 
}


import org.apache.spark.mllib.evaluation.RegressionMetrics
val regressionMetrics2 = new RegressionMetrics(predAndObs) 
val MSE = regressionMetrics2.meanSquaredError
val R2 = regressionMetrics2.r2

//residuos 
predAndObs.map(_.toString.replace(")","").replace("(","")).saveAsTextFile("residuos.csv")



// Precision by label
import org.apache.spark.mllib.evaluation.MulticlassMetrics
val metrics = new MulticlassMetrics(predAndObs)
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}


