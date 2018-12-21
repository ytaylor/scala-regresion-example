//Haciendo un liner regresion  usa el ml.
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.regression.LinearRegressionSummary


val data = sc.textFile("day.csv")
val filterRDD = data.zipWithIndex().collect { case (r, i) if i != 0 => r }

val parsedData1 = filterRDD.map(line => {
    val parts = line.split(",")
    LabeledPoint(parts(8).toDouble, Vectors.dense(parts(0).toDouble, parts(1).toDouble, parts(2).toDouble, parts(3).toDouble, parts(4).toDouble, parts(5).toDouble,parts(6).toDouble,parts(7).toDouble))
    })


val lr = new LinearRegression()
  lr.setMaxIter(1000)
  lr.setFitIntercept(true)
  lr.setSolver("normal") // "auto" o "l-bfgs"
val parsedData1DF = parsedData1.toDF.cache() // la primera columna "label", lo siguiente "features"
val model1 = lr.fit(parsedData1DF)


//coeficientes e intercept de la regresión
println(s"Intercept: ${model1.intercept} Coeficientes: ${model1.coefficients} ")

//resumen
val resumen1 = model1.summary
println(s"numIteraciones: ${resumen1.totalIterations}")
println(s"MSE: ${resumen1.meanSquaredError}")
println(s"r2: ${resumen1.r2}")
resumen1.pValues.foreach(println)  // pvalues es un miembro lazy


///Prediccion del modelo
val predAndObs = parsedData1.map{ case LabeledPoint(label, features) => val prediction = model1.predict(features)
(prediction, label)
}

model1.clearThreshold

val scorAndLabels = parsedData1.map{ case LabeledPoint(label, features) => val score = model1.predict(features)
(score, label) 
}


import org.apache.spark.mllib.evaluation.RegressionMetrics
val regressionMetrics2 = new RegressionMetrics(predAndObs) 
val MSE = regressionMetrics2.meanSquaredError
val R2 = regressionMetrics2.r2

predAndObs.map(_.toString.replace(")","").replace("(","")).saveAsTextFile("residuos.csv")
