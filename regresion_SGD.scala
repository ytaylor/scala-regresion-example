import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val data = sc.textFile("/home/yanelis/Downloads/TEAD/ENTREGABLES/Proyecto_2/day.csv")

val filterRDD = data.zipWithIndex().collect { case (r, i) if i != 0 => r }

val labeledPoints = filterRDD.map {line =>
 val valores = line.split(',').map(_.toDouble)
 val features = Vectors.dense(valores.init)
 val label = valores.last
 LabeledPoint(label, features)
}	

labeledPoints.cache


val alg = new LinearRegressionWithSGD()
    alg.setIntercept(false)
    alg.optimizer.setNumIterations(100)
    alg.optimizer.setStepSize(10)

//antes de seguir escalamos las variables predictoras o independientes (features)
import org.apache.spark.mllib.feature.StandardScaler
val scaler = new StandardScaler(true, true).
fit(labeledPoints.map(x => x.features))

val parsedScaledData = labeledPoints.map(point =>
       LabeledPoint(point.label,scaler.transform(point.features))
       ).cache() 

//ajustamos el modelo: obsérvese que no funciona train, tiene que ser run
val modelSGD = alg.run(parsedScaledData) 


// check the model parameters
val intercept = modelSGD.intercept
val weights = modelSGD.weights

// get actual and predicted label for each observation in the training set
val observedAndPredictedLabels = labeledPoints.map { observation =>
                val predictedLabel = modelSGD.predict(observation.features)
                (observation.label, predictedLabel)
                }

// calculate square of difference between predicted and actual label for each observation
val squaredErrors = observedAndPredictedLabels.map{case(actual, predicted) =>
                    math.pow((actual - predicted), 2)
                    }

// calculate the mean of squared errors.
val meanSquaredError = squaredErrors.mean()


//utilizando la clase RegressionMetrics: creo una instancia
import org.apache.spark.mllib.evaluation.RegressionMetrics
val regressionMetrics2 = new RegressionMetrics(observedAndPredictedLabels) 
val MSE = regressionMetrics2.meanSquaredError
val R2 = regressionMetrics2.r2


///////////////////////////////////////////////////////////////////////
// encontrar el nº de iteraciones y paso del SGD adecuado
// con escalado: posible
///////////////////////////////////////////////////////////////////////
val iterNums = Array(1000,10000)
val stepSizes = Array(0.001,0.01, 0.1,1,10)
for(numIter <- iterNums; step <- stepSizes) {
   val alg = new LinearRegressionWithSGD()
       alg.setIntercept(true)
       alg.optimizer.setNumIterations(numIter)
       alg.optimizer.setStepSize(step)
       alg.optimizer.setConvergenceTol(1.0E-6)
       
   //escalado    
   val scaler = new StandardScaler(true, true).fit(labeledPoints.map(x => x.features))
   val parsedScaledData = labeledPoints.map(point =>
       LabeledPoint(point.label,scaler.transform(point.features))
       ).cache() 
   
   val model = alg.run(parsedScaledData)
   
   val predAndObs3 = parsedScaledData.map(point => {
          val prediction = model.predict(point.features)
          (prediction, point.label)
          })
   
   val regM = new RegressionMetrics(predAndObs3,false)
   val mse = regM.meanSquaredError
   val r2 = regM.r2
   println("%d, %12.10f -> %10.7f, %10.7f".format(numIter, step, mse, r2))
   //predAndObs3.foreach(println)
}







