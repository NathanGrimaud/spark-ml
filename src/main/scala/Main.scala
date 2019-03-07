import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover, StringIndexer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Encoders, SQLContext, SparkSession, Row, DataFrame, Dataset}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF,IDF}

// https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35

case class LabeledText(category: Integer, text: String)

object ML{
  def train(spark: SparkSession){
    //We read the data from the file taking into account there's a header.
    //na.drop() will return rows where all values are non-null.
    val df = spark.read
      .format("com.databricks.spark.csv")
      .option("header", true) // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("delimiter",";")
      .csv("parsed.csv")
      .na
      .drop()

    df.printSchema()

    // split the input to get data on wich we can test our model
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3),seed= 100)

    val regexTokenizer =  new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val stopwords = Array("is")
    val stopWordsRemover = new StopWordsRemover()
        .setInputCol("words")
        .setOutputCol("filtered")
        .setStopWords(stopwords)

    //val countVectorizer = new CountVectorizer()
    //  .setInputCol("filtred")
    // .setOutputCol("features")
    //.setVocabSize(10000)
    //.setMinDF(5)

    val hashingTF = new HashingTF()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setNumFeatures(10000)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .setMinDocFreq(5)

    val stringIndexer = new StringIndexer()
    .setInputCol("value")
        .setOutputCol("label")


    val lr = new LogisticRegression()
      .setMaxIter(20)
      .setRegParam(0.3)
      .setElasticNetParam(0.0)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          regexTokenizer,
          stopWordsRemover,
          hashingTF,
          idf,
          stringIndexer,
          //implement tf-idf here
          lr
        )
        )

    val lrModel = pipeline.fit(trainingData)
    lrModel.write.overwrite.save("lrModel")
    val predictions = lrModel.transform(testData)

    // our row is like : 
    // |text|value|words|filtred|features|label|rawPrediction|probability|prediction|

    predictions
      .filter(_.getInt(1) != 0)
      .select("text","label","probability","prediction")
      .show(50)

    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setPredictionCol("prediction")

    val accuracy = evaluator.evaluate(predictions)

    println(s"accuracy : ${accuracy}")

  }

  case class Log(text:String, value: Int)

  def predict(spark: SparkSession, log:String){
    val model = PipelineModel.load("lrModel")
    val parsedLog = Log(log,4)
    import spark.implicits._
    // from implicits :
    val logs = List(parsedLog)
    val df = logs.toDF()
    df.printSchema()

    val predictions = model.transform(df)
    predictions
      .select("text","label","probability","prediction")
      .show(50)
  }
}

object Main extends App{

  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.OFF)

  val spark = SparkSession.builder
    .appName("log parser")
    .master("local")
    .getOrCreate()

  ML.train(spark)

  ML.predict(spark, "déc. 29 17:46:34 localhost.localdomain unix_chkpwd[2427]: password check failed for user (root)")

  spark.stop()


}
