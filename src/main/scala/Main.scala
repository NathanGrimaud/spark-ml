import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, StringIndexer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.streaming._

case class LabeledText(category: Integer, text: String)

object ML{
  def train(spark: SparkSession): PipelineModel = {
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


    val regexCleaner = new CleanInputTextTransformer()
      .setInputCol("text")
      .setOutputCol("cleaned_text")

    val regexTokenizer =  new RegexTokenizer()
      .setInputCol("cleaned_text")
      .setOutputCol("words")
      .setPattern("\\W") 

    val stopwords = Array("is")
    val stopWordsRemover = new StopWordsRemover()
        .setInputCol("words")
        .setOutputCol("filtered")
        .setStopWords(stopwords)

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
          regexCleaner,
          regexTokenizer,
          stopWordsRemover,
          hashingTF,
          idf,
          stringIndexer,
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
    lrModel
  }

  case class Log( text:String, value: Int)

  def listenStream(conf: SparkConf, session: SparkSession): StreamingContext = {
    import session.implicits._
    val ssc = new StreamingContext(session.sparkContext, Seconds(5))
    val lines = ssc.socketTextStream("localhost", 9999)
    // we have to train our model before being able to predict
    val model = ML.train(session)
    lines.foreachRDD((linesRDD: RDD[String])=>{
      val transformed = linesRDD.map((line:String)=>{
        ML.Log(line,4)
      })
      val df = transformed.toDF()
      df.printSchema()
      val predictions = model.transform(df)
      predictions
        .select("text","label","probability","prediction")
        .show(50)

     })
    ssc.start()
    return ssc
  }
}

object Main extends App{

  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.OFF)

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("log parser")

  val session = SparkSession.builder.config(conf).getOrCreate()
  val ssc = ML.listenStream(conf, session)
  
  ssc.awaitTermination()
  session.sparkContext.stop()
}
