name := "spark-ml"

version := "0.1"

scalaVersion := "2.12.8"
ensimeIgnoreScalaMismatch:=true
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"