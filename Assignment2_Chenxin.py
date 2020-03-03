# Databricks notebook source
# Q1: Create an artist recommender system to recommend 10 music artists to users. You can use Alternative Least Squares (ALS) as an algorithm. Create a pipeline for your solution and test two hyperparameter value settings for the elastic net parameter: [0.9, 0.1]. You can use the column ‘weight’ as the rating column for ALS. Evaluate your solution using a ranking metric.

# COMMAND ----------

# read in the data
artist = spark.read\
        .format("csv")\
        .option("header", "true")\
        .option("delimiter", "\t")\
        .load("/FileStore/tables/user_artists.dat")
 
artist.show(4)

# COMMAND ----------

# change userID and artistID to int, weight to float
from pyspark.sql.types import IntegerType, FloatType
artist = artist.withColumn("userID", artist["userID"].cast(IntegerType()))
artist = artist.withColumn("artistID", artist["artistID"].cast(IntegerType()))
artist = artist.withColumn("weight", artist["weight"].cast(FloatType()))
artist.printSchema()

# COMMAND ----------

artist.describe().show()

# COMMAND ----------

from pyspark.ml.recommendation import ALS

#Split data
training, test = artist.randomSplit([0.9, 0.1])

#Estimate model
als = ALS()\
  .setMaxIter(5)\
  .setRegParam(0.01)\
  .setImplicitPrefs(True)\
  .setUserCol("userID")\
  .setItemCol("artistID")\
  .setRatingCol("weight")

alsModel = als.fit(training)
predictions = alsModel.transform(test)

# COMMAND ----------

#Output top 10 artist recommendations for each user
alsModel.recommendForAllUsers(10).selectExpr("userID", "explode(recommendations)").show()

# COMMAND ----------

# Evaluate with Ranking Metrics
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr

perUserActual = predictions\
  .where("weight > 2.5")\
  .groupBy("userID")\
  .agg(expr("collect_set(artistID) as artists"))

perUserPredictions = predictions\
  .orderBy(col("userID"), expr("prediction DESC"))\
  .groupBy("userID")\
  .agg(expr("collect_list(artistID) as artists"))

perUserActualvPred = perUserActual.join(perUserPredictions, ["userID"])\
  .rdd.map(lambda row: (row[1], row[2][:15]))

ranks = RankingMetrics(perUserActualvPred)

# COMMAND ----------

#How precise is algorithm (how many artists are correctly ranked)?
print(ranks.meanAveragePrecision)

#How many of the first 5 artists are correct on average for all users?
print(ranks.precisionAt(5))

# COMMAND ----------

# Q2: Create a graph using the table user_friends. Determine the top 5 most important users and the top 5 users with the highest indegree/outdegree.

# COMMAND ----------

# read in the data
friends = spark.read\
        .format("csv")\
        .option("header", "true")\
        .option("delimiter", "\t")\
        .load("/FileStore/tables/user_friends.dat")

friends.show(4)

# COMMAND ----------

# change userID and friendID to int, weight to float
from pyspark.sql.types import IntegerType
friends = friends.withColumn("userID", friends["userID"].cast(IntegerType()))
friends = friends.withColumn("friendID", friends["friendID"].cast(IntegerType()))

friends.printSchema()

# COMMAND ----------

friendsVertices = friends.withColumnRenamed("userID", "id").distinct().show()

# COMMAND ----------

#Build a graph
friendsVertices = friends.withColumnRenamed("userID", "id").distinct()
friendsEdges = friends.withColumnRenamed("userID", "src")\
              .withColumnRenamed("friendID", "dst")

#Build GraphFrame object
from graphframes import GraphFrame

friendsGraph = GraphFrame(friendsVertices, friendsEdges)
friendsGraph.cache()

# COMMAND ----------

#Inspect the GraphFrame
friendsGraph.vertices.show(3,False)
friendsGraph.edges.show(3,False)

# COMMAND ----------

#Count the number of friends in a given user (in-degree) and out of a given user (out-degree)
from pyspark.sql.functions import desc
inDeg = friendsGraph.inDegrees
inDeg.orderBy(desc("inDegree")).show(5, False)

outDeg = friendsGraph.outDegrees
outDeg.orderBy(desc("outDegree")).show(5, False)

# COMMAND ----------

#Look at the ratio of in-degree and out-degree
#Higher value tells us where a larger number of trips end (but rarely begin), lower value tells us where trips often begin (but infrequently end)
degreeRatio = inDeg.join(outDeg, "id")\
  .selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio")

degreeRatio.orderBy(desc("degreeRatio")).show(5, False)
degreeRatio.orderBy("degreeRatio").show(5, False)

# COMMAND ----------

#Q3: Using the dataset “activity-data”, create a stream that outputs in one table:
#   - real-time time difference between arrival_time and creation_time aggregated by user and device.
#   - historical minimum and maximum time difference values between arrival_time and creation_time aggregated across all users.
#   - real-time updated difference between real-time time difference and historical minimum and maximum time difference.

# COMMAND ----------

#Inspect the data to read the activity-data "statically"
filePath = "/FileStore/tables/activity-data_full/" 
staticDF = spark.read\
  .json(filePath)

dataSchema = staticDF.schema
staticDF.show(4)

# COMMAND ----------

staticDF.printSchema()

# COMMAND ----------

# Create streaming DF
streamingDF = spark.readStream\
  .schema(dataSchema)\
  .option("maxFilesPerTrigger", 1)\
  .json(filePath)


# COMMAND ----------

print("Streaming DF: " + str(streamingDF.isStreaming))
print("Static DF: " + str(staticDF.isStreaming))

# COMMAND ----------

#2. Create a processing statement ("action")
#real-time time difference between arrival_time and creation_time aggregated by user and device.
# change format of "Arrival_Time", "Creation_Time" to timestamp
from pyspark.sql.functions import col, to_timestamp, datediff
streamingDF = streamingDF\
         .withColumn("Arrival_Time", col("Arrival_Time").cast("long"))\
         .withColumn("Creation_Time", col("Creation_Time").cast("long"))

streamingDF = streamingDF\
         .withColumn("Arrival_Time2",to_timestamp(streamingDF['Arrival_Time']))\
         .withColumn("Creation_Time2",to_timestamp(streamingDF['Creation_Time']))\
         .withColumn("time_diff", datediff(col("Arrival_Time2"), col("Creation_Time2")) )

timeDiff = streamingDF.groupBy('User', 'Device').sum('time_diff')

# COMMAND ----------

timeDiff.show()

# COMMAND ----------

#Set shuffle partitions to a small value to avoid creating too many shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", 5)

# COMMAND ----------

#3. Define output sink to start the query
#Write to a memory sink which keeps an in-memory table of the results

timeDiffStream = timeDiff.writeStream\
  .queryName("timeDiff")\
  .format("memory").outputMode("complete")\
  .start()

#Complete mode: rewrite all keys along with their counts after every trigger
#Other modes: update, append

#Wait for the termination of the query (don't let the driver exit while the query is active)
#activityStream.awaitTermination()

# COMMAND ----------

#List all active streams
spark.streams.active

# COMMAND ----------

timeDiffStream.isActive

# COMMAND ----------

#Stop stream
timeDiffStream.stop()

# COMMAND ----------

# historical minimum and maximum time difference values between arrival_time and creation_time aggregated across all users.
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_timestamp, datediff
staticDF = staticDF\
         .withColumn("Arrival_Time", col("Arrival_Time").cast("long"))\
         .withColumn("Creation_Time", col("Creation_Time").cast("long"))

staticDF = staticDF\
         .withColumn("Arrival_Time2",to_timestamp(staticDF['Arrival_Time']))\
         .withColumn("Creation_Time2",to_timestamp(staticDF['Creation_Time']))\
         .withColumn("time_diff", datediff(col("Arrival_Time2"), col("Creation_Time2")) )


time_dif_his = staticDF.groupBy('User','Device').agg(F.min(staticDF.time_diff),F.max(staticDF.time_diff))
time_dif_his.show(5)

# COMMAND ----------

#real-time updated difference between real-time time difference and historical minimum and maximum time difference.
timeDiffDataStream = streamingDF.groupBy('User', 'Device').sum('time_diff')\
  .join(time_dif_his, ["User", "Device"])\
  .writeStream\
  .queryName("user_timeDiff_update")\
  .format("memory")\
  .outputMode("complete")\
  .start()


# COMMAND ----------

#Query the table user_timeDiff_update every 2 seconds
from time import sleep
for x in range(5):
    spark.sql("select * from user_timeDiff_update").show(3)
    sleep(5)

# COMMAND ----------

timeDiffDataStream.stop()

# COMMAND ----------

# Q4: Using the dataset “activity-data”, create a stream that outputs in one table the total number of meters user g travels per activity in time intervals of resp. 15 minutes and 30 minutes. Order the table by the most distance travelled per activity. Hint: you can use the columns x, y, z to calculate the distance travelled.

# COMMAND ----------

withDistance = streamingDF.withColumn("distance", sqrt(pow((streamingDF['x']), 2)+ pow((streamingDF['y']), 2)+ pow((streamingDF['y']), 2)))

# COMMAND ----------

travelMeterStream = withDistance\
  .drop("Arrival_Time", "Creation_Time", "Device", "Index", "Model", "gt")\
  .cube("User").sum("distance")\
  .where("User=='g'")\
  .writeStream\
  .queryName("travelMeter_User")\
  .format("memory")\
  .outputMode("complete")\
  .start()

# COMMAND ----------

travelMeterStream.stop()

# COMMAND ----------

#Aggregate keys over a window of time
#Create a trigger every 15 minutes
from pyspark.sql.functions import window, col
timeWindow_10m = withDistance.groupBy(window(col("event_time"), "15 minutes")).sum("distance")\
  .where("User=='g'")\
  .writeStream\
  .queryName("events_per_window")\
  .format("memory")\
  .outputMode("complete")\
  .start()

# COMMAND ----------

#Query this stream with SQL
spark.sql("""
  select * from events_per_window order by distance
""").show(10,False)

# COMMAND ----------

timeWindow_10m.stop()

# COMMAND ----------

from pyspark.sql.functions import window, col, desc
stimeWindow_30m_15m = withDistance.groupBy(window(col("event_time"), "30 minutes", "15 minutes"))\
  .count()\
  .where("User=='g'")\
  .writeStream\
  .queryName("events_per_window_slide")\
  .format("memory")\
  .outputMode("complete")\
  .start()


# COMMAND ----------

#Query this stream with SQL
spark.sql("""
  select * from events_per_window_slide order by distance
""").show(10,False)

# COMMAND ----------

stimeWindow_30m_15m.stop()
