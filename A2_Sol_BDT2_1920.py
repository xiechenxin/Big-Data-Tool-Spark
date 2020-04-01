# Databricks notebook source
# DBTITLE 1,Solution Assignment 2: Recommendation Engines, Graph Analytics, and Streaming
#This notebook contains the solution for Assignment 2 of BDT2.

# COMMAND ----------

# DBTITLE 1,Read in the data for Q1 & Q2
#Read data
user_artists = spark.read.format("csv").option("delimiter", "\t").option("header","true").option("inferSchema","true").load("/FileStore/tables/music/user_artists.dat")
user_friends = spark.read.format("csv").option("delimiter", "\t").option("header","true").option("inferSchema","true").load("/FileStore/tables/music/user_friends.dat")
user_taggedartists = spark.read.format("csv").option("delimiter", "\t").option("header","true").option("inferSchema","true").load("/FileStore/tables/music/user_taggedartists.dat")

user_artists.show(3)
user_friends.show(3)
user_taggedartists.show(3)

# COMMAND ----------

# DBTITLE 1,Q1: Artist Recommender System
"""
Create an artist recommender system to recommend 10 music artists to users. You can use Alternative Least Squares (ALS) as an algorithm. Create a pipeline for your solution and test two hyperparameter value settings for the elastic net parameter: [0.9, 0.1]. You can use the column ‘weight’ as the rating column for ALS. Evaluate your solution using a ranking metric.
"""

# COMMAND ----------

#Use table user_artists to make a recommendation
#column "weight" represents the listening count that a user listened to an artist

# COMMAND ----------

user_artists.describe().show()

# COMMAND ----------

#Split data
training, test = user_artists.randomSplit([0.7, 0.3])

# COMMAND ----------

from pyspark.ml.recommendation import ALS

#Estimate model
als = ALS()\
  .setMaxIter(5)\
  .setUserCol("userID")\
  .setItemCol("artistID")\
  .setRatingCol("weight")\
  .setImplicitPrefs(True)\
  .setColdStartStrategy("drop")\
  .setNonnegative(True)

#alsModel = als.fit(training,[{"regparam",0.05},{"regparam",0.80}])
#predictions = alsModel.transform(test)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

#Define pipeline
pipeline = Pipeline().setStages([als])

#Set param grid
params = ParamGridBuilder()\
  .addGrid(als.regParam, [0.9, 0.01])\
  .build()

#Cross-validation of entire pipeline
cv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEstimatorParamMaps(params)\
  .setEvaluator(RegressionEvaluator(labelCol="weight",metricName="rmse"))\
  .setNumFolds(2)

#Run cross-validation, and choose the best set of parameters.
alsModel = cv.fit(training)

# COMMAND ----------

alsBestModel = alsModel.bestModel.stages[-1]

# COMMAND ----------

#Output top 10 music recommendations for each user
alsBestModel.recommendForAllUsers(10).selectExpr("userID", "explode(recommendations)").show()

# COMMAND ----------

#Use a ranking metric to compare the performance
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr

predictions = alsBestModel.transform(test)

user_artists_actual = predictions\
  .groupBy("userID")\
  .agg(expr("collect_set(artistID) as artists"))

user_artists_pred = predictions\
  .orderBy(col("userID"), expr("prediction DESC"))\
  .groupBy("userID")\
  .agg(expr("collect_list(artistID) as artists"))

user_artists_diff = user_artists_actual.join(user_artists_pred, ["userID"])\
  .rdd.map(lambda row: (row[1], row[2][:10]))

ranks = RankingMetrics(user_artists_diff)

#Get precision
print(ranks.meanAveragePrecision)

# COMMAND ----------

# DBTITLE 1,Q2: Creating a Graph
"""
Create a graph using the table user_friends. Determine the top 5 most important users and the top 5 users with the highest indegree/outdegree.
"""

# COMMAND ----------

#Change some column names to prepare to create a graph
vertices = user_friends.withColumnRenamed("userID","id").select("id").distinct()
edges = user_friends.withColumnRenamed("userID","src").withColumnRenamed("friendID","dst")
vertices.show(3)
edges.show(3)

# COMMAND ----------

#Build GraphFrame object
from graphframes import GraphFrame

friendGraph = GraphFrame(vertices, edges)
friendGraph.vertices.show(3)
friendGraph.edges.show(3)

# COMMAND ----------

#Determine the top 5 most important users
ranks = friendGraph.pageRank(resetProbability=0.15, maxIter=10) #0.15 is same as original value in Google Search Engine; function returns a GraphFrame
ranks.vertices.orderBy(desc("pagerank")).select("id", "pagerank").limit(5).show(5,False)

# COMMAND ----------

#Determine the top 5 users with the highest indegree/outdegree
from pyspark.sql.functions import desc

inDeg = friendGraph.inDegrees
inDeg.orderBy(desc("inDegree")).limit(5).show(5, False)

#Alternative
#outDeg = friendGraph.outDegrees
#outDeg.orderBy(desc("outDegree")).limit(5).show(5, False)

# COMMAND ----------

# DBTITLE 1,Q3: Create a Stream for Time Passed
"""
Using the dataset “activity-data”, create a stream that outputs in one table:
- real-time time difference between arrival_time and creation_time aggregated by user and device.
- historical minimum and maximum time difference values between arrival_time and creation_time aggregated across all users.
- real-time updated difference between real-time time difference and historical minimum and maximum time difference.
"""

# COMMAND ----------

import pyspark.sql.functions as f

# COMMAND ----------

filePath = "/FileStore/tables/streaming/"

# COMMAND ----------

#Inspect the data to read the activity-data "statically"
#filePath = "/FileStore/tables/activity-data/" #you can only define a file path.
staticDF = spark.read\
  .json(filePath)

dataSchema = staticDF.schema
staticDF.show(4)

# COMMAND ----------

#Create timestamps for arrival_time and creation_time
staticDF = staticDF\
  .selectExpr("*","cast(cast(Creation_Time as double)/1000000000 as timestamp) as creation_time_ts", "cast(cast(Arrival_Time as double)/1000 as timestamp) as arrival_time_ts")

# COMMAND ----------

#Create a streaming DF
streamingDF = spark.readStream\
  .schema(dataSchema)\
  .option("maxFilesPerTrigger", 1)\
  .json(filePath)

# COMMAND ----------

#Create timestamps for arrival_time and creation_time
streamingDF = streamingDF\
  .selectExpr("*","cast(cast(Creation_Time as double)/1000000000 as timestamp) as creation_time_ts", "cast(cast(Arrival_Time as double)/1000 as timestamp) as arrival_time_ts")

# COMMAND ----------

#historical minimum and maximum time difference values between arrival_time and creation_time aggregated across all users.
histData = staticDF.selectExpr("max(unix_timestamp(arrival_time_ts) - unix_timestamp(creation_time_ts)) as max_hist_timediff", "min(unix_timestamp(arrival_time_ts) - unix_timestamp(creation_time_ts)) as min_hist_timediff")
histData.show()

# COMMAND ----------

spark.conf.set("spark.sql.crossJoin.enabled", "true")

# COMMAND ----------

#Create streaming query
comboDataStream = streamingDF\
  .withColumn("realtime_timediff",f.col("arrival_time_ts").cast("double") - f.col("creation_time_ts").cast("double"))\
  .groupBy("user","device").agg(f.min(f.col("realtime_timediff")).alias("min_realtime_timediff"),f.max(f.col("realtime_timediff")).alias("max_realtime_timediff"))\
  .join(histData)\
  .selectExpr("*","min_hist_timediff - min_realtime_timediff as dev_min","max_hist_timediff - max_realtime_timediff as dev_max")\
  .writeStream\
  .queryName("q3")\
  .format("memory")\
  .outputMode("complete")\
  .start()

# COMMAND ----------

comboDataStream.stop()

# COMMAND ----------

#Query the table device_join_counts every 5 seconds
from time import sleep
for x in range(5):
    spark.sql("select * from q3").show(3)
    sleep(5)

# COMMAND ----------

# DBTITLE 1,Q4: Create a Stream for Distance Travelled
"""
Using the dataset “activity-data”, create a stream that outputs in one table the total number of meters user g travels per activity in time intervals of resp. 15 minutes and 30 minutes. Order the table by the most distance travelled per activity. Hint: you can use the columns x, y, z to calculate the distance travelled.
"""

# COMMAND ----------

import pyspark.sql.functions as f

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

#Create a historical df to determine the starting position
hist_q4 = staticDF.where(f.col("User")=='g').orderBy("gt","creation_time_ts").groupBy("gt").agg(f.first("x").alias("firstX"),f.first("y").alias("firstY"),f.first("z").alias("firstZ"))
hist_q4.show(10,False)

# COMMAND ----------

#Create a stream to calculate the difference between that starting position and the incoming data on the location of the user (15 minutes).
stream_q4_15m = streamingDF\
  .where(f.col("User")=='g')\
  .select("gt", "creation_time_ts", "x", "y", "z")\
  .join(hist_q4,["gt"])\
  .groupBy("gt",window(col("creation_time_ts"), "15 minutes").alias('tw'))\
  .agg(sum(sqrt(pow(col("x")-col("firstX"), 2)+pow(col("y")-col("firstY"), 2)+pow(col("z")-col("firstZ"), 2))).alias("total_distance_travelled"))\
  .writeStream\
  .queryName("distance_travelled_15m")\
  .format("memory")\
  .outputMode("complete")\
  .start()

# COMMAND ----------

#Query the table every 5 seconds
from time import sleep
for x in range(5):
    spark.sql("""
    select * 
    from distance_travelled_15m
    order by total_distance_travelled desc
    """).show(3,False)
    sleep(5)

# COMMAND ----------

#Create a stream to calculate the difference between that starting position and the incoming data on the location of the user (30 minutes).
stream_q4_30m = streamingDF\
  .where(f.col("User")=='g')\
  .select("gt", "creation_time_ts", "x", "y", "z")\
  .join(hist_q4,["gt"])\
  .groupBy("gt",window(col("creation_time_ts"), "30 minutes").alias('tw'))\
  .agg(sum(sqrt(pow(col("x")-col("firstX"), 2)+pow(col("y")-col("firstY"), 2)+pow(col("z")-col("firstZ"), 2))).alias("total_distance_travelled"))\
  .writeStream\
  .queryName("distance_travelled_30m")\
  .format("memory")\
  .outputMode("complete")\
  .start()

# COMMAND ----------

#Query the table every 5 seconds
from time import sleep
for x in range(5):
    spark.sql("""
    select * 
    from distance_travelled_30m
    order by total_distance_travelled desc
    """).show(3,False)
    sleep(5)
