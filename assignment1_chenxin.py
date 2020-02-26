# Databricks notebook source
#set file path
json_filePath="/FileStore/tables/AB_NYC_2019.JSON"
csv_filePath="/FileStore/tables/AB_NYC_2019.csv"

# COMMAND ----------

#Q1A
#read in json file with an inferred schema
ab=spark.read\
.format('json')\
.option('inferSchema','true')\
.option('mode','FAILFAST')\
.load(json_filePath).show(5)

# COMMAND ----------

#Q1B
# read csv file using a manually defined schema
from pyspark.sql.types import StructField, StructType, StringType, LongType, DateType, FloatType
#set manual schema
myManualSchema = StructType ([
  StructField("id", StringType(),True),
  StructField("name", StringType(),True),
  StructField("host_id", StringType(),True),
  StructField("neighbourhood_group", StringType(),True),
  StructField("neighbourhood", StringType(),True),
  StructField("latitude", LongType(),False),
  StructField("longtitude", LongType(),False),
  StructField("room_type", StringType(),True),
  StructField("price", LongType(),False),
  StructField("minimum_nights", LongType(),False),
  StructField("number_of_reviews", LongType(),False),
  StructField("last_review", DateType(),True),
  StructField("reviews_per_month", FloatType(),False),
  StructField("calculated_host_listings_count", LongType(),False),
  StructField("availability_365", LongType(),False)
])

# read csv file with myManualSchema
ab_schema=spark\
.read\
.format("csv")\
.option("header","true")\
.schema(myManualSchema)\
.load(csv_filePath)\
.coalesce(5)

ab_schema.show(5)
ab_schema.printSchema()

# COMMAND ----------

ab_schema.createOrReplaceTempView("ab_schema")

# COMMAND ----------

#Q2: Get the summary values separately for all numeric columns and all string columns
#over view
ab_schema.describe().show()

# COMMAND ----------

#summary of numeric columns
ab_schema.select("price").summary("min","max","mean").show()
ab_schema.select("number_of_reviews").summary("min","max","mean").show()
ab_schema.select("minimum_nights").summary("min","max","mean").show()
from pyspark.sql.functions import min, max
ab_schema.select(min("reviews_per_month"), max("reviews_per_month")).show()
ab_schema.select(min("calculated_host_listings_count"), max("calculated_host_listings_count")).show()
spark.sql("select max(availability_365) from ab_schema").show()
spark.sql("select max(last_review) from ab_schema").show()

# COMMAND ----------

#summary of string columns
from pyspark.sql.functions import countDistinct
ab_schema.select("neighbourhood").distinct().show()
ab_schema.select("neighbourhood_group").distinct().show()
ab_schema.select(countDistinct("neighbourhood_group")).show()
ab_schema.select("room_type").distinct().show()
ab_schema.select(countDistinct("name")).show()

# COMMAND ----------

#Q3: Calculate the average price and its standard deviation by room_type and per neighborhood group and neighborhood. 
#Create a statement using SQL, DF, and a grouping sets function.
from pyspark.sql.functions import grouping_id, avg, stddev,col, desc
cubeGroupAB=ab_schema.cube("neighbourhood","neighbourhood_group", "room_type").agg(grouping_id().alias("grouped"),avg(col("price")),stddev(col("price"))).orderBy(desc("grouped"))
cubeGroupAB.show(5)

# COMMAND ----------

#Q4: Handle the missing values in the dataset as follows: for neighbourhood and neighbourhood_group, replace the missing values with ‘NA’; for “availability_365”, replace the missing values with 0.
fill_cols_vals = {"neighbourhood": "NA", "neighbourhood_group": "NA", "availability_365" : 0}
ab_schema.na.fill(fill_cols_vals).show(2)

# COMMAND ----------

#Q5: Get the top 3 places where it’s cheapest to stay in Manhattan for 3 days.
from pyspark.sql.functions import expr
#define filter
neighbourhoodFilter = col("neighbourhood_group")=="Manhattan"
nightFilter = col("minimum_nights") > 3
#order by price*3
top3 = ab_schema.where(neighbourhoodFilter & nightFilter)\
                .select(col("name"), col("host_id"), col("neighbourhood"),col("neighbourhood_group"),expr("3*price as total"))\
                .orderBy("total").show(3)

