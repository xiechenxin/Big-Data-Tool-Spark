# Databricks notebook source
# DBTITLE 1,Question 1A: Read the Data Using an Inferred Schema (.JSON)
#Read in the data
hotels = spark.read.format("json").option("header","true").option("inferSchema","true")\
  .load("/FileStore/tables/Assignments_BDT2_1920/AB_NYC_2019.JSON")

# COMMAND ----------

#Check the data
hotels.describe().show()

# COMMAND ----------

#Check the types per column
hotels.dtypes

# COMMAND ----------

# DBTITLE 1,Question 1B: Read the Data Using A Manual Schema
#Read in the data with a custom schema
from pyspark.sql.types import *

hotelsSchema = StructType([
  StructField("id", StringType(), False),
  StructField("name", StringType(), False),
  StructField("host_id", StringType(), False),
  StructField("host_name", StringType(), False),
  StructField("neighbourhood_group", StringType(), True),
  StructField("neighbourhood", StringType(), True),
  StructField("latitude", DoubleType(), True),
  StructField("longitude", DoubleType(), True),
  StructField("room_type", StringType(), True),
  StructField("price", FloatType(), True),
  StructField("minimum_nights", IntegerType(), True),
  StructField("number_of_reviews", IntegerType(), True),
  StructField("last_review", DateType(), True),
  StructField("reviews_per_month", FloatType(), True),
  StructField("calculated_host_listings_count", IntegerType(), True),
  StructField("availability_365", ShortType(), True)
])

hotels = spark.read.format("csv").option("header","true").schema(hotelsSchema).option("escape","\"")\
  .load("/FileStore/tables/Assignments_BDT2_1920/AB_NYC_2019.csv")
hotels.createOrReplaceTempView("hotels")
hotels.cache()

hotels.show(5,False)

# COMMAND ----------

hotels.dtypes

# COMMAND ----------

# DBTITLE 1,Question 2A: Explore the Summary Values


# COMMAND ----------

hotels.columns

# COMMAND ----------

#Define string columns
cols = hotels.columns
string_cols = cols[0:6]
string_cols.append(cols[8])
print(string_cols)

# COMMAND ----------

#Define numeric columns
num_cols = cols[6:8]
num_cols.extend(cols[9:16])
print(num_cols)

# COMMAND ----------

#Get the summary values of the string columns
hotels.describe(string_cols).show()

# COMMAND ----------

#Get the summary values of the numeric columns
hotels.describe(num_cols).show()

# COMMAND ----------

# DBTITLE 1,Question 3A: Avg and STD of price by room_type, neighborhood group, and neighborhood
#Using SQL
spark.sql("""
  select neighbourhood_group, neighbourhood, round(avg(price),2) as avg_price, round(std(price),2) as std_price 
  from hotels 
  where neighbourhood_group is not null 
  group by neighbourhood_group, neighbourhood 
  order by std_price
""").show(3)

# COMMAND ----------

#Using DF syntax
from pyspark.sql.functions import avg, round, col, stddev
hotels\
  .groupBy(col("neighbourhood_group"), col("neighbourhood"))\
  .agg(round(avg(col("price")),2).alias("avg_price"), round(stddev(col("price")),2).alias("std_price"))\
  .na.drop("all")\
  .orderBy("std_price")\
  .show(3)

# COMMAND ----------

#Using a DF grouping sets function
from pyspark.sql.functions import grouping_id

hotelsNoNull = hotels.na.drop("all")
rolledUpDF = hotelsNoNull\
  .rollup("neighbourhood_group", "neighbourhood")\
  .agg(grouping_id().alias("level"), round(avg(col("price")),2).alias("avg_price"), round(stddev(col("price")),2).alias("std_price"))\
  .orderBy("std_price")

rolledUpDF.show()

# COMMAND ----------

# DBTITLE 1,Question 4B: Handle missing values
fill_cols_vals = {"neighbourhood": "NA", "neighbourhood_group": "NA", "availability_365" : 0}
hotels_imp = hotels.na.fill(fill_cols_vals).orderBy("availability_365")
hotels_imp.show(2)

# COMMAND ----------

# DBTITLE 1,Question 5B: Make a top 3 ranking of the places where it's the cheapest to stay in Manhattan for 3 days.
spark.sql("""
  select id, room_type, price*3 as cost, reviews_per_month 
  from hotels 
  where neighbourhood_group='Manhattan' and reviews_per_month is not null and minimum_nights==1 
  order by cost asc, reviews_per_month desc limit 3
""").show()

# COMMAND ----------

from pyspark.sql.functions import asc, desc

filterManhattan = col("neighbourhood_group")=='Manhattan'
filterMinimumNights =  col("minimum_nights")==1

hotels\
  .na.drop("any").select(col("id"), col("room_type"),(col("price")*3).alias("cost"), col("reviews_per_month"))\
  .where(filterManhattan & filterMinimumNights)\
  .orderBy(col("cost").asc(), col("reviews_per_month").desc())\
  .limit(3)\
  .show()
