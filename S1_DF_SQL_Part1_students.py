# Databricks notebook source
# DBTITLE 1,Session 1: Working with the SQL + DataFrame Library of Spark
#This notebook will introduce you to:
#1. Working with DataFrames
#2. Working with the DataFrame API syntax and SQL API syntax

# COMMAND ----------

# DBTITLE 1,Reading Data
#Read in data Flights
######################

#Step 1: Upload the dataset to Databricks (see slide on "Uploading Data"). Copy/paste the path to the file.

#Step 2: Run the following code. Don't forget to set up the path!
flights=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/flights_2015.csv")

# COMMAND ----------

#Check the DF's Schema
flights.printSchema()

# COMMAND ----------

# DBTITLE 1,Checking Data
#Check the first few rows.
flights.head(3)

# COMMAND ----------

#Check the first few rows.
flights.take(3)

# COMMAND ----------

#Check the first few rows.
flights.show(3)
flights.show(3, truncate=False) #Do not truncate, display the results in full length.

# COMMAND ----------

#Check the column names of the DF.
flights.columns

# COMMAND ----------

#Check a summary of the values of the columns in the DF.
flights.describe().show()
#describe returns the min, max, mean, std for numeric values; first observation and last observation (alphabetically), and count for string values

# COMMAND ----------

# DBTITLE 1,Selecting Data
#In this Section, we will introduce the DF API and SQL API simultaneously.

#The typical structure of a SELECT statement in SQL is as follows:
"""

SELECT
FROM
WHERE
  
GROUP BY
HAVING
  
ORDER BY

"""
#The DF API has been designed to mimic this SQL pattern.

# COMMAND ----------

#Create a table or temp view to access the SQL queries on the flights table.
flights.createOrReplaceTempView("flights")
#flights.createOrReplaceGlobalTempView("flights")

# COMMAND ----------

#Select a list of destination countries.
#Using Spark SQL API
spark.sql("select dest_country_name from flights").show(3)

#Using Spark DF API
flights.select("dest_country_name").show(3)

# COMMAND ----------

#Alternatives using Spark DF API
from pyspark.sql.functions import col, column, expr

flights.select("dest_country_name").show(3)

flights.select(col("dest_country_name")).show(3)
flights.select(column("dest_country_name")).show(3)

flights.select(expr("dest_country_name")).show(3)
flights.selectExpr("dest_country_name").show(3)

# COMMAND ----------

#Count the number of unique destination countries.
spark.sql("select count(distinct dest_country_name) from flights").show()

flights.select("dest_country_name").distinct().count()

# COMMAND ----------

#Adding a constant value (literal)
spark.sql("select *,1 from flights").show(2)

from pyspark.sql.functions import lit
flights.select(expr("*"),lit(1)).show(2)
flights.withColumn("1",lit(1)).show(2)

# COMMAND ----------

# DBTITLE 1,Renaming columns
#Change column names using SQL
spark.sql("select dest_country_name as destination from flights").show(2)

#Change column names using DF
flights.select(expr("dest_country_name as destination")).show(2)
flights.selectExpr("dest_country_name as destination").show(2)

flights.select(col("dest_country_name").alias("destination")).show(2)

flights.withColumnRenamed("dest_country_name","destination").show(2)

# COMMAND ----------

#Clean up the previous table with prettier formating: rounding numbers and naming columns
spark.sql("select round(avg(count),2) as avg_flights, count(distinct(dest_country_name)) as cnt_dests from flights").show()

from pyspark.sql.functions import avg, round
flights.selectExpr("round(avg(count),2) as avg_flights","count(distinct(dest_country_name)) as cnt_dests").show()
flights.select(round(avg("count"),2).alias("avg_flights"),expr("count(distinct(dest_country_name)) as cnt_dests")).show()

# COMMAND ----------

# DBTITLE 1,Dropping columns
#Delete column dest_country_name
flights.show(1)

#Delete column dest_country_name
df_drop = flights.drop("dest_country_name")
df_drop.show(1)
df_drop.columns

# COMMAND ----------

# DBTITLE 1,Subsetting Data (WHERE statement)
#The typical structure of a SELECT statement in SQL is as follows:
"""

SELECT
FROM
#############    WHERE
  
GROUP BY
HAVING
  
ORDER BY

"""

# COMMAND ----------

#Count the number of unique countries with less than 2 flights per day
spark.sql("select count(distinct dest_country_name) from flights where count<2").show()
flights.where(col("count")<2).select("dest_country_name").distinct().count()

# COMMAND ----------

#The order in which you define the statements, can have a different outcome on the results.
flights.select("dest_country_name").distinct().where(col("count")<2).count()

# COMMAND ----------

flights.where(col("count")<2).select("dest_country_name").distinct().count()
#This output is the same as the SQL output because SQL first applies the WHERE condition, then group by statement, then having statement, then order by, and then select.

#Therefore, to assure the same behavior as SQL, define the where() statement first and then define the select() statement

# COMMAND ----------

#Select all distination countries for flights leaving from Greece
from pyspark.sql.functions import col

spark.sql("select dest_country_name from flights where origin_country_name = 'Greece'").show()
flights.where(col("origin_country_name") == "Greece").select("dest_country_name").show()

#Count the number of distination countries for flights NOT originating from Croatia
spark.sql("select count(distinct dest_country_name) from flights where origin_country_name!='Croatia'").show()
flights.where(col("origin_country_name") != "Croatia").select("dest_country_name").distinct().count()

# COMMAND ----------

#Select all destination countries with less than 2 flights per day from the United States
spark.sql("select distinct dest_country_name from flights where origin_country_name='United States' and count<2").show()
flights.where((col("origin_country_name") == "United States") & (col("count")<2)).select("dest_country_name").distinct().show()

# COMMAND ----------

#We can combine multiple "AND" conditions using pipelined where() statements
flights.where((col("origin_country_name") == "United States") & (col("count")<2)).select("dest_country_name").distinct().show()
flights.where(col("origin_country_name") == "United States").where(col("count")<2).select("dest_country_name").distinct().show()

#Using expressions
flights.where(col("origin_country_name") == "United States").where("count<2").select("dest_country_name").distinct().show()
flights.where("origin_country_name='United States' and count<2").select("dest_country_name").distinct().show()

# COMMAND ----------

#Simplify even further
countryFilter = col("origin_country_name") == "United States"
countFilter = col("count")<2

flights.where(countryFilter & countFilter).select("dest_country_name").show()

# COMMAND ----------

#Exercise: Count the number of destination countries from a US or Greece airport

# COMMAND ----------

#Exercise: Select all countries with in-land flights

# COMMAND ----------

# DBTITLE 1,SQL Subquerries


# COMMAND ----------

#Uncorrelated subquery

#Select the origin and destination country of the flight with the most flights per day
spark.sql("""
 select * from flights 
 where count = (select max(count) from flights)
""").show()

#The subquery can be run separately
spark.sql("""
 select max(count) from flights
""").show()

# COMMAND ----------

#Correlated subquery

#Select all return flights (= flights that both function as a destination and origin country)
spark.sql("""
  SELECT * FROM flights f1
  WHERE EXISTS (SELECT 1 FROM flights f2
            WHERE f1.dest_country_name = f2.origin_country_name)
  AND EXISTS (SELECT 1 FROM flights f2
            WHERE f2.dest_country_name = f1.origin_country_name)
""").show()

# COMMAND ----------

# DBTITLE 1,Aggregating Data


# COMMAND ----------

#Read other dataset
df = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("/FileStore/tables/online_retail_dataset.csv")\
  .coalesce(5)

df.cache()
df.show(3)

df.createOrReplaceTempView("df")

# COMMAND ----------

#Count observations (1)
df.count()

# COMMAND ----------

#Count observations (2)
from pyspark.sql.functions import count
df.select(count("*")).show()
df.select(count("StockCode")).show()

# COMMAND ----------

#Count distinct (exact)
from pyspark.sql.functions import countDistinct
df.select(count("StockCode"),countDistinct("StockCode")).show()

# COMMAND ----------

#Count distinct (approximated)
from pyspark.sql.functions import approx_count_distinct
df.select(approx_count_distinct("StockCode", 0.1)).show()

#the second value specifies the error that is allowed; the lower the value, the lower the error is allowed to be.
df.select(approx_count_distinct("StockCode", 0.01)).show()

# COMMAND ----------

#Exercise: Replicate the previous results using SQL code.

# COMMAND ----------

#Sum a column
from pyspark.sql.functions import sum, sumDistinct
df.select(sum("Quantity"),sumDistinct("Quantity")).show()

spark.sql("select sum(quantity), sum(distinct quantity) from df").show()

# COMMAND ----------

#Get first and last row value of a column
from pyspark.sql.functions import first, last
df.select(first("StockCode"), last("StockCode")).show()

# COMMAND ----------

#Cross table
person = spark.createDataFrame([
    (0, "Bill Chambers", 0, [100]),
    (1, "Matei Zaharia", 1, [500, 250, 100]),
    (2, "Michael Armbrust", 1, [250, 100])])\
  .toDF("id", "name", "graduate_program", "spark_status")

person.stat.crosstab("id", "name").show()

# COMMAND ----------

#Get an overview of the summary statistics
df.describe().show()

#Get specific summary statistics of numeric variables
df.select("quantity").summary("min","max").show()

#Alternative
from pyspark.sql.functions import min, max
df.select(min("quantity"), max("quantity")).show()

#Get a specific statistic
spark.sql("select max(quantity) from df").show()

from pyspark.sql.functions import max
df.select(max("quantity")).show()


# COMMAND ----------

#Calculate the mean quantity sold and the number of customers
spark.sql("select avg(quantity), count(distinct(customerid)) from df").show()

from pyspark.sql.functions import avg, expr
df.select(avg("quantity"),expr("count(distinct(customerid))")).show()
df.selectExpr("avg(quantity)","count(distinct(customerid))").show()
df.selectExpr("mean(quantity)","count(distinct(customerid))").show()

#Create new column names for the calculated fields
df.select(round(avg("quantity"),0).alias("avg_quantity"),expr("count(distinct(customerid)) as n_custs")).show()

# COMMAND ----------

#Exercise: Calculate the variance and SD of UnitPrice to check the spread around the mean

# COMMAND ----------

#Calculate Pearson correlation coefficient
from pyspark.sql.functions import corr
df.select(corr("UnitPrice", "Quantity")).show()
df.stat.corr("Quantity", "UnitPrice")

# COMMAND ----------

#There are plenty of functions in the stats package
df.stat.approxQuantile("UnitPrice", [0.50], 0.05)

# COMMAND ----------

#More complex calculations on numeric values
spark.sql("select stddev(5*power(quantity,3)+1-6) from df").show()

from pyspark.sql.functions import stddev,col,round
df.selectExpr("stddev(5*pow(quantity,3)+1-6)").show()
df.select(stddev(5*pow(col("quantity"),3)+1-6)).show()

#Official source of all pyspark functions where you easily find different functions:
#https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions

# COMMAND ----------

# DBTITLE 1,Grouping Data
#Here we discuss grouping numerical data by a key

#The typical structure of a SELECT statement in SQL is as follows:
"""

SELECT
FROM
WHERE
  
#############    GROUP BY
#############    HAVING
  
ORDER BY

"""

# COMMAND ----------

#Count the number of invoices by customer
spark.sql("select CustomerID, count(InvoiceNo) from df group by CustomerID").show(3)
df.groupBy("CustomerID").agg(count("InvoiceNo")).show(3)
df.groupBy("CustomerID").agg(expr("count(InvoiceNo)")).show(3)

# COMMAND ----------

#Get the average and standard deviation of UnitPrice and the total quantity per customer
df.groupBy("CustomerID").agg(avg(col("UnitPrice")),stddev(col("UnitPrice")),sum(col("Quantity"))).show(3)
df.groupBy("CustomerID").agg(expr("mean(UnitPrice)"),expr("stddev(UnitPrice)"),expr("sum(Quantity)")).show(3)

spark.sql("select CustomerID, avg(UnitPrice), std(UnitPrice), sum(Quantity) from df group by CustomerID").show(3)

# COMMAND ----------

#Calculate the average quantity per customer of customers who bought at least 1000 items.
spark.sql("select CustomerID, mean(quantity) from df group by CustomerID having sum(quantity)>1000").show(3)
#spark.sql("select CustomerID, mean(quantity) from df group by CustomerID where sum(quantity)>1000").show(3) #doesn't work in SQL since you cannot use an aggregated variable in a where() statement

from pyspark.sql.functions import avg
df.groupBy("CustomerID").agg(avg(col("quantity"))).where("sum(quantity)>1000").show(3)
df.groupBy("CustomerID").avg("quantity").where("sum(quantity)>1000").show(3)

df.groupBy("CustomerID").avg("quantity").show(3) #different result

# COMMAND ----------

#Calculate the number of items sold per invoice

# COMMAND ----------

#We can also group data on multiple columns using Grouping Sets:
################################################################
################################################################

#Rollup
#Cube
#Pivot

# COMMAND ----------

#Rollup: creates a new DataFrame that calculates per unique combination the grand total overall (where Date=null), per country (where Country=null), and per Date,Country (value of total_quantity)
from pyspark.sql.functions import col, to_date, sum
dfNoNull = retail.withColumn("Date", to_date(col("InvoiceDate"), "MM/d/yyyy H:mm")).na.drop("all")

rolledUpDF = dfNoNull.rollup("Date", "Country").agg(sum("Quantity"))\
  .selectExpr("Date", "Country", "`sum(Quantity)` as total_quantity")\
  .orderBy("Date")
rolledUpDF.show()

# COMMAND ----------

#We can verify the output
rolledUpDF.where("Country IS NULL").show() #shows total quantity per date
rolledUpDF.where("Date IS NULL").show() #shows overall total quantity

# COMMAND ----------

#Cube: Applies the grouping across all dimensions: total across all dates and countries, total for each date across all countries, total for each country on each date, total for each country across all dates
#Easy to create a summary table
from pyspark.sql.functions import sum

cubeDF=dfNoNull.cube("Date", "Country").sum("Quantity")\
  .select("Date", "Country", "sum(Quantity)")\
  .orderBy("Date")\
  .show()

# COMMAND ----------

#Denote the group level of aggregation using grouping_id()
from pyspark.sql.functions import grouping_id, sum, desc

#Cube
cubeGroupDF=dfNoNull.cube("CustomerID", "StockCode").agg(grouping_id().alias("grouped"),sum(col("Quantity"))).orderBy(desc("grouped"))
cubeGroupDF.show(5)

#Rollup
rollGroupDF=dfNoNull.rollup("CustomerID", "StockCode").agg(grouping_id().alias("grouped"),sum(col("Quantity"))).orderBy(desc("grouped"))
rollGroupDF.show(5)

# COMMAND ----------

#Check the levels of the variable grouped.
cubeGroupDF.groupBy("grouped").count().show()
rollGroupDF.groupBy("grouped").count().show()

#grouped has 3 levels: 
#3. total quantity, regardless of CustomerID or StockCode
#2. total quantity per StockCode
#1. total quantity per CustomerID
#0. total quantity per CustomerID and StockCode

# COMMAND ----------

#Pivot data: creates a column for every combination of country, Quantity, and UnitPrice per date
dfWithDate = retail.withColumn("Date", to_date(col("InvoiceDate"), "MM/d/yyyy H:mm"))
pivoted = dfWithDate.groupBy("Date").pivot("Country").sum("Quantity","UnitPrice").orderBy("date")
pivoted.columns

# COMMAND ----------

#Get Data
pivoted.where("Date > '2011-12-05'").select("Date","`USA_sum(UnitPrice)`").show() #`` denotes a literal String

# COMMAND ----------

# DBTITLE 1,Ordering Data
#Here we discuss ordering and sorting data by one or more columns

#The typical structure of a SELECT statement in SQL is as follows:
"""

SELECT
FROM
WHERE
  
GROUP BY
HAVING
  
#############    ORDER BY

"""

# COMMAND ----------

#Sorting on 1 column
spark.sql("select * from df order by quantity").show(2)

df.sort("quantity").show(2)
df.orderBy("quantity").show(2)

# COMMAND ----------

#Sorting 1 column on ascending and descending
from pyspark.sql.functions import asc, desc

#Sort descending
spark.sql("select * from df order by quantity desc").show(2)
df.orderBy(col("quantity").desc()).show(2)
df.orderBy(desc("quantity")).show(2)

#Sort ascending
spark.sql("select * from df order by quantity asc").show(2)
df.orderBy(col("quantity").asc()).show(2)
df.orderBy(asc("quantity")).show(2)

# COMMAND ----------

#Sorting on multiple columns
spark.sql("select CustomerID, InvoiceDate, Quantity from df where CustomerID is not null order by CustomerID, InvoiceDate desc, Quantity desc").show(3)

df.where("CustomerID is not null")\
  .orderBy(col("CustomerID"),col("InvoiceDate").desc(),col("Quantity").desc())\
  .select(col("CustomerID"),col("InvoiceDate"),col("Quantity"))\
  .show(3)

# COMMAND ----------

# DBTITLE 0,Summarizing Example of SQL and DF API syntax
#Exercise

#Get the top 5 destination countries originating from the US using Spark SQL API syntax

#Get the top 5 destination countries originating from the US using Spark DF API syntax

# COMMAND ----------

# DBTITLE 1,Joining Data


# COMMAND ----------

#Create some datasets
person = spark.createDataFrame([
    (0, "Bill Chambers", 0, [100]),
    (1, "Matei Zaharia", 1, [500, 250, 100]),
    (2, "Michael Armbrust", 1, [250, 100])])\
  .toDF("id", "name", "graduate_program", "spark_status")

graduateProgram = spark.createDataFrame([
    (0, "Masters", "School of Information", "UC Berkeley"),
    (2, "Masters", "EECS", "UC Berkeley"),
    (1, "Ph.D.", "EECS", "UC Berkeley")])\
  .toDF("id", "degree", "department", "school")

sparkStatus = spark.createDataFrame([
    (500, "Vice President"),
    (250, "PMC Member"),
    (100, "Contributor")])\
  .toDF("id", "status")

#Allow access through SQL (transfers the tables from DF to SQL)
person.createOrReplaceTempView("person")
graduateProgram.createOrReplaceTempView("graduateProgram")
sparkStatus.createOrReplaceTempView("sparkStatus")

# COMMAND ----------

#Inner join person and graduateProgram
spark.sql("select * from person a join graduateProgram b on a.graduate_program=b.id").show(3)
#spark.sql("select * from person a inner join graduateProgram b on a.graduate_program=b.id").show(3)

joinExpression = person["graduate_program"] == graduateProgram['id']
joinType="inner"

person.join(graduateProgram,joinExpression,joinType).show(3) #for an inner join, we can leave out the joinType variable, like so: person.join(graduateProgram,joinExpression).show(3)

# COMMAND ----------

#Outer join person and graduateProgram

#Full outer join
joinType="outer"
#spark.sql("select * from person a full outer join graduateProgram b on a.graduate_program=b.id").show()

#Left outer join
joinType="left_outer"
#spark.sql("select * from person a left outer join graduateProgram b on a.graduate_program=b.id").show()

#Right outer join
joinType="right_outer"
#spark.sql("select * from person a right outer join graduateProgram b on a.graduate_program=b.id").show()

person.join(graduateProgram,joinExpression,joinType).show()

# COMMAND ----------

#Semi joins: compare values to see if a value exists in the 2nd DF. If it exists, keep value.
joinType="left_semi"

gradProgram2 = graduateProgram\
  .union(spark.createDataFrame([(0, "Masters", "Duplicated Row", "Duplicated School")]))

#Semi join using DF
gradProgram2.join(person,joinExpression,joinType).show()

#Semi join using SQL
gradProgram2.createOrReplaceTempView("gradProgram2")
spark.sql("select * from gradProgram2 a left semi join person b on a.id=b.graduate_program").show(3)

# COMMAND ----------

#Anti joins: compare values to see if a value exists in the 2nd DF. If it exists, do not keep the value.
joinType="left_anti"

gradProgram2.join(person,joinExpression,joinType).show()
spark.sql("select * from gradProgram2 a left anti join person b on a.id=b.graduate_program").show()

# COMMAND ----------

#Cross joins: join all rows of person and graduateProgram
joinType="cross"
graduateProgram.join(person,joinExpression,joinType).show()
spark.sql("select * from graduateProgram a cross join person b on b.graduate_program=a.id").show()

person.crossJoin(graduateProgram).show()
spark.sql("select * from person cross join graduateProgram").show()

# COMMAND ----------

#Force a broadcast join using broadcast()
from pyspark.sql.functions import broadcast
person.join(broadcast(graduateProgram),joinExpression,joinType).show(3)
