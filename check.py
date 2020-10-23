# Retrieve embeddings stores as string using PySpark UDF leveraging json library
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
import json

def parse_embedding_from_string(x):
    res = json.loads(x)
    return res

appName = "Vector Search Pyspark"
master = 'local'


# Create Spark session
spark = SparkSession.builder \
    .master(master) \
    .appName(appName) \
    .getOrCreate()
    
    
retrieve_embedding = F.udf(parse_embedding_from_string, T.ArrayType(T.DoubleType()))


df = spark.createDataFrame(data=[['[1.4, 2.256, 2.987]'], ['[45.56, 23.564, 2.987]'], ['[343.0, 1.23, 9.01]'],  ['[5.4, 3.1, -1.23]'], ['[6.54, -89.1, 3.1]'], ['[4.0, 1.0, -0.56]'], ['[1.0, 4.5, 6.7]'], ['[45.4, 3.45, -0.98]']], schema=["embedding"])

df = df.withColumn("embedding_new", retrieve_embedding(F.col("embedding")))

df.printSchema()
df.show(truncate=False)

value = df.select('embedding_new').collect()[0][0]
print(value)
print("---------------------------------------------------------")
