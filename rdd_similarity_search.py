
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.context import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType, FloatType, StructField, StructType, IntegerType



import time
import numpy as np
import json

import config



appName = "Vector Search Pyspark"
master = 'local'



def parse_embedding_from_string(x):
    res = json.loads(x)
    return res

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

# Create Spark session
spark = SparkSession.builder \
    .master(master) \
    .appName(appName) \
    .getOrCreate()
    
retrieve_embedding = F.udf(parse_embedding_from_string, T.ArrayType(T.DoubleType()))

# Feature vector to data frame
feature_df = spark.read.format('csv') \
                .option('header',True) \
                .option('multiLine', True) \
                .load(config.HDFS_DATA_PATH)


# Query vector
query_df = spark.read.format('csv') \
                .option('header',True) \
                .option('multiLine', True) \
                .load(config.HDFS_QUERY_PATH)
               
                             
feature_df.show()
feature_df.printSchema()
query_df.show()
query_df.printSchema()


feature_df = feature_df.withColumn("features_new", retrieve_embedding(F.col("features")))
feature_df.printSchema()
query_df = query_df.withColumn("features_new", retrieve_embedding(F.col("features")))
query_df.printSchema()

value = query_df.select('features_new').collect()[0][0]
print("--------------------*****----------------------------------")

def cos_sim(vec):
    if (np.linalg.norm(value) * np.linalg.norm(vec)) != 0:
        dot_value = np.dot(value, vec) / (np.linalg.norm(value)*np.linalg.norm(vec))
        return dot_value.tolist()

# cos_sim_udf = F.udf(cos_sim, T.FloatType())
spark.udf.register("cos_sim", cos_sim, FloatType())
                   
start = time.time()
feature_df_rdd = feature_df.rdd

# feature_df_rdd_new = feature_df_rdd.map(lambda x: x + cos_sim_udf(x[2]))

get_schema = StructType(
[StructField('label', StringType(), True),
 StructField('cosine_distance', FloatType(), True),
 StructField('image_path', StringType(), True)]
)

feature_df_rdd_new  = feature_df_rdd.map(lambda x: (x[1], cos_sim(x[4]), x[3])).toDF(get_schema) #cos_sim_udf
top_match = feature_df_rdd_new.rdd.max(key=lambda x: x["cosine_distance"])
print(top_match)

print(top_match.__getattr__("image_path"))
# print(max_values)

# print(list(top_match.asDict()))
# print(type(top_match))
# end = time.time()

# print("Top matches are {}".format(top_matches))
# print("Total time is {}".format(end-start))
# print("---------------------------------------------------------")




# max_values = feature_df_cos.select('image_paths','cos_dis').orderBy('cos_dis', ascending=False).limit(5).collect()
# top_matches = []
# for x in max_values:
#     top_matches.append(x[0])

# end = time.time()

# print("Top matches are {}".format(top_matches))
# print("Total time is {}".format(end-start))
# print("---------------------------------------------------------")

