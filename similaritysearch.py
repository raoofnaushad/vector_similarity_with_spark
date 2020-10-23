
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F

import numpy as np
import json

import config

appName = "Vector Search Pyspark"
master = 'local'



def parse_embedding_from_string(x):
    res = json.loads(x)
    return res

def cos_sim(vec):
    if (np.linalg.norm(value) * np.linalg.norm(vec)) != 0:
        dot_value = np.dot(value, vec) / (np.linalg.norm(value)*np.linalg.norm(vec))
        return dot_value.tolist()


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
print("Query vector is {}".format(value))
print("--------------------*****----------------------------------")


cos_sim_udf = F.udf(cos_sim, T.FloatType())

feature_df_cos = query_df.withColumn('cos_dis', cos_sim_udf('features_new')).dropna(subset='cos_dis')
max_values = feature_df_cos.select('image_paths','cos_dis').orderBy('cos_dis', ascending=False).limit(5).collect()
top_matches = []
for x in max_values:
    top_matches.append(x[0])
    
print("Top matches are {}".format(top_matches))
print("---------------------------------------------------------")

