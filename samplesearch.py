import numpy as np
import pandas as pd
from pyspark.sql.functions import udf

from pyspark.sql.types import StringType, FloatType
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

column=[]
num_rows = 10000 #change to 2000000 to really slow your computer down!
for x in range(num_rows):
    sample = np.random.uniform(low=-1, high=1, size=(300,)).tolist()
    column.append(sample)
index = [i for i in range(10000)]
df_pd = pd.DataFrame([index, column]).T
df_pd.head()
df = spark.createDataFrame(df_pd).withColumnRenamed('0', 'Index').withColumnRenamed('1', 'Vectors')
df.show()

## Query Vector
new_input = np.random.uniform(low=-1, high=1, size=(300,)).tolist()
df_pd_new = pd.DataFrame([[new_input]])
df_new = spark.createDataFrame(df_pd_new, ['Input_Vector'])
df_new.show()



## Cosine similarity
value = df_new.select('Input_Vector').collect()[0][0]
print(value)
print("---------------------------------------------------------")
def cos_sim(vec):
    if (np.linalg.norm(value) * np.linalg.norm(vec)) != 0:
        dot_value = np.dot(value, vec) / (np.linalg.norm(value)*np.linalg.norm(vec))
        return dot_value.tolist()
    
cos_sim_udf = udf(cos_sim, FloatType())

df_cos = df.withColumn('cos_dis', cos_sim_udf('Vectors')).dropna(subset='cos_dis')
df_cos.show()
print("---------------------------------------------------------")
max_values = df_cos.select('index','cos_dis').orderBy('cos_dis', ascending=False).limit(5).collect()
print(max_values)
top_indicies = []
for x in max_values:
    top_indicies.append(x[0])
print("---------------------------------------------------------")
print (top_indicies)



sc.stop()
