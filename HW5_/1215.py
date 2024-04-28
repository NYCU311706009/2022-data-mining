#!/usr/bin/env python
# coding: utf-8

# In[1]:


sc.master


# In[2]:


global Path
if sc.master[0:5]=="local" :
    Path="file:/home/hduser/Documents/"
else:   
    Path="hdfs://master:9000/user/hduser/"


# In[3]:


import pyspark
from pyspark.sql import SparkSession


# In[4]:


sqlContext = SparkSession.builder.getOrCreate()


# In[5]:


data = sqlContext.read.format("csv").option("header", "true").load(Path+"movieRating.csv")


# In[6]:


data.select("TrainDataID", "UserID", "MovieID", "Rating").show(10)


# In[7]:


from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.evaluation import RegressionMetrics


# In[8]:


rates_data = data.rdd.map(lambda x: Rating (int(x[1]), int(x[2]), float(x[3])))


# In[9]:


# rates_data = rates_data.toDF().dropna().rdd


# In[10]:


(train, test) = rates_data.randomSplit([0.8, 0.2])


# In[11]:


model = ALS.train(train, 20, nonnegative=False)


# In[12]:


test_x = test.map(lambda x: (x[0], x[1]))
test_y = test.map(lambda x: x[2])


# In[13]:


result = model.predictAll(test_x)


# In[14]:


result = result.map(lambda r: ((r.user, r.product), r.rating))
ratesAndPreds = test.map(lambda r: ((r.user, r.product), r.rating)).join(result)
predictAndTrue = ratesAndPreds.map(lambda r: r[1])


# In[15]:


regressionMetrics = RegressionMetrics(predictAndTrue)


# In[16]:


print(regressionMetrics.meanAbsoluteError)

