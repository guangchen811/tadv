import pydeequ
from pyspark.sql import Row, SparkSession
import numpy as np


def init_files():
    spark = (
        SparkSession.builder.config("spark.jars.packages", pydeequ.deequ_maven_coord)
        .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
        .getOrCreate()
    )

    df = spark.sparkContext.parallelize(
        [Row(a="foo", b=1, c=5), Row(a="bar", b=2, c=6), Row(a="baz", b=3, c=None)]
    ).toDF()
    return spark, df


def spark_df_from_pandas_df(pandas_df):
    spark = (
        SparkSession.builder
        .config("spark.jars.ivy.log", "none")
        .config("spark.hadoop.native.lib", "false")
        .config("spark.jars.packages", pydeequ.deequ_maven_coord)
        .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )
    # spark.sparkContext.setLogLevel("ERROR")
    pandas_df = pandas_df.where(pandas_df.notna(), None)
    pandas_df = pandas_df.replace(np.nan, None)
    spark_df = spark.createDataFrame(pandas_df)
    return spark_df, spark
