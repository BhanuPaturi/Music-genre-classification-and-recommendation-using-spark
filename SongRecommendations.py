# Python and pyspark modules required

import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.sql import functions as F

from pyspark.ml.feature import StringIndexer
from pyspark.sql.window import Window

from pyspark.ml.recommendation import ALS
# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

import numpy as np

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M




# -----------------------------------------------------------------------------

# Load sid_matches_manually_accepted.txt, sid_mismatches.txt

mismatches_schema = StructType([
  StructField("song_id", StringType(), True),
  StructField("song_artist", StringType(), True),
  StructField("song_title", StringType(), True),
  StructField("track_id", StringType(), True),
  StructField("track_artist", StringType(), True),
  StructField("track_title", StringType(), True)
])

with open("/scratch-network/courses/2022/DATA420-22S1/data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt", "r") as f:
  lines = f.readlines()
  sid_matches_manually_accepted = []
  for line in lines:
    if line.startswith("< ERROR: "):
      a = line[10:28]
      b = line[29:47]
      c, d = line[49:-1].split("  !=  ")
      e, f = c.split("  -  ")
      g, h = d.split("  -  ")
      sid_matches_manually_accepted.append((a, e, f, b, g, h))

matches_manually_accepted = spark.createDataFrame(sc.parallelize(sid_matches_manually_accepted, 8), schema=mismatches_schema)
matches_manually_accepted.cache()
matches_manually_accepted.show(10, 20)
# :<EOF>
# +------------------+-----------------+--------------------+------------------+--------------------+--------------------+|           song_id|      song_artist|          song_title|          track_id|        track_artist|         track_title|+------------------+-----------------+--------------------+------------------+--------------------+--------------------+|SOFQHZM12A8C142342|     Josipa Lisac|             razloga|TRMWMFG128F92FFEF2|        Lisac Josipa|        1000 razloga||SODXUTF12AB018A3DA|       Lutan Fyah|Nuh Matter the Cr...|TRMWPCD12903CCE5ED|             Midnite|Nah Matter the Cr...||SOASCRF12A8C1372E6|Gaetano Donizetti|L'Elisir d'Amore:...|TRMHIPJ128F426A2E2|Gianandrea Gavazz...|L'Elisir D'Amore_...||SOITDUN12A58A7AACA|     C.J. Chenier|           Ay, Ai Ai|TRMHXGK128F42446AB|     Clifton Chenier|           Ay_ Ai Ai||SOLZXUM12AB018BE39|           許志安|            男人最痛|TRMRSOF12903CCF516|            Andy Hui|    Nan Ren Zui Tong||SOTJTDT12A8C13A8A6|                S|                   h|TRMNKQE128F427C4D8|         Sammy Hagar|20th Century Man ...||SOGCVWB12AB0184CE2|                H|                   Y|TRMUNCZ128F932A95D|            Hawkwind|25 Years (Alterna...||SOKDKGD12AB0185E9C|     影山ヒロノブ|Cha-La Head-Cha-L...|TRMOOAH12903CB4B29|    Takahashi Hiroki|Maka fushigi adve...||SOPPBXP12A8C141194|    Αντώνης Ρέμος|    O Trellos - Live|TRMXJDS128F42AE7CF|       Antonis Remos|           O Trellos||SODQSLR12A8C133A01|    John Williams|Concerto No. 1 fo...|TRWHMXN128F426E03C|English Chamber O...|II. Andantino sic...|+------------------+-----------------+--------------------+------------------+--------------------+--------------------+only showing top 10 rows

# 488
# +------------------+-------------------+--------------------+------------------+--------------+--------------------+
# |           song_id|        song_artist|          song_title|          track_id|  track_artist|         track_title|
# +------------------+-------------------+--------------------+------------------+--------------+--------------------+
# |SOUMNSI12AB0182807|Digital Underground|    The Way We Swing|TRMMGKQ128F9325E10|      Linkwood|Whats up with the...|
# |SOCMRBE12AB018C546|         Jimmy Reed|The Sun Is Shinin...|TRMMREB12903CEB1B1|    Slim Harpo|I Got Love If You...|
# |SOLPHZY12AC468ABA8|      Africa HiTech|            Footstep|TRMMBOC12903CEB46E|Marcus Worgull|Drumstern (BONUS ...|
# |SONGHTM12A8C1374EF|     Death in Vegas|        Anita Berber|TRMMITP128F425D8D0|     Valen Hsu|              Shi Yi|
# |SONGXCA12A8C13E82E| Grupo Exterminador|       El Triunfador|TRMMAYZ128F429ECE6|     I Ribelli|           Lei M'Ama|
# |SOMBCRC12A67ADA435|      Fading Friend|         Get us out!|TRMMNVU128EF343EED|     Masterboy|  Feel The Heat 2000|
# |SOTDWDK12A8C13617B|       Daevid Allen|          Past Lives|TRMMNCZ128F426FF0E| Bhimsen Joshi|Raga - Shuddha Sa...|
# |SOEBURP12AB018C2FB|  Cristian Paduraru|          Born Again|TRMMPBS12903CE90E1|     Yespiring|      Journey Stages|
# |SOSRJHS12A6D4FDAA3|         Jeff Mills|  Basic Human Design|TRMWMEL128F421DA68|           M&T|       Drumsettester|
# |SOIYAAQ12A6D4F954A|           Excepter|                  OG|TRMWHRI128F147EA8E|    The Fevers|Não Tenho Nada (N...|
# +------------------+-------------------+--------------------+------------------+--------------+--------------------+
# only showing top 10 rows

print(matches_manually_accepted.count())
#488

with open("/scratch-network/courses/2022/DATA420-22S1/data/msd/tasteprofile/mismatches/sid_mismatches.txt", "r") as f:
  lines = f.readlines()
  sid_mismatches = []
  for line in lines:
    if line.startswith("ERROR: "):
      a = line[8:26]
      b = line[27:45]
      c, d = line[47:-1].split("  !=  ")
      e, f = c.split("  -  ")
      g, h = d.split("  -  ")
      sid_mismatches.append((a, e, f, b, g, h))

mismatches = spark.createDataFrame(sc.parallelize(sid_mismatches, 64), schema=mismatches_schema)
mismatches.cache()
mismatches.show(10, 20)

# 19094
# 2022-06-03 09:19:08,478 WARN execution.CacheManager: Asked to cache already cached data.
# +----------------------------------------+------------------+-----+
# |                                 user_id|           song_id|plays|
# +----------------------------------------+------------------+-----+
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQEFDN12AB017C52B|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOIUJ12A6701DAA7|    2|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOKKD12A6701F92E|    4|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSDVHO12AB01882C7|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSKICX12A6701F932|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSNUPV12A8C13939B|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSVMII12A6701F92D|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTUNHI12B0B80AFE2|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTXLTZ12AB017C535|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTZDDX12A6701F935|    1|
# +----------------------------------------+------------------+-----+
# only showing top 10 rows

print(mismatches.count())

# 19094

##Load the triplets dataset


triplets_schema = StructType([
  StructField("user_id", StringType(), True),
  StructField("song_id", StringType(), True),
  StructField("plays", IntegerType(), True)
])
triplets = (
  spark.read.format("csv")
  .option("header", "false")
  .option("delimiter", "\t")
  .option("codec", "gzip")
  .schema(triplets_schema)
  .load("hdfs:///data/msd/tasteprofile/triplets.tsv/")
  .cache()
)
triplets.cache()
triplets.show(10, 50)
# 
# 2022-05-29 17:44:33,538 WARN execution.CacheManager: Asked to cache already cached data.
# +----------------------------------------+------------------+-----+
# |                                 user_id|           song_id|plays|
# +----------------------------------------+------------------+-----+
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQEFDN12AB017C52B|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOIUJ12A6701DAA7|    2|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOKKD12A6701F92E|    4|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSDVHO12AB01882C7|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSKICX12A6701F932|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSNUPV12A8C13939B|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSVMII12A6701F92D|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTUNHI12B0B80AFE2|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTXLTZ12AB017C535|    1|
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTZDDX12A6701F935|    1|
# +----------------------------------------+------------------+-----+
# only showing top 10 rows

mismatches_not_accepted = mismatches.join(matches_manually_accepted, on="song_id", how="left_anti")
triplets_not_mismatched = triplets.join(mismatches_not_accepted, on="song_id", how="left_anti")

triplets_not_mismatched = triplets_not_mismatched.repartition(partitions).cache()

print(mismatches_not_accepted.count())
print(triplets.count())
print(triplets_not_mismatched.count())

# 19093
# 48373586
# 45795111

# -----------------------------------------------------------------------------
## Q1 
# Data analysis

def get_user_counts(triplets):
  return (
    triplets
    .groupBy("user_id")
    .agg(
      F.count(F.col("song_id")).alias("song_count"),
      F.sum(F.col("plays")).alias("play_count"),
    )
    .orderBy(F.col("play_count").desc())
  )

def get_song_counts(triplets):
  return (
    triplets
    .groupBy("song_id")
    .agg(
      F.count(F.col("user_id")).alias("user_count"),
      F.sum(F.col("plays")).alias("play_count"),
    )
    .orderBy(F.col("play_count").desc())
  )

# User statistics
## Q1 b) Most active User  
user_counts = (
  triplets_not_mismatched
  .groupBy("user_id")
  .agg(
    F.count(F.col("song_id")).alias("song_count"),
    F.sum(F.col("plays")).alias("play_count"),
  )
  .orderBy(F.col("play_count").desc())
)
user_counts.cache()
user_counts.count()
user_counts.show(10, False)

# +----------------------------------------+----------+----------+
# |user_id                                 |song_count|play_count|
# +----------------------------------------+----------+----------+
# |093cb74eb3c517c5179ae24caf0ebec51b24d2a2|195       |13074     |
# |119b7c88d58d0c6eb051365c103da5caf817bea6|1362      |9104      |
# |3fa44653315697f42410a30cb766a4eb102080bb|146       |8025      |
# |a2679496cd0af9779a92a13ff7c6af5c81ea8c7b|518       |6506      |
# |d7d2d888ae04d16e994d6964214a1de81392ee04|1257      |6190      |
# |4ae01afa8f2430ea0704d502bc7b57fb52164882|453       |6153      |
# |b7c24f770be6b802805ac0e2106624a517643c17|1364      |5827      |
# |113255a012b2affeab62607563d03fbdf31b08e7|1096      |5471      |
# |99ac3d883681e21ea68071019dba828ce76fe94d|939       |5385      |
# |6d625c6557df84b60d90426c0116138b617b9449|1307      |5362      |
# +----------------------------------------+----------+----------+
# only showing top 10 rows


##Q1 c) User Activity
userActivity = (
    user_counts
    .select('user_id', 'song_count')
)
userActivity.write.csv('hdfs:///user/bpa78/outputs/msd/userActivity', mode = 'overwrite', header = True)
#userActivity = spark.read.load('hdfs:///user/bpa78/outputs/msd/userActivity', format = "com.databricks.spark.csv", header = True, inferSchema=True)

statistics = (
  user_counts
  .select("song_count", "play_count")
  .describe()
  .toPandas()
  .set_index("summary")
  .rename_axis(None)
  .T
)
print(statistics)


              # count                mean              stddev min    max
# song_count  1019318   44.92720721109605    54.9111319974736   3   4316
# play_count  1019318  128.82423149596102  175.43956510304736   3  13074

user_counts.approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
#Out[12]: [3.0, 15.0, 25.0, 49.0, 4316.0]

user_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
#Out[13]: [3.0, 28.0, 67.0, 135.0, 13074.0]



# Song statistics
## Q1 a) Song count
song_counts = (
  triplets_not_mismatched
  .groupBy("song_id")
  .agg(
    F.count(F.col("user_id")).alias("user_count"),
    F.sum(F.col("plays")).alias("play_count"),
  )
  .orderBy(F.col("play_count").desc())
)
song_counts.cache()
song_counts.count()
song_counts.show(10, False)

# 2022-06-07 21:16:44,969 WARN execution.CacheManager: Asked to cache already cached data.
# +------------------+----------+----------+
# |song_id           |user_count|play_count|
# +------------------+----------+----------+
# |SOBONKR12A58A7A7E0|84000     |726885    |
# |SOSXLTC12AF72A7F54|80656     |527893    |
# |SOEGIYH12A6D4FC0E3|69487     |389880    |
# |SOAXGDH12A8C13F8A1|90444     |356533    |
# |SONYKOW12AB01849C9|78353     |292642    |
# |SOPUCYA12A8C13A694|46078     |274627    |
# |SOUFTBI12AB0183F65|37642     |268353    |
# |SOVDSJC12A58A7A271|36976     |244730    |
# |SOOFYTN12A6D4F9B35|40403     |241669    |
# |SOHTKMO12AB01843B0|46077     |236494    |
# +------------------+----------+----------+
##Q1 c) Song popularity

songPopularity = (
    song_counts
    .select('song_id', 'play_count')
)

songPopularity.write.csv('hdfs:///user/bpa78/outputs/msd/songPopularity', mode = 'overwrite', header = True)
#songPopularity = spark.read.load('hdfs:///user/bpa78/outputs/msd/songPopularity', format = "com.databricks.spark.csv", header = True, inferSchema=True)
statistics = (
  song_counts
  .select("user_count", "play_count")
  .describe()
  .toPandas()
  .set_index("summary")
  .rename_axis(None)
  .T
)
print(statistics)

             # count                mean             stddev min     max
# user_count  378310  121.05181200602681  748.6489783736946   1   90444
# play_count  378310   347.1038513388491  2978.605348838228   1  726885


song_counts.approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
#Out[30]: [1.0, 4.0, 12.0, 51.0, 90444.0] 

song_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
# Out[3]: [1.0, 7.0, 27.0, 131.0, 726885.0]




# -----------------------------------------------------------------------------

# Q1 d) Limiting Filtering the users and songs based on the threshold

user_song_count_threshold = 34
song_user_count_threshold = 5

triplets_limited = triplets_not_mismatched

triplets_limited = (
  triplets_limited
  .join(
    triplets_limited.groupBy("user_id").count().where(F.col("count") > song_user_count_threshold).select("user_id"),
    on="user_id",
    how="inner"
  )
)

triplets_limited = (
  triplets_limited
  .join(
    triplets_limited.groupBy("song_id").count().where(F.col("count") > user_song_count_threshold).select("song_id"),
    on="song_id",
    how="inner"
  )
)

triplets_limited.cache()
triplets_limited.count()
#Out[29]: 43270167

(
  triplets_limited
  .agg(
    F.countDistinct(F.col("user_id")).alias('user_count'),
    F.countDistinct(F.col("song_id")).alias('song_count')
  )
  .toPandas()
  .T
  .rename(columns={0: "value"})
)

         

              # value
# user_count  1019193
# song_count   116791

print(get_user_counts(triplets_limited).approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))

print(get_song_counts(triplets_limited).approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))



# :<EOF>
# [1.0, 14.0, 24.0, 46.0, 3071.0]
# [35.0, 53.0, 106.0, 305.0, 90444.0]
# -----------------------------------------------------------------------------

# Encoding converting the user_id and song_id

user_id_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_encoded")
song_id_indexer = StringIndexer(inputCol="song_id", outputCol="song_id_encoded")

user_id_indexer_model = user_id_indexer.fit(triplets_limited)
song_id_indexer_model = song_id_indexer.fit(triplets_limited)

triplets_limited = user_id_indexer_model.transform(triplets_limited)
triplets_limited = song_id_indexer_model.transform(triplets_limited)

triplets_limited.printSchema()
# root
 # |-- song_id: string (nullable = true)
 # |-- user_id: string (nullable = true)
 # |-- plays: integer (nullable = true)
 # |-- user_id_encoded: double (nullable = false)
 # |-- song_id_encoded: double (nullable = false)


# -----------------------------------------------------------------------------

# Splitting

training, test = triplets_limited.randomSplit([0.75, 0.25])

test_not_training = test.join(training, on="user_id", how="left_anti")

training.cache()
test.cache()
test_not_training.cache()

print(f"training:      {training.count()}") #training:      32451690
print(f"test:        {test.count()}") #test:        10818477
print(f"test_not_training: {test_not_training.count()}") #test_not_training: 128

test_not_training.show(50, False)


counts = test_not_training.groupBy("user_id").count().toPandas().set_index("user_id")["count"].to_dict()

temp = (
  test_not_training
  .withColumn("id", F.monotonically_increasing_id())
  .withColumn("random", F.rand())
  .withColumn(
    "row",
    F.row_number()
    .over(
      Window
      .partitionBy("user_id")
      .orderBy("random")
    )
  )
)

for k, v in counts.items():
  temp = temp.where((F.col("user_id") != k) | (F.col("row") < v * 0.75))

temp = temp.drop("id", "random", "row")
temp.cache()

temp.show(50, False)

training = training.union(temp.select(training.columns))
test = test.join(temp, on=["user_id", "song_id"], how="left_anti")
test_not_training = test.join(training, on="user_id", how="left_anti")

print(f"training:      {training.count()}")#training:       32451735
print(f"test:        {test.count()}")#test:        10818432
print(f"test_not_training: {test_not_training.count()}")#test_not_training: 44


#Q2 a) Training an inplicit matrix factorization model using Alternating Least Squares(ALS) model
als = ALS(maxIter=5, regParam=0.01, userCol="user_id_encoded", itemCol="song_id_encoded", ratingCol="plays", implicitPrefs=True)
alsModel = als.fit(training)
predictions = alsModel.transform(test)

predictions.cache()
predictions.count()
predictions.show(10, 50)

# +----------------------------------------+------------------+-----+---------------+---------------+-----------+
# |                                 user_id|           song_id|plays|user_id_encoded|song_id_encoded| prediction|
# +----------------------------------------+------------------+-----+---------------+---------------+-----------+
# |a70cd8f173d11174ac3bca06a381af3805a7f2e9|SOHTKMO12AB01843B0|    1|          443.0|           12.0|0.016148984|
# |ae58ceb405fa1c28749c834e0fe50a7094361951|SOHTKMO12AB01843B0|    7|          600.0|           12.0| 0.30274162|
# |f6ae5e682750e815c1709ca99138d03b039839d6|SOHTKMO12AB01843B0|    1|          613.0|           12.0|  0.8911237|
# |0150221bdccf11ec9d756440b0c1aa0a1540d177|SOHTKMO12AB01843B0|    5|          641.0|           12.0| 0.60515213|
# |702808bc4ec9022306208ef0a96ec93f9e14a7bc|SOHTKMO12AB01843B0|    5|         1672.0|           12.0| 0.38026038|
# |ddaa3a5eb4e7fd644e76b6c5159084a6fdbe6213|SOHTKMO12AB01843B0|    1|         1830.0|           12.0| 0.20424375|
# |69343b79c5c748b88a8dc5c2226833261f7255d4|SOHTKMO12AB01843B0|   16|         2030.0|           12.0|  0.7263354|
# |397c801950791a6f098b5a5cd8567d6d180c6bb3|SOHTKMO12AB01843B0|    3|         2429.0|           12.0|  0.5422786|
# |e3253118abfabf373e0a78ad994ea5e335fe786e|SOHTKMO12AB01843B0|    2|         2778.0|           12.0| 0.35795376|
# |7ffc14a55b6256c9fa73fc5c5761d210deb7f738|SOHTKMO12AB01843B0|   22|         2873.0|           12.0|  0.8501222|
# +----------------------------------------+------------------+-----+---------------+---------------+-----------+
# only showing top 10 rows
 

##Q2 b


Top 3 users

top_users = test.select(['user_id_encoded']).distinct().limit(3)
top_users.cache()
top_users.show()

# +---------------+
# |user_id_encoded|
# +---------------+
# |        53133.0|
# |         3986.0|
# |       141443.0|
# +---------------+

topUsers = alsModel.recommendForUserSubset(top_users, 10)
topUsers.cache()
topUsers.show(3, False)

# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |user_id_encoded|recommendations                                                                                                                                                                                  |
# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |53133          |[{16, 0.39670417}, {12, 0.35300282}, {29, 0.31762907}, {23, 0.30953634}, {20, 0.30727294}, {47, 0.29087225}, {35, 0.28899887}, {30, 0.27517468}, {5, 0.2573607}, {80, 0.25146678}]               |
# |3986           |[{31, 0.86407346}, {6, 0.7333927}, {138, 0.6700421}, {86, 0.62996227}, {11, 0.6232021}, {141, 0.540807}, {111, 0.44759223}, {104, 0.4405578}, {63, 0.43606928}, {16, 0.43590248}]                |
# |141443         |[{177, 0.034445345}, {141, 0.03309813}, {76, 0.03256192}, {254, 0.0315436}, {81, 0.029796176}, {260, 0.02929705}, {238, 0.027455023}, {187, 0.027346537}, {166, 0.027342826}, {197, 0.026679039}]|
# +---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


## helper functions to extract top k songs
def extract_songs_top_k(x, k):
  x = sorted(x, key=lambda x: -x[1])
  return [x[0] for x in x][0:k]

extract_songs_top_k_udf = F.udf(lambda x: extract_songs_top_k(x, k), ArrayType(IntegerType()))
## helper function to extract songs
def extract_songs(x):
  x = sorted(x, key=lambda x: -x[1])
  return [x[0] for x in x]

extract_songs_udf = F.udf(lambda x: extract_songs(x), ArrayType(IntegerType()))

recommended_songs = (
  topUsers
  .withColumn("recommended_songs", extract_songs_top_k_udf(F.col("recommendations")))
  .select("user_id_encoded", "recommended_songs")
)
recommended_songs.cache()
recommended_songs.count()
recommended_songs.show(10, 50)

# +---------------+--------------------------------------------------+
# |user_id_encoded|recommended_songs                                 |
# +---------------+--------------------------------------------------+
# |          53133|               [16, 12, 29, 20, 47, 35, 30, 5, 80]|
# |           3986|       [31, 6, 138, 86, 11, 141, 111, 104, 63, 16]|
# |         141443|[177, 141, 76, 76, 254, 81, 260, 38, 187, 166,197]|
# +---------------+--------------------------------------------------+


# Relevant songs (ground truth) songs that user actually played
users = [53133, 3986, 141443]
actualPlayed = (
  test
  .where(
    F.col("user_id_encoded").isin(users)#,
    #F.col("song_id_encoded").cast(IntegerType()),
    #F.col("plays")
  )
  .groupBy('user_id_encoded')
  .agg(
    F.collect_list(
      F.array(
        F.col("song_id_encoded"),
        F.col("plays")
      )
    ).alias('relevance')
  )
  .withColumn("relevant_songs", extract_songs_udf(F.col("relevance")))
  .select("user_id_encoded", "relevance")
)

actualPlayed.cache()
actualPlayed.count()
actualPlayed.show(3, 50)


# +---------------+--------------------------------------------------+
# |user_id_encoded|                                         relevance|
# +---------------+--------------------------------------------------+
# |           3986|[1666, 1706, 19050, 1587, 4322, 1279, 1052, 91251,|
# |          53133|[19145, 5410, 105, 7935, 154, 6886, 12716, 6153,..|
# |         141443|[10437, 718, 2607, 12309, 336, 95146, 16559, 97755|
# +---------------+--------------------------------------------------+

##combine and compare
combined = (
  recommended_songs.join(actualPlayed, on='user_id_encoded', how='inner')
  .rdd
  .map(lambda row: (row[1], row[2]))
)
combined.cache()
combined.count()
print(combined.take(1))


# [([31, 6, 138], [[1666,1706, 19050, 1587, 4322, 1279, 1052, 91251, 4623, 5383, 13740, 1252, 76385, 329, 3617, 883, 
# 597, 11005, 36255, 2666, 2158, 3768, 10482, 19448, 318, 3593, 7692, 223, 56, 1333, 19518, 242, 710, 259, 3255, 2163,
# 4206, 33949, 2011, 16372, 3101, 8206, 3003, 3444, 10457, 1453, 227, 1604, 1004, 3368, 20216, 5867, 24434, 32814, 8659,
# 8978, 5729, 289, 248, 1379, 1921, 2717, 19226, 20674, 326, 11104, 72630, 401, 609, 4719, 866, 858, 2473, 11456, 2031,
# 26948, 1825, 13366, 249, 44530, 19337, 540, 11740])]



from pyspark.mllib.evaluation import RankingMetrics
##Q2 c computing the metrics
# precision @ 10, NDCG @ 10 and Mean Average Precision(MAP)
rankingMetrics = RankingMetrics(combined)
ndcgAtK = rankingMetrics.ndcgAt(10)
print(ndcgAtK)
#0.029896389065203238
print(rankingMetrics.precisionAt(10))
#0.0165226291558473  
print(rankingMetrics.meanAveragePrecision)
#0.012721952570188318
