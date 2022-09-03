# Python and pyspark modules required

import sys
import pandas as pd
import numpy as np

from pyspark import SparkContext
from pyspark.sql import SparkSession, Row, DataFrame, Window, functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *
###add this for audio similarity to get Correlation
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
###used to answer Q4 to convert the categorical variable to numeric using oneHotEncoder 
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler


from pretty import SparkPretty  # download pretty.py from LEARN
pretty = SparkPretty(limit=5)

# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M


# Processing Q2 (a)

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
matches_manually_accepted.show(10, 40)

print(matches_manually_accepted.count())  # 488
# :<EOF>
# +------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
# |           song_id|      song_artist|                              song_title|          track_id|                            track_artist|                             track_title|
# +------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
# |SOFQHZM12A8C142342|     Josipa Lisac|                                 razloga|TRMWMFG128F92FFEF2|                            Lisac Josipa|                            1000 razloga|
# |SODXUTF12AB018A3DA|       Lutan Fyah|     Nuh Matter the Crisis Feat. Midnite|TRMWPCD12903CCE5ED|                                 Midnite|                   Nah Matter the Crisis|
# |SOASCRF12A8C1372E6|Gaetano Donizetti|L'Elisir d'Amore: Act Two: Come sen v...|TRMHIPJ128F426A2E2|Gianandrea Gavazzeni_ Orchestra E Cor...|L'Elisir D'Amore_ Act 2: Come Sen Va ...|
# |SOITDUN12A58A7AACA|     C.J. Chenier|                               Ay, Ai Ai|TRMHXGK128F42446AB|                         Clifton Chenier|                               Ay_ Ai Ai|
# |SOLZXUM12AB018BE39|           許志安|                                男人最痛|TRMRSOF12903CCF516|                                Andy Hui|                        Nan Ren Zui Tong| 
# |SOTJTDT12A8C13A8A6|                S|                                       h|TRMNKQE128F427C4D8|                             Sammy Hagar|                 20th Century Man (Live)|
# |SOGCVWB12AB0184CE2|                H|                                       Y|TRMUNCZ128F932A95D|                                Hawkwind|                25 Years (Alternate Mix)|
# |SOKDKGD12AB0185E9C|        影山ヒロノブ|Cha-La Head-Cha-La (2005 ver./DRAGON ...|TRMOOAH12903CB4B29|                        Takahashi Hiroki|Maka fushigi adventure! (2005 Version...|
# |SOPPBXP12A8C141194|    Αντώνης Ρέμος|                        O Trellos - Live|TRMXJDS128F42AE7CF|                           Antonis Remos|                               O Trellos|
# |SODQSLR12A8C133A01|    John Williams|Concerto No. 1 for Guitar and String ...|TRWHMXN128F426E03C|               English Chamber Orchestra|II. Andantino siciliano from Concerto...|            
# +------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
# only showing top 10 rows                                                                                                                                                                    
# 488            

matches_manually_accepted.write.csv('hdfs:///user/bpa78/outputs/msd/mismatches', mode = 'overwrite', header = True)

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
mismatches.show(10, 40)

print(mismatches.count())  # 19094
# :<EOF>
# +------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
# |           song_id|        song_artist|                              song_title|          track_id|  track_artist|                             track_title|                                    
# +------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+                                    
# |SOUMNSI12AB0182807|Digital Underground|                        The Way We Swing|TRMMGKQ128F9325E10|      Linkwood|           Whats up with the Underground|                                    
# |SOCMRBE12AB018C546|         Jimmy Reed|The Sun Is Shining (Digitally Remaste...|TRMMREB12903CEB1B1|    Slim Harpo|               I Got Love If You Want It|                                    
# |SOLPHZY12AC468ABA8|      Africa HiTech|                                Footstep|TRMMBOC12903CEB46E|Marcus Worgull|                 Drumstern (BONUS TRACK)|                                    
# |SONGHTM12A8C1374EF|     Death in Vegas|                            Anita Berber|TRMMITP128F425D8D0|     Valen Hsu|                                  Shi Yi|                                    
# |SONGXCA12A8C13E82E| Grupo Exterminador|                           El Triunfador|TRMMAYZ128F429ECE6|     I Ribelli|                               Lei M'Ama|                                    
# |SOMBCRC12A67ADA435|      Fading Friend|                             Get us out!|TRMMNVU128EF343EED|     Masterboy|                      Feel The Heat 2000|                                    
# |SOTDWDK12A8C13617B|       Daevid Allen|                              Past Lives|TRMMNCZ128F426FF0E| Bhimsen Joshi|            Raga - Shuddha Sarang_ Aalap|                                    
# |SOEBURP12AB018C2FB|  Cristian Paduraru|                              Born Again|TRMMPBS12903CE90E1|     Yespiring|                          Journey Stages|                                    
# |SOSRJHS12A6D4FDAA3|         Jeff Mills|                      Basic Human Design|TRMWMEL128F421DA68|           M&T|                           Drumsettester|                                    
# |SOIYAAQ12A6D4F954A|           Excepter|                                      OG|TRMWHRI128F147EA8E|    The Fevers|Não Tenho Nada (Natchs Scheint Die So...|                                    
# +------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+                                    
# only showing top 10 rows                                                                                                                                                                                                                                                                                                                                                                        
# 19094

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

#triplets.write.csv('hdfs:///user/bpa78/outputs/msd/triplets', mode = 'overwrite', header = True)

mismatches_not_accepted = mismatches.join(matches_manually_accepted, on="song_id", how="left_anti")
triplets_not_mismatched = triplets.join(mismatches_not_accepted, on="song_id", how="left_anti")

print(triplets.count())         # 48373586
# :<EOF>                                                                                                                                                                                          2022-05-16 21:15:30,757 WARN execution.CacheManager: Asked to cache already cached data.                                                                                                       
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
# +----------------------------------------+------------------+-----+                                                                                                                             only showing top 10 rows                                                                                                                                                                                                                                                                                                                                                                        
# 48373586 
print(triplets_not_mismatched.count())  # 45795111

#triplets.write.csv('hdfs:///user/bpa78/outputs/msd/triplets', mode = 'overwrite', header = True)
#45795111  


# Processing Q2 (b)

# hdfs dfs -cat "/data/msd/audio/attributes/*" | awk -F',' '{print $2}' | sort | uniq

# NUMERIC
# real
# real 
# string
# string
# STRING

audio_attribute_type_mapping = {
  "NUMERIC": DoubleType(),
  "real": DoubleType(),
  "string": StringType(),
  "STRING": StringType()
}

audio_dataset_names = [
  "msd-jmir-area-of-moments-all-v1.0",
  "msd-jmir-lpc-all-v1.0",
  "msd-jmir-methods-of-moments-all-v1.0",
  "msd-jmir-mfcc-all-v1.0",
  "msd-jmir-spectral-all-all-v1.0",
  "msd-jmir-spectral-derivatives-all-all-v1.0",
  "msd-marsyas-timbral-v1.0",
  "msd-mvd-v1.0",
  "msd-rh-v1.0",
  "msd-rp-v1.0",
  "msd-ssd-v1.0",
  "msd-trh-v1.0",
  "msd-tssd-v1.0"
]

audio_dataset_schemas = {}
for audio_dataset_name in audio_dataset_names:
  print(audio_dataset_name)

  audio_dataset_path = f"/scratch-network/courses/2022/DATA420-22S1/data/msd/audio/attributes/{audio_dataset_name}.attributes.csv"
  with open(audio_dataset_path, "r") as f:
    rows = [line.strip().split(",") for line in f.readlines()]

  # you could rename feature columns with a short generic name

  # rows[-1][0] = "track_id"
  # for i, row in enumerate(rows[0:-1]):
  #   row[0] = f"feature_{i:04d}"

  audio_dataset_schemas[audio_dataset_name] = StructType([
  StructField(row[0], audio_attribute_type_mapping[row[1]], True) for row in rows
  ])
  
  print(audio_dataset_schemas[audio_dataset_name])

# Audio Similarity Q1 (a)

# you can now use these schemas to load any one of the above datasets, e.g.

audio_dataset_name = "msd-jmir-methods-of-moments-all-v1.0"

schema = audio_dataset_schemas[audio_dataset_name]

audioFeatures = (
    spark.read.format("com.databricks.spark.csv") 
    .option("header", "false") 
    .option("inferSchema", "false") 
    .schema(schema) 
    .load(f"hdfs:///data/msd/audio/features/{audio_dataset_name}.csv")
    .cache()
)
print(pretty(audioFeatures.head().asDict()))
# :<EOF>
# {
  # 'MSD_TRACKID': "'TRHFHQZ12903C9E2D5'",
  # 'Method_of_Moments_Overall_Average_1': 0.319,
  # 'Method_of_Moments_Overall_Average_2': 33.41,
  # 'Method_of_Moments_Overall_Average_3': 1371.0,
  # 'Method_of_Moments_Overall_Average_4': 64240.0,
  # 'Method_of_Moments_Overall_Average_5': 8398000.0,
  # 'Method_of_Moments_Overall_Standard_Deviation_1': 0.1545,
  # 'Method_of_Moments_Overall_Standard_Deviation_2': 13.11,
  # 'Method_of_Moments_Overall_Standard_Deviation_3': 840.0,
  # 'Method_of_Moments_Overall_Standard_Deviation_4': 41080.0,
  # 'Method_of_Moments_Overall_Standard_Deviation_5': 7108000.0
# }  

audioFeatures.count()
#Out[7]: 994623  

#statistics of featureData
statistics = (
    audioFeatures
    .drop('MSD_TRACKID')
    .describe()
    .toPandas()
    .set_index('summary')
    .round(2)
    .rename_axis(None)
    .transpose()
)
print(statistics)

# :<EOF>
                                                 # count                 mean               stddev        min       max
# Method_of_Moments_Overall_Standard_Deviation_1  994623  0.15498176001746336  0.06646213086143025        0.0     0.959
# Method_of_Moments_Overall_Standard_Deviation_2  994623   10.384550576952307   3.8680013938746836        0.0     55.42
# Method_of_Moments_Overall_Standard_Deviation_3  994623    526.8139724398096    180.4377549977526        0.0    2919.0
# Method_of_Moments_Overall_Standard_Deviation_4  994623    35071.97543290272   12806.816272955562        0.0  407100.0
# Method_of_Moments_Overall_Standard_Deviation_5  994623    5297870.369577217   2089356.4364558065        0.0   4.657E7
# Method_of_Moments_Overall_Average_1             994623   0.3508444432531317  0.18557956834383812        0.0     2.647
# Method_of_Moments_Overall_Average_2             994623    27.46386798784071    8.352648595163764        0.0     117.0
# Method_of_Moments_Overall_Average_3             994623   1495.8091812075545   505.89376391902306        0.0    5834.0
# Method_of_Moments_Overall_Average_4             994623   143165.46163257837   50494.276171032274  -146300.0  452500.0
# Method_of_Moments_Overall_Average_5             994623  2.396783048473542E7    9307340.299219666        0.0   9.477E7

features = audioFeatures.drop('MSD_TRACKID')
col_names = features.columns
vector_col = "corrFreatures"
assembler = VectorAssembler(inputCols = features.columns, outputCol = vector_col)
df_vector = assembler.transform(features).select(vector_col)
corrMatrix = Correlation.corr(df_vector, vector_col,'pearson').collect()[0][0].toArray().tolist()
df_corr = spark.createDataFrame(corrMatrix, col_names)
df_corr.show()
# :<EOF>
# +----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
# |Method_of_Moments_Overall_Standard_Deviation_1|Method_of_Moments_Overall_Standard_Deviation_2|Method_of_Moments_Overall_Standard_Deviation_3|Method_of_Moments_Overall_Standard_Deviation_4|Method_of_Moments_Overall_Standard_Deviation_5|Method_of_Moments_Overall_Average_1|Method_of_Moments_Overall_Average_2|Method_of_Moments_Overall_Average_3|Method_of_Moments_Overall_Average_4|Method_of_Moments_Overall_Average_5|
# +----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
# |                                           1.0|                            0.4262803507126525|                            0.2963058884428714|                          0.061038653434281755|                          -0.05533585373212792|                  0.754207868124561|                0.49792898449141226|                 0.4475646091410656|                0.16746556577732985|                0.10040743826468972|
# |                            0.4262803507126525|                                           1.0|                            0.8575486565280155|                            0.6095209087098502|                           0.43379677138588096|                0.02522827251692226|                0.40692287092548374|                 0.3963535260228954|               0.015606573104508177|               -0.04090215286558332|
# |                            0.2963058884428714|                            0.8575486565280155|                                           1.0|                            0.8030096521046056|                             0.682909352578237|                -0.0824150748064068|                 0.1259102539613699|                0.18496247123427453|               -0.08817390503127001|               -0.13505636493077303|
# |                          0.061038653434281755|                            0.6095209087098502|                            0.8030096521046056|                                           1.0|                            0.9422444252722715|               -0.32769149973561645|               -0.22321966439720595|                -0.1582307411081193|               -0.24503391729599744|               -0.22087302938348774|
# |                          -0.05533585373212792|                           0.43379677138588096|                             0.682909352578237|                            0.9422444252722715|                                           1.0|                -0.3925512500751903|               -0.35501873907701154|               -0.28596556089550484|               -0.26019779474328325|                -0.2118128137911948|
# |                             0.754207868124561|                           0.02522827251692226|                           -0.0824150748064068|                          -0.32769149973561645|                           -0.3925512500751903|                                1.0|                 0.5490152221868251|                 0.5185026975023406|                  0.347112009220111|                 0.2785128021376318|
# |                           0.49792898449141226|                           0.40692287092548374|                            0.1259102539613699|                          -0.22321966439720595|                          -0.35501873907701154|                 0.5490152221868251|                                1.0|                 0.9033667462436238|                 0.5164990583386245|                0.42254939616245646|
# |                            0.4475646091410656|                            0.3963535260228954|                           0.18496247123427453|                           -0.1582307411081193|                          -0.28596556089550484|                 0.5185026975023406|                 0.9033667462436238|                                1.0|                   0.77280689539197|                 0.6856452835776947|
# |                           0.16746556577732985|                          0.015606573104508177|                          -0.08817390503127001|                          -0.24503391729599744|                          -0.26019779474328325|                  0.347112009220111|                 0.5164990583386245|                   0.77280689539197|                                1.0|                 0.9848665037806468|
# |                           0.10040743826468972|                          -0.04090215286558332|                          -0.13505636493077303|                          -0.22087302938348774|                           -0.2118128137911948|                 0.2785128021376318|                0.42254939616245646|                 0.6856452835776947|                 0.9848665037806468|                                1.0|
# +----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+


schemaGenre = StructType([
    StructField('TRACK_ID', StringType()),
    StructField('GENRE_LABEL', StringType())
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
genreData = (
    spark.read.format("com.databricks.spark.csv") 
    .option("header", "false")
    .option("inferSchema", "false")
    .option("delimiter", "\t")
    .schema(schemaGenre)
    .load("hdfs:////data/msd/genre/msd-MAGD-genreAssignment.tsv")
    .cache()
)
genreData.show()

# :<EOF>
# 2022-06-01 15:21:09,022 WARN execution.CacheManager: Asked to cache already cached data.
# +------------------+--------------+
# |          TRACK_ID|   GENRE_LABEL|
# +------------------+--------------+
# |TRAAAAK128F9318786|      Pop_Rock|
# |TRAAAAV128F421A322|      Pop_Rock|
# |TRAAAAW128F429D538|           Rap|
# |TRAAABD128F429CF47|      Pop_Rock|
# |TRAAACV128F423E09E|      Pop_Rock|
# |TRAAADT12903CCC339|Easy_Listening|
# |TRAAAED128E0783FAB|         Vocal|
# |TRAAAEF128F4273421|      Pop_Rock|
# |TRAAAEM128F93347B9|    Electronic|
# |TRAAAFD128F92F423A|      Pop_Rock|
# |TRAAAFP128F931B4E3|           Rap|
# |TRAAAGR128F425B14B|      Pop_Rock|
# |TRAAAGW12903CC1049|         Blues|
# |TRAAAHD128F42635A5|      Pop_Rock|
# |TRAAAHE12903C9669C|      Pop_Rock|
# |TRAAAHJ128F931194C|      Pop_Rock|
# |TRAAAHZ128E0799171|           Rap|
# |TRAAAIR128F1480971|           RnB|
# |TRAAAJG128F9308A25|          Folk|
# |TRAAAMO128F1481E7F|     Religious|
# +------------------+--------------+
# only showing top 20 rows

genreData.count()
#Out[53]: 422714

genreDistribution = (
    genreData
    .groupBy(F.col('GENRE_LABEL'))
    .count()
    .orderBy(F.col('count').alias('COUNT').desc())
)
genreDistribution.show(21)

# +--------------+------+
# |   GENRE_LABEL| count|
# +--------------+------+
# |      Pop_Rock|238786|
# |    Electronic| 41075|
# |           Rap| 20939|
# |          Jazz| 17836|
# |         Latin| 17590|
# |           RnB| 14335|
# | International| 14242|
# |       Country| 11772|
# |     Religious|  8814|
# |        Reggae|  6946|
# |         Blues|  6836|
# |         Vocal|  6195|
# |          Folk|  5865|
# |       New Age|  4010|
# | Comedy_Spoken|  2067|
# |        Stage |  1614|
# |Easy_Listening|  1545|
# |   Avant_Garde|  1014|
# |     Classical|   556|
# |      Children|   477|
# |       Holiday|   200|
# +--------------+------+


audioFeatures = (
    audioFeatures
    .withColumn("MSD_TRACKID", regexp_replace("MSD_TRACKID", "'",""))
    .withColumnRenamed("MSD_TRACKID","TRACK_ID")
)

audioGenre = (
    audioFeatures
    .join(
        genreData,
        on="TRACK_ID",
        how="left"
    )
    .filter(F.col("GENRE_LABEL").isNotNull())
)

audioGenre.printSchema()
# root
 # |-- TRACK_ID: string (nullable = true)
 # |-- Method_of_Moments_Overall_Standard_Deviation_1: double (nullable = true)
 # |-- Method_of_Moments_Overall_Standard_Deviation_2: double (nullable = true)
 # |-- Method_of_Moments_Overall_Standard_Deviation_3: double (nullable = true)
 # |-- Method_of_Moments_Overall_Standard_Deviation_4: double (nullable = true)
 # |-- Method_of_Moments_Overall_Standard_Deviation_5: double (nullable = true)
 # |-- Method_of_Moments_Overall_Average_1: double (nullable = true)
 # |-- Method_of_Moments_Overall_Average_2: double (nullable = true)
 # |-- Method_of_Moments_Overall_Average_3: double (nullable = true)
 # |-- Method_of_Moments_Overall_Average_4: double (nullable = true)
 # |-- Method_of_Moments_Overall_Average_5: double (nullable = true)
 # |-- GENRE_LABEL: string (nullable = true)
audioGenre.show()
        
# :<EOF>

# In [65]: audioGenre.show()
# +------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-------------+
# |          TRACK_ID|Method_of_Moments_Overall_Standard_Deviation_1|Method_of_Moments_Overall_Standard_Deviation_2|Method_of_Moments_Overall_Standard_Deviation_3|Method_of_Moments_Overall_Standard_Deviation_4|Method_of_Moments_Overall_Standard_Deviation_5|Method_of_Moments_Overall_Average_1|Method_of_Moments_Overall_Average_2|Method_of_Moments_Overall_Average_3|Method_of_Moments_Overall_Average_4|Method_of_Moments_Overall_Average_5|  GENRE_LABEL|
# +------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-------------+
# |TRAAABD128F429CF47|                                        0.1308|                                         9.587|                                         459.9|                                       27280.0|                                     4303000.0|                             0.2474|                              26.02|                             1067.0|                            67790.0|                          8281000.0|     Pop_Rock|
# |TRAABPK128F424CFDB|                                        0.1208|                                         6.738|                                         215.1|                                       11890.0|                                     2278000.0|                             0.4882|                              41.76|                             2164.0|                           220400.0|                             3.79E7|     Pop_Rock|
# |TRAACER128F4290F96|                                        0.2838|                                         8.995|                                         429.5|                                       31990.0|                                     5272000.0|                             0.5388|                              28.29|                             1656.0|                           185100.0|                            3.164E7|     Pop_Rock|
# |TRAADYB128F92D7E73|                                        0.1346|                                         7.321|                                         499.6|                                       38460.0|                                     5877000.0|                             0.2839|                              15.75|                              929.6|                           116500.0|                            2.058E7|         Jazz|
# |TRAAGHM128EF35CF8E|                                        0.1563|                                         9.959|                                         502.8|                                       26190.0|                                     3660000.0|                             0.3835|                              28.24|                             1864.0|                           180800.0|                            2.892E7|   Electronic|
# |TRAAGRV128F93526C0|                                        0.1076|                                         7.401|                                         389.7|                                       19350.0|                                     2739000.0|                             0.4221|                              30.99|                             1861.0|                           191700.0|                            3.166E7|     Pop_Rock|
# |TRAAGTO128F1497E3C|                                        0.1069|                                         8.987|                                         562.6|                                       43100.0|                                     7057000.0|                             0.1007|                               22.9|                             1346.0|                           157700.0|                            2.738E7|     Pop_Rock|
# |TRAAHAU128F9313A3D|                                       0.08485|                                         9.031|                                         445.9|                                       23750.0|                                     3274000.0|                             0.2583|                              35.59|                             2015.0|                           198400.0|                            3.336E7|     Pop_Rock|
# |TRAAHEG128E07861C3|                                        0.1699|                                         17.22|                                         741.3|                                       52440.0|                                     8275000.0|                             0.2812|                              28.83|                             1671.0|                           160800.0|                            2.695E7|          Rap|
# |TRAAHZP12903CA25F4|                                        0.1654|                                         12.31|                                         565.1|                                       33100.0|                                     5273000.0|                             0.1861|                              38.38|                             1962.0|                           196600.0|                            3.355E7|          Rap|
# |TRAAICW128F1496C68|                                        0.1104|                                         7.123|                                         398.2|                                       19540.0|                                     3240000.0|                             0.2871|                              28.53|                             1807.0|                           189400.0|                            3.156E7|International|
# |TRAAJJW12903CBDDCB|                                        0.2267|                                         14.88|                                         592.7|                                       37980.0|                                     4569000.0|                             0.4219|                              36.17|                             2111.0|                           179400.0|                            2.952E7|International|
# |TRAAJKJ128F92FB44F|                                       0.03861|                                          6.87|                                         407.8|                                       41310.0|                                     7299000.0|                             0.0466|                              15.79|                              955.1|                           121700.0|                            2.124E7|         Folk|
# |TRAAKLX128F934CEE4|                                        0.1647|                                         16.77|                                         850.0|                                       64420.0|                                       1.011E7|                             0.2823|                              26.52|                             1600.0|                           152000.0|                            2.587E7|   Electronic|
# |TRAAKWR128F931B29F|                                       0.04881|                                         9.331|                                         564.0|                                       34410.0|                                     4920000.0|                            0.08647|                               18.1|                              880.5|                            57700.0|                          6429000.0|     Pop_Rock|
# |TRAALQN128E07931A4|                                        0.1989|                                         12.83|                                         578.7|                                       30690.0|                                     4921000.0|                             0.5452|                              33.37|                             2019.0|                           188700.0|                             3.14E7|   Electronic|
# |TRAAMFF12903CE8107|                                        0.1385|                                         9.699|                                         581.6|                                       31590.0|                                     4569000.0|                             0.3706|                              23.63|                             1554.0|                           163800.0|                             2.72E7|     Pop_Rock|
# |TRAAMHG128F92ED7B2|                                        0.1799|                                         10.52|                                         551.4|                                       29170.0|                                     4396000.0|                             0.4046|                              30.78|                             1806.0|                           183200.0|                            3.059E7|International|
# |TRAAROH128F42604B0|                                        0.1192|                                          16.4|                                         737.3|                                       41670.0|                                     6295000.0|                             0.2284|                              31.04|                             1878.0|                           169100.0|                            2.829E7|   Electronic|
# |TRAARQN128E07894DF|                                        0.2559|                                         15.23|                                         757.1|                                       61750.0|                                       1.065E7|                             0.5417|                              40.96|                             2215.0|                           189000.0|                             3.21E7|     Pop_Rock|
# +------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-------------+
# only showing top 20 rows
#audioGenre.write.csv('hdfs:///user/bpa78/outputs/msd/audioSimilarity', mode = 'overwrite', header = True)
#audioGenre = spark.read.load('hdfs:///user/bpa78/outputs/msd/audioSimilarity', format = "com.databricks.spark.csv", header = True, inferSchema=True)

def print_class_balance(data, name):
  N = data.count()
  counts = data.groupBy("label").count().toPandas()
  counts["ratio"] = counts["count"] / N
  print(name)
  print(N)
  print(counts)
  print("")

#####Q2 a
###Features selected 
finalData = audioGenre.select([col for col in audioGenre.columns if col in ["Method_of_Moments_Overall_Standard_Deviation_1","Method_of_Moments_Overall_Standard_Deviation_2","Method_of_Moments_Overall_Standard_Deviation_3","Method_of_Moments_Overall_Standard_Deviation_4","Method_of_Moments_Overall_Average_1","Method_of_Moments_Overall_Average_3","Method_of_Moments_Overall_Average_5", "GENRE_LABEL", "TRACK_ID"]])
finalData.write.csv('hdfs:///user/bpa78/outputs/msd/audioSimFinalData', mode = 'overwrite', header = True)



###Q2 b
finalData = (
    finalData
    .withColumn("label", F.when(finalData.GENRE_LABEL == 'Electronic', 1).otherwise(0)
    )
 )
 
#Class balance
class_balance = (
    finalData
    .groupBy("label")
    .count()
    .show(2)
)

# :<EOF>
# +-----+------+
# |label| count|
# +-----+------+
# |    1| 40666|
# |    0|379954|
# +-----+------+

##2c Splitting the datasets

##Stratified Random Sampling
temp = (
  finalData
  .withColumn("id", monotonically_increasing_id())
  .withColumn("Random", rand())
  .withColumn(
    "Row",
    row_number()
    .over(
      Window
      .partitionBy("label")
      .orderBy("Random")
    )
  )
)

class_counts = (
  finalData
  .groupBy("label")
  .count()
  .toPandas()
  .set_index("label")["count"]
  .to_dict()
)
classes = sorted(class_counts.keys())

train = temp
for c in classes:
  train = train.where((col("label") != c) | (col("Row") < class_counts[c] * 0.7))

train.cache()

test = temp.join(train, on="id", how="left_anti")
test.cache()

train = train.drop("id", "Random", "Row")
test = test.drop("id", "Random", "Row")

print_class_balance(finalData, "finalData")
print_class_balance(train, "train")
print_class_balance(test, "test")

# :<EOF>
# finalData
# 420620
   # label   count     ratio
# 0      1   40666  0.096681
# 1      0  379954  0.903319

# train
# 294433
   # label   count     ratio
# 0      1   28466  0.096681
# 1      0  265967  0.903319

# test
# 126187
   # label   count     ratio
# 0      1   12200  0.096682
# 1      0  113987  0.903318

###Normalising and transforming the data
assembler = VectorAssembler(
  inputCols=[col for col in finalData.columns if col.startswith("M")],
  outputCol="Features"
)
scaler = StandardScaler(inputCol="Features", outputCol="ScaledFeatures", withMean = False, withStd = True)
#pca = PCA(k=5, inputCol = "ScaledFeatures", outputCol = "TransformedFeatures")


#Pipeline to set the stages
pipeline = Pipeline(stages = [assembler, scaler])#, pca])

#fit the pipeline to features
pipelineFit = pipeline.fit(train)
trainData = pipelineFit.transform(train)
trainData.select(['TRACK_ID', 'Features', 'ScaledFeatures', 'label']).show(5)
testData = pipelineFit.transform(test)

# +------------------+--------------------+--------------------+-----+
# |          TRACK_ID|            Features|      ScaledFeatures|label|
# +------------------+--------------------+--------------------+-----+
# |TRAABPK128F424CFDB|[0.1208,6.738,215...|[1.87165692771989...|    0|
# |TRAACER128F4290F96|[0.2838,8.995,429...|[4.39715427224260...|    0|
# |TRAADYB128F92D7E73|[0.1346,7.321,499...|[2.08547204032365...|    0|
# |TRAAGRV128F93526C0|[0.1076,7.401,389...|[1.66713812435977...|    0|
# |TRAAHAU128F9313A3D|[0.08485,9.031,44...|[1.31465306553835...|    0|
# +------------------+--------------------+--------------------+-----+
# only showing top 5 rows

#helpers udf
def with_custom_prediction(predictions, threshold, probabilityCol="probability", customPredictionCol="customPrediction"):

  def apply_custom_threshold(probability, threshold):
    return int(probability[1] > threshold)

  apply_custom_threshold_udf = udf(lambda x: apply_custom_threshold(x, threshold), IntegerType())

  return predictions.withColumn(customPredictionCol, apply_custom_threshold_udf(F.col(probabilityCol)))

def print_binary_metrics(predictions, threshold = 0.5, labelCol="label", predictionCol="prediction", rawPredictionCol="rawPrediction", probabilityCol="probability"):
 
  if threshold != 0.5:

    predictions = with_custom_prediction(predictions, threshold)
    predictionCol = "customPrediction"

  total = predictions.count()
  positive = predictions.filter((F.col(labelCol) == 1)).count()
  negative = predictions.filter((F.col(labelCol) == 0)).count()
  nP = predictions.filter((F.col(predictionCol) == 1)).count()
  nN = predictions.filter((F.col(predictionCol) == 0)).count()
  TP = predictions.filter((F.col(predictionCol) == 1) & (F.col(labelCol) == 1)).count()
  FP = predictions.filter((F.col(predictionCol) == 1) & (F.col(labelCol) == 0)).count()
  FN = predictions.filter((F.col(predictionCol) == 0) & (F.col(labelCol) == 1)).count()
  TN = predictions.filter((F.col(predictionCol) == 0) & (F.col(labelCol) == 0)).count()
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  F1score = 2*((precision*recall)/(precision+recall)) if (precision+recall)>0 else 0
  binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol=rawPredictionCol, labelCol=labelCol, metricName="areaUnderROC")
  auroc = binary_evaluator.evaluate(predictions)

  print('actual total:    {}'.format(total))
  print('actual positive: {}'.format(positive))
  print('actual negative: {}'.format(negative))
  print('threshold:       {}'.format(threshold))
  print('nP:              {}'.format(nP))
  print('nN:              {}'.format(nN))
  print('TP:              {}'.format(TP))
  print('FP:              {}'.format(FP))
  print('FN:              {}'.format(FN))
  print('TN:              {}'.format(TN))
  print('precision:       {}'.format(precision))#TP / (TP + FP)) if (TP + FP) > 0 else 0)
  print('recall:          {}'.format(recall))#TP / (TP + FN)) if (TP + FN) > 0 else 0)
  print('accuracy:        {}'.format((TP + TN) / total))
  print('F1 score:        {}'.format(F1score))#2*((precision*recall)/(precision+recall)) if (precision+recall)>0 else 0) 
  print('auroc:           {}'.format(auroc))

##2d
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier, DecisionTreeClassifier  
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.ml.classification import LinearSVC
#training the three classification algorithms
lr = LogisticRegression(featuresCol='ScaledFeatures', labelCol='label')
gbt = GBTClassifier(featuresCol='ScaledFeatures', labelCol='label')
dt = DecisionTreeClassifier(featuresCol='ScaledFeatures', labelCol='label')
rf = RandomForestClassifier(featuresCol='ScaledFeatures', labelCol='label')


lr_model = lr.fit(trainData)
predictions = lr_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions,threshold = 0.1)

# 2022-06-07 23:34:29,876 WARN netlib.BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
# 2022-06-07 23:34:29,877 WARN netlib.BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.1
# nP:              43926
# nN:              82261
# TP:              7609
# FP:              36317
# FN:              4591
# TN:              77670
# precision:       0.17322314802167282
# recall:          0.6236885245901639
# accuracy:        0.675814465832455
# F1 score:        0.2711399351459217
# auroc:           0.6975074271483647
gbt_model = gbt.fit(trainData)
predictions = gbt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions, threshold = 0.1)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.1
# nP:              42489
# nN:              83698
# TP:              8874
# FP:              33615
# FN:              3326
# TN:              80372
# precision:       0.20885405634399493
# recall:          0.7273770491803279
# accuracy:        0.7072519356193586
# F1 score:        0.32452595585949645
# auroc:           0.7948656871570198

rf_model = rf.fit(trainData)
predictions = rf_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions,threshold = 0.1)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.1
# nP:              32910
# nN:              93277
# TP:              7043
# FP:              25867
# FN:              5157
# TN:              88120
# precision:       0.2140079003342449
# recall:          0.5772950819672131
# accuracy:        0.7541426612884053
# F1 score:        0.3122589226335624
# auroc:           0.720696473943606

dt_model = dt.fit(trainData)
predictions = dt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions,threshold = 0.1)

# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.1
# nP:              31552
# nN:              94635
# TP:              6924
# FP:              24628
# FN:              5276
# TN:              89359
# precision:       0.21944726166328601
# recall:          0.5675409836065574
# accuracy:        0.7630183774873799
# F1 score:        0.31651124520021945
# auroc:           0.5472715244922236
lr_model = lr.fit(trainData)
predictions = lr_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              302
# nN:              125885
# TP:              74
# FP:              228
# FN:              12126
# TN:              113759
# precision:       0.24503311258278146
# recall:          0.0060655737704918035
# accuracy:        0.9020976804266684
# F1 score:        0.01183810590305551
# auroc:           0.6975073721377775


gbt_model = gbt.fit(trainData)
predictions = gbt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              1695
# nN:              124492
# TP:              1068
# FP:              627
# FN:              11132
# TN:              113360
# precision:       0.6300884955752213
# recall:          0.08754098360655738
# accuracy:        0.9068129046573736
# F1 score:        0.15372436128103634
# auroc:           0.794864875660972

rf_model = rf.fit(trainData)
predictions = rf_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)


# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              0
# nN:              126187
# TP:              0
# FP:              0
# FN:              12200
# TN:              113987
# precision:       0
# recall:          0.0
# accuracy:        0.9033180914040273
# F1 score:        0
# auroc:           0.720696473943606

dt_model = dt.fit(trainData)
predictions = dt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              202
# nN:              125985
# TP:              129
# FP:              73
# FN:              12071
# TN:              113914
# precision:       0.6386138613861386
# recall:          0.010573770491803278
# accuracy:        0.903761877213976
# F1 score:        0.020803096274794385
# auroc:           0.5472715244922236




# Downsampling

trainDownsampled = (
    trainData
    .withColumn("Random", rand())
    .where((col("label") != 0) | ((col("label") == 0) & (col("Random") < 1 * (40666 / 379954))))
)
trainDownsampled.cache()

print_class_balance(trainDownsampled, "trainDownsampled")
# :<EOF>
# trainDownsampled
# 56932
   # label  count  ratio
# 0      1  28466    0.5
# 1      0  28466    0.5


lr_model = lr.fit(trainDownsampled)
predictions = lr_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              45673
# nN:              80514
# TP:              7736
# FP:              37937
# FN:              4464
# TN:              76050
# precision:       0.16937796947868544
# recall:          0.6340983606557377
# accuracy:        0.6639828191493578
# F1 score:        0.2673440118880998
# auroc:           0.6976190972021975

gbt_model = gbt.fit(trainDownsampled)
predictions = gbt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              37487
# nN:              88700
# TP:              8425
# FP:              29062
# FN:              3775
# TN:              84925
# precision:       0.22474457812041507
# recall:          0.6905737704918032
# accuracy:        0.7397750956913153
# F1 score:        0.33912290941292494
# auroc:           0.7972970224387105


rf_model = rf.fit(trainDownsampled)
predictions = rf_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)
# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              38482
# nN:              87705
# TP:              8166
# FP:              30316
# FN:              4034
# TN:              83671
# precision:       0.21220310794657243
# recall:          0.6693442622950819
# accuracy:        0.7277849540760934
# F1 score:        0.3222445838759323
# auroc:           0.7779953836409588

dt_model = dt.fit(trainDownsampled)
predictions = dt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              34676
# nN:              91511
# TP:              7803
# FP:              26873
# FN:              4397
# TN:              87114
# precision:       0.22502595455069788
# recall:          0.6395901639344262
# accuracy:        0.7521931736232734
# F1 score:        0.33292089768751604
# auroc:           0.5940313390641182

import numpy as np

##Upsampling
ratio = 10
n = 19
p = ratio / n  # ratio < n such that probability < 1

def randomResample(x, n, p):
    # Can implement custom sampling logic per class,
    if x == 0:
        return [0]  # no sampling
    if x == 1:
        return list(range((np.sum(np.random.random(n) > p))))  # upsampling
    return []  # drop

randomResample_udf = udf(lambda x: randomResample(x, n, p), ArrayType(IntegerType()))

trainUpsampled= (
    trainData
    .withColumn("Sample", randomResample_udf(col("label")))
    .select(
        col("ScaledFeatures"),
        col("label"),
        explode(col("Sample")).alias("Sample")
    )
    .drop("Sample")
)

print_class_balance(trainUpsampled, "trainUpsampled")
# :<EOF>

# trainUpsampled
# 522527
   # label   count     ratio
# 0      1  256495  0.490874
# 1      0  265967  0.509001

lr_model = lr.fit(trainUpsampled)
predictions = lr_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)
# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              43031
# nN:              83156
# TP:              7488
# FP:              35543
# FN:              4712
# TN:              78444
# precision:       0.17401408287048872
# recall:          0.6137704918032787
# accuracy:        0.6809893253663214
# F1 score:        0.2711520703952491
# auroc:           0.6976667029329057

gbt_model = gbt.fit(trainUpsampled)
predictions = gbt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              35078
# nN:              91109
# TP:              8158
# FP:              26920
# FN:              4042
# TN:              87067
# precision:       0.23256742117566567
# recall:          0.6686885245901639
# accuracy:        0.7546339955779914
# F1 score:        0.3451076610685731
# auroc:           0.7978809903832865

rf_model = rf.fit(trainUpsampled)
predictions = rf_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)
# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              37189
# nN:              88998
# TP:              8029
# FP:              29160
# FN:              4171
# TN:              84827
# precision:       0.21589717389550672
# recall:          0.6581147540983606
# accuracy:        0.735860270867839
# F1 score:        0.3251331268096134
# auroc:           0.7787093110416533

dt_model = dt.fit(trainUpsampled)
predictions = dt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)
# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              30809
# nN:              95378
# TP:              7359
# FP:              23450
# FN:              4841
# TN:              90537
# precision:       0.2388587750332695
# recall:          0.6031967213114754
# accuracy:        0.7758009937632244
# F1 score:        0.34220744495338185
# auroc:           0.5831932261616835


##3Q a)Hyperparameters for each of the above classification model

gbtModel = GBTClassifier(maxDepth = 5, maxBins = 32, maxIter = 20, featuresCol='ScaledFeatures', labelCol='label')
gbt_model = gbtModel.fit(trainData)
predictions = gbt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              1695
# nN:              124492
# TP:              1068
# FP:              627
# FN:              11132
# TN:              113360
# precision:       0.6300884955752213
# recall:          0.08754098360655738
# accuracy:        0.9068129046573736
# F1 score:        0.15372436128103634
# auroc:           0.7948598146869494

gbt_model = gbtModel.fit(trainUpsampled)
predictions = gbt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)
# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              36110
# nN:              90077
# TP:              8277
# FP:              27833
# FN:              3923
# TN:              86154
# precision:       0.22921628357795623
# recall:          0.6784426229508197
# accuracy:        0.748341746772647
# F1 score:        0.3426619747464293
# auroc:           0.7977946528846328

gbt_model = gbtModel.fit(trainDownsampled)
predictions = gbt_model.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              37487
# nN:              88700
# TP:              8425
# FP:              29062
# FN:              3775
# TN:              84925
# precision:       0.22474457812041507
# recall:          0.6905737704918032
# accuracy:        0.7397750956913153
# F1 score:        0.33912290941292494
# auroc:           0.7972894802355229

#3b Using Cross-Validation to tune hyperparameters for the best classification model
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=5)
cvModel = cv.fit(trainDownsampled)
predictions = cvModel.transform(testData)
predictions.cache()
print_binary_metrics(predictions)

# :<EOF>
# actual total:    126187
# actual positive: 12200
# actual negative: 113987
# threshold:       0.5
# nP:              36245
# nN:              89942
# TP:              8426
# FP:              27819
# FN:              3774
# TN:              86168
# precision:       0.23247344461305008
# recall:          0.6906557377049181
# accuracy:        0.749633480469462
# F1 score:        0.3478583961193106
# auroc:           0.8030699391662007

###Q4
#4a Random forest model is choosen for multiclass classification
##4b Converting the Genre column into an integer index that encodes each genre consistently.
audioMC = spark.read.load('hdfs:///user/bpa78/outputs/msd/audioSimFinalData', format = "com.databricks.spark.csv", header = True, inferSchema=True) 

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
label_stringIdx = StringIndexer(inputCol = "GENRE_LABEL", outputCol = "label")
label_data = label_stringIdx.fit(audioMC).transform(audioMC)
label_data.show(5)


# :<EOF>
# +------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------+-----+
# |          TRACK_ID|Method_of_Moments_Overall_Standard_Deviation_1|Method_of_Moments_Overall_Standard_Deviation_2|Method_of_Moments_Overall_Standard_Deviation_3|Method_of_Moments_Overall_Standard_Deviation_4|Method_of_Moments_Overall_Average_1|Method_of_Moments_Overall_Average_3|Method_of_Moments_Overall_Average_5|GENRE_LABEL|label|
# +------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------+-----+
# |TRAAACV128F423E09E|                                         0.231|                                         4.524|                                         235.0|                                       18830.0|                             0.6727|                              905.8|                          5539000.0|   Pop_Rock|  0.0|
# |TRAABVM128F14970CA|                                        0.1966|                                         10.88|                                         556.1|                                       30410.0|                             0.4061|                             1095.0|                          7518000.0|       Folk| 12.0|
# |TRAABWX128F1464374|                                        0.1351|                                         22.03|                                         966.1|                                       62050.0|                             0.2367|                             2163.0|                             2.54E7| Electronic|  1.0|
# |TRAACPE128F421C1B9|                                       0.09932|                                         10.29|                                         513.4|                                       34960.0|                             0.1759|                              975.1|                          9254000.0|        RnB|  5.0|
# |TRAACQT128F9331780|                                       0.06933|                                         11.79|                                         519.1|                                       32470.0|                             0.1688|                             1616.0|                            3.096E7|   Pop_Rock|  0.0|
# +------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------+-----+
# only showing top 5 rows


##4c Splitting the data to train the random forest algorithm 

temp = (
  label_data
  .withColumn("id", monotonically_increasing_id())
  .withColumn("Random", rand())
  .withColumn(
    "Row",
    row_number()
    .over(
      Window
      .partitionBy("label")
      .orderBy("Random")
    )
  )
)
class_counts = (
  label_data
  .groupBy("label")
  .count()
  .toPandas()
  .set_index("label")["count"]
  .to_dict()
)
classes = sorted(class_counts.keys())

train = temp
for c in classes:
  train = train.where((col("label") != c) | (col("Row") < class_counts[c] * 0.7))

train.cache()

test = temp.join(train, on="id", how="left_anti")
test.cache()

train = train.drop("id", "Random", "Row")
test = test.drop("id", "Random", "Row")

print_class_balance(label_data, "label_data")
print_class_balance(train, "train")
print_class_balance(test, "test")

# :<EOF>
# label_data
# 420620
    # label   count     ratio
# 0    13.0    4000  0.009510
# 1    12.0    5789  0.013763
# 2     6.0   14194  0.033745
# 3     1.0   40666  0.096681
# 4    10.0    6801  0.016169
# 5    16.0    1535  0.003649
# 6    19.0     463  0.001101
# 7     5.0   14314  0.034031
# 8     7.0   11691  0.027795
# 9     4.0   17504  0.041615
# 10   17.0    1012  0.002406
# 11   20.0     200  0.000475
# 12    9.0    6931  0.016478
# 13    8.0    8780  0.020874
# 14   14.0    2067  0.004914
# 15    0.0  237649  0.564997
# 16   18.0     555  0.001319
# 17    3.0   17775  0.042259
# 18    2.0   20899  0.049686
# 19   15.0    1613  0.003835
# 20   11.0    6182  0.014697

# train
# 294422
    # label   count     ratio
# 0     6.0    9935  0.033744
# 1    12.0    4052  0.013763
# 2    13.0    2799  0.009507
# 3     1.0   28466  0.096684
# 4    10.0    4760  0.016167
# 5    16.0    1074  0.003648
# 6    19.0     324  0.001100
# 7     5.0   10019  0.034029
# 8     7.0    8183  0.027793
# 9     4.0   12252  0.041614
# 10   17.0     708  0.002405
# 11   20.0     139  0.000472
# 12    8.0    6145  0.020871
# 13    9.0    4851  0.016476
# 14   14.0    1446  0.004911
# 15    0.0  166354  0.565019
# 16   18.0     388  0.001318
# 17    3.0   12442  0.042259
# 18    2.0   14629  0.049687
# 19   11.0    4327  0.014697
# 20   15.0    1129  0.003835

# test
# 126198
    # label  count     ratio
# 0     6.0   4259  0.033749
# 1    12.0   1737  0.013764
# 2    13.0   1201  0.009517
# 3     1.0  12200  0.096673
# 4    10.0   2041  0.016173
# 5    16.0    461  0.003653
# 6    19.0    139  0.001101
# 7     5.0   4295  0.034034
# 8     7.0   3508  0.027798
# 9     4.0   5252  0.041617
# 10   17.0    304  0.002409
# 11   20.0     61  0.000483
# 12    8.0   2635  0.020880
# 13    9.0   2080  0.016482
# 14   14.0    621  0.004921
# 15    0.0  71295  0.564946
# 16   18.0    167  0.001323
# 17    3.0   5333  0.042259
# 18    2.0   6270  0.049684
# 19   11.0   1855  0.014699
# 20   15.0    484  0.003835

assembler = VectorAssembler(
  inputCols=[col for col in label_data.columns if col.startswith("M")],
  outputCol="Features"
)
scaler = StandardScaler(inputCol="Features", outputCol="ScaledFeatures", withMean = False, withStd = True)
#pca = PCA(k=5, inputCol = "ScaledFeatures", outputCol = "TransformedFeatures")


#Pipeline to set the stages
pipeline = Pipeline(stages = [assembler, scaler])#, pca])
pipelineMClass = pipeline.fit(train)
trainMCData = pipelineMClass.transform(train)
testMCData = pipelineMClass.transform(test)

rf_model = rf.fit(trainMCData)
dt_model = dt.fit(trainMCData)

##class_labels is used for Multiclass metrics
class_labels = (
    label_data 
    .groupBy("label")
    .count()
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.mllib.evaluation import MulticlassMetrics 

                                    
def multiclassMetrics(model, test):
    predictions = model.transform(test)
    predictionsAndLabels = predictions.select(["prediction","label"])
    metrics = MulticlassMetrics(predictionsAndLabels.rdd)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted Recall = %s" % metrics.weightedRecall)
    print("Accuracy =   %s" % metrics.accuracy)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())

    labels = class_labels.select('label').rdd.map(lambda row : row[0]).collect()
    for label in labels:
        print("Class %s precision =  %s" % (label, metrics.precision(label)))
        print("       %s  recall =     %s" % (label, metrics.recall(label)))
        print("       %s  F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    
    confusion_matrix = metrics.confusionMatrix().toArray().astype(int)
    labels = [int(l) for l in metrics.call('labels')]
    confusion_matrix = pd.DataFrame(confusion_matrix , index=labels, columns=labels)
    cm = confusion_matrix.transpose()
    print(cm)
    
multiclassMetrics(rf_model, testMCData) 
multiclassMetrics(dt_model, testMCData) 

# :<EOF>
# Weighted precision = 0.3698748880339072
# Weighted Recall = 0.5707142743942059
# Accuracy =   0.5717919459896353
# Weighted F(1) Score = 0.4272015328818513
# Class 6.0 precision =  0.0
       # 6.0  recall =     0.0
       # 6.0  F1 Measure = 0.0
# Class 13.0 precision =  0.0
       # 13.0  recall =     0.0
       # 13.0  F1 Measure = 0.0
# Class 12.0 precision =  0.0
       # 12.0  recall =     0.0
       # 12.0  F1 Measure = 0.0
# Class 1.0 precision =  0.4789762340036563
       # 1.0  recall =     0.10737704918032787
       # 1.0  F1 Measure = 0.17542684968195515
# Class 10.0 precision =  0.0
       # 10.0  recall =     0.0
       # 10.0  F1 Measure = 0.0
# Class 19.0 precision =  0.0
       # 19.0  recall =     0.0
       # 19.0  F1 Measure = 0.0
# Class 16.0 precision =  0.0
       # 16.0  recall =     0.0
       # 16.0  F1 Measure = 0.0
# Class 5.0 precision =  0.0
       # 5.0  recall =     0.0
       # 5.0  F1 Measure = 0.0
# Class 7.0 precision =  0.0
       # 7.0  recall =     0.0
       # 7.0  F1 Measure = 0.0
# Class 4.0 precision =  0.0
       # 4.0  recall =     0.0
       # 4.0  F1 Measure = 0.0
# Class 17.0 precision =  0.0
       # 17.0  recall =     0.0
       # 17.0  F1 Measure = 0.0
# Class 20.0 precision =  0.0
       # 20.0  recall =     0.0
       # 20.0  F1 Measure = 0.0
# Class 8.0 precision =  0.0
       # 8.0  recall =     0.0
       # 8.0  F1 Measure = 0.0
# Class 9.0 precision =  0.0
       # 9.0  recall =     0.0
       # 9.0  F1 Measure = 0.0
# Class 14.0 precision =  0.0
       # 14.0  recall =     0.0
       # 14.0  F1 Measure = 0.0
# Class 0.0 precision =  0.5727464908515102
       # 0.0  recall =     0.9918367346938776
       # 0.0  F1 Measure = 0.726162725022849
# Class 18.0 precision =  0.0
       # 18.0  recall =     0.0
       # 18.0  F1 Measure = 0.0
# Class 3.0 precision =  0.0
       # 3.0  recall =     0.0
       # 3.0  F1 Measure = 0.0
# Class 2.0 precision =  0.0
       # 2.0  recall =     0.0
       # 2.0  F1 Measure = 0.0
# Class 11.0 precision =  0.0
       # 11.0  recall =     0.0
       # 11.0  F1 Measure = 0.0
# Class 15.0 precision =  0.0
       # 15.0  recall =     0.0
       # 15.0  F1 Measure = 0.0
              # 0      1     2     3     4     5     6     7     8     9     10    11    12    13   14   15   16   17   18   19  20
# 0   70713  10890  5843  5282  5212  4212  4196  3504  2610  1982  2025  1849  1728  1189  619  482  460  302  167  137  61
# 1     582   1310   427    51    40    83    63     4    25    98    16     6     9    12    2    2    1    2    0    2   0
# 2       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 3       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 4       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 5       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 6       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 7       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 8       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 9       0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 10      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 11      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 12      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 13      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 14      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 15      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 16      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 17      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 18      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 19      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 20      0      0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# In [41]: multiclassMetrics(dt_model, testMCData)
# Weighted precision = 0.3937494944770384
# Weighted Recall = 0.5730677189812834
# Weighted F(1) Score = 0.44645412409867025
# Class 6.0 precision =  0.0
       # 6.0  recall =     0.0
       # 6.0  F1 Measure = 0.0
# Class 13.0 precision =  0.0
       # 13.0  recall =     0.0
       # 13.0  F1 Measure = 0.0
# Class 12.0 precision =  0.0
       # 12.0  recall =     0.0
       # 12.0  F1 Measure = 0.0
# Class 1.0 precision =  0.47245980157372564
       # 1.0  recall =     0.11319672131147542
       # 1.0  F1 Measure = 0.18263572042584142
# Class 10.0 precision =  0.0
       # 10.0  recall =     0.0
       # 10.0  F1 Measure = 0.0
# Class 16.0 precision =  0.0
       # 16.0  recall =     0.0
       # 16.0  F1 Measure = 0.0
# Class 19.0 precision =  0.0
       # 19.0  recall =     0.0
       # 19.0  F1 Measure = 0.0
# Class 5.0 precision =  0.0
       # 5.0  recall =     0.0
       # 5.0  F1 Measure = 0.0
# Class 7.0 precision =  0.0
       # 7.0  recall =     0.0
       # 7.0  F1 Measure = 0.0
# Class 4.0 precision =  0.0
       # 4.0  recall =     0.0
       # 4.0  F1 Measure = 0.0
# Class 17.0 precision =  0.0
       # 17.0  recall =     0.0
       # 17.0  F1 Measure = 0.0
# Class 20.0 precision =  0.0
       # 20.0  recall =     0.0
       # 20.0  F1 Measure = 0.0
# Class 8.0 precision =  0.0
       # 8.0  recall =     0.0
       # 8.0  F1 Measure = 0.0
# Class 9.0 precision =  0.0
       # 9.0  recall =     0.0
       # 9.0  F1 Measure = 0.0
# Class 14.0 precision =  0.0
       # 14.0  recall =     0.0
       # 14.0  F1 Measure = 0.0
# Class 0.0 precision =  0.5884737563767363
       # 0.0  recall =     0.9691843747808402
       # 0.0  F1 Measure = 0.732303909619848
# Class 18.0 precision =  0.0
       # 18.0  recall =     0.0
       # 18.0  F1 Measure = 0.0
# Class 3.0 precision =  0.0
       # 3.0  recall =     0.0
       # 3.0  F1 Measure = 0.0
# Class 2.0 precision =  0.31437841530054644
       # 2.0  recall =     0.2936204146730462
       # 2.0  F1 Measure = 0.3036450602012205
# Class 11.0 precision =  0.0
       # 11.0  recall =     0.0
       # 11.0  F1 Measure = 0.0
# Class 15.0 precision =  0.0
       # 15.0  recall =     0.0
       # 15.0  F1 Measure = 0.0
       # 0     1     2     3     4     5     6     7     8     9     10    11    12    13   14   15   16   17   18   19  20
# 0   69098  9461  3973  5233  5003  3921  4026  3499  2555  1693  2010  1843  1718  1180  604  480  458  299  166  139  60
# 1     659  1381   456    61    54    83    59     3    20   102    16     4     8    10    0    3    1    3    0    0   0
# 2    1538  1358  1841    39   195   291   174     6    60   285    15     8    11    11   17    1    2    2    1    0   1
# 3       0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 4       0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 5       0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 6       0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 7       0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 8       0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 9       0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 10      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 11      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 12      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 13      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 14      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 15      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 16      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 17      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 18      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 19      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
# 20      0     0     0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0   0
###Random downsampling(uniform) and upsampling(Poisson)
count_upper_bound = 50000
count_lower_bound = 500


def random_resample(x, class_counts, count_upper_bound, count_lower_bound):
    # Can implement custom sampling logic per class,

    count = class_counts[x]

    if count > count_upper_bound:
        if np.random.rand() < count_upper_bound / count: # keep with probability, randomly downsample to count_upper_bound
            return [x]
        else:
            return []

    if count < count_lower_bound:
        return [x] * int(1 + np.random.poisson((count_lower_bound - count) / count))  # return at least one, randomly upsample to count_lower_bound

    return [x]  # do nothing

random_resample_udf = udf(lambda x: random_resample(x, class_counts, count_upper_bound, count_lower_bound), ArrayType(IntegerType()))
training_resampled = (
    trainMCData
    .withColumn("Sample", random_resample_udf(col("label")))
    .select(
        col("ScaledFeatures"),
        col("label"),
        explode(col("Sample")).alias("Sample")
    )
    .drop("Sample")
)

print_class_balance(training_resampled, "training_resampled")

# training_resampled
# 135522
    # label  count     ratio
# 0     6.0   9935  0.073309
# 1    12.0   4052  0.029899
# 2    13.0   2799  0.020653
# 3     1.0  28466  0.210047
# 4    10.0   4760  0.035123
# 5    16.0   1074  0.007925
# 6    19.0    344  0.002538
# 7     5.0  10019  0.073929
# 8     7.0   8183  0.060381
# 9     4.0  12252  0.090406
# 10   17.0    708  0.005224
# 11   20.0    344  0.002538
# 12    8.0   6145  0.045343
# 13    9.0   4851  0.035795
# 14   14.0   1446  0.010670
# 15    0.0   7408  0.054663
# 16   18.0    388  0.002863
# 17    3.0  12442  0.091808
# 18    2.0  14629  0.107946
# 19   11.0   4327  0.031928
# 20   15.0   1129  0.008331



rf_resampled = rf.fit(training_resampled)
multiclassMetrics(rf_resampled, testMCData) 

# :<EOF>
# Weighted precision = 0.5669163057345212
# Weighted Recall = 0.22307009619803803
# Accuracy =   0.22585936385679647
# Weighted F(1) Score = 0.22398355813360968
# Class 13.0 precision =  0.0
       # 13.0  recall =     0.0
       # 13.0  F1 Measure = 0.0
# Class 12.0 precision =  0.0
       # 12.0  recall =     0.0
       # 12.0  F1 Measure = 0.0
# Class 6.0 precision =  0.0
       # 6.0  recall =     0.0
       # 6.0  F1 Measure = 0.0
# Class 1.0 precision =  0.13989147286821704
       # 1.0  recall =     0.7395901639344262
       # 1.0  F1 Measure = 0.23528031290743154
# Class 10.0 precision =  0.0
       # 10.0  recall =     0.0
       # 10.0  F1 Measure = 0.0
# Class 16.0 precision =  0.0
       # 16.0  recall =     0.0
       # 16.0  F1 Measure = 0.0
# Class 19.0 precision =  0.0
       # 19.0  recall =     0.0
       # 19.0  F1 Measure = 0.0
# Class 5.0 precision =  0.10645249115164715
       # 5.0  recall =     0.0910360884749709
       # 5.0  F1 Measure = 0.0981425702811245
# Class 7.0 precision =  0.07283301024646913
       # 7.0  recall =     0.1499429874572406
       # 7.0  F1 Measure = 0.09804287045666357
# Class 4.0 precision =  0.09548431575318855
       # 4.0  recall =     0.05274181264280274
       # 4.0  F1 Measure = 0.06795044768796761
# Class 17.0 precision =  0.0
       # 17.0  recall =     0.0
       # 17.0  F1 Measure = 0.0
# Class 20.0 precision =  0.0
       # 20.0  recall =     0.0
       # 20.0  F1 Measure = 0.0
# Class 9.0 precision =  0.0
       # 9.0  recall =     0.0
       # 9.0  F1 Measure = 0.0
# Class 8.0 precision =  0.0
       # 8.0  recall =     0.0
       # 8.0  F1 Measure = 0.0
# Class 14.0 precision =  0.0
       # 14.0  recall =     0.0
       # 14.0  F1 Measure = 0.0
# Class 0.0 precision =  0.9332944095752445
       # 0.0  recall =     0.17936741706992076
       # 0.0  F1 Measure = 0.3009047378142758
# Class 18.0 precision =  0.0
       # 18.0  recall =     0.0
       # 18.0  F1 Measure = 0.0
# Class 3.0 precision =  0.11521368941313287
       # 3.0  recall =     0.5201575098443653
       # 3.0  F1 Measure = 0.18864331859911593
# Class 2.0 precision =  0.23431788995357108
       # 2.0  recall =     0.37830940988835726
       # 2.0  F1 Measure = 0.28939181357896665
# Class 11.0 precision =  0.0
       # 11.0  recall =     0.0
       # 11.0  F1 Measure = 0.0
# Class 15.0 precision =  0.0
       # 15.0  recall =     0.0
       # 15.0  F1 Measure = 0.0
       # 0     1     2     3     4     5     6     7     8     9     10    11   12   13   14   15   16   17   18  19  20
# 0   12788   202    50    35   138    65    89    37   155     7    67     9   15   10   13    2    3   16    0   0   1
# 1   35822  9023  3296  1971  2745  1860  2375  1408  1367  1249  1037   501  745  335  276  118  127  106   38  67  34
# 2    3755  1312  2372   108   610   727   315    85   161   466    34    37   37   11   63    5    7    3    2  11   2
# 3   10250  1336   225  2774   837   945  1148  1167   519   143   775  1141  793  827  172  356  305  174  121  47  22
# 4    1862    57    76    52   277   145    79   136   112    28    16    24   22    2    8    0    1    0    0   4   0
# 5    1674   133   163   198   284   391   122   149   157   137    20    89   64    7   61    0   10    2    3   7   2
# 6       0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 7    5144   137    88   195   361   162   131   526   164    50    92    54   61    9   28    3    8    3    3   3   0
# 8       0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 9       0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 10      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 11      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 12      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 13      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 14      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 15      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 16      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 17      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 18      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 19      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0
# 20      0     0     0     0     0     0     0     0     0     0     0     0    0    0    0    0    0    0    0   0   0



dt_resampled = dt.fit(training_resampled)
multiclassMetrics(dt_resampled, testMCData)

# :<EOF>
# Weighted precision = 0.5590414274752618
# Weighted Recall = 0.22531260400323302
# Weighted F(1) Score = 0.23200691955255895
# Class 13.0 precision =  0.0
       # 13.0  recall =     0.0
       # 13.0  F1 Measure = 0.0
# Class 12.0 precision =  0.0
       # 12.0  recall =     0.0
       # 12.0  F1 Measure = 0.0
# Class 6.0 precision =  0.0
       # 6.0  recall =     0.0
       # 6.0  F1 Measure = 0.0
# Class 1.0 precision =  0.14776331163411785
       # 1.0  recall =     0.5804918032786885
       # 1.0  F1 Measure = 0.23556412985630656
# Class 10.0 precision =  0.0
       # 10.0  recall =     0.0
       # 10.0  F1 Measure = 0.0
# Class 19.0 precision =  0.0
       # 19.0  recall =     0.0
       # 19.0  F1 Measure = 0.0
# Class 16.0 precision =  0.0
       # 16.0  recall =     0.0
       # 16.0  F1 Measure = 0.0
# Class 5.0 precision =  0.08413161182840483
       # 5.0  recall =     0.0940628637951106
       # 5.0  F1 Measure = 0.08882049027151807
# Class 7.0 precision =  0.05641905530884134
       # 7.0  recall =     0.15935005701254276
       # 7.0  F1 Measure = 0.08333333333333333
# Class 4.0 precision =  0.10194460813199764
       # 4.0  recall =     0.03293983244478294
       # 4.0  F1 Measure = 0.04979133688300475
# Class 17.0 precision =  0.0
       # 17.0  recall =     0.0
       # 17.0  F1 Measure = 0.0
# Class 20.0 precision =  0.0
       # 20.0  recall =     0.0
       # 20.0  F1 Measure = 0.0
# Class 9.0 precision =  0.0
       # 9.0  recall =     0.0
       # 9.0  F1 Measure = 0.0
# Class 8.0 precision =  0.0
       # 8.0  recall =     0.0
       # 8.0  F1 Measure = 0.0
# Class 14.0 precision =  0.0
       # 14.0  recall =     0.0
       # 14.0  F1 Measure = 0.0
# Class 0.0 precision =  0.9216985597403475
       # 0.0  recall =     0.1911915281576548
       # 0.0  F1 Measure = 0.3166906742251754
# Class 18.0 precision =  0.0
       # 18.0  recall =     0.0
       # 18.0  F1 Measure = 0.0
# Class 3.0 precision =  0.1027333868081693
       # 3.0  recall =     0.6244140258766173
       # 3.0  F1 Measure = 0.17643786261159825
# Class 2.0 precision =  0.22203274215552524
       # 2.0  recall =     0.5191387559808612
       # 2.0  F1 Measure = 0.3110367892976588
# Class 11.0 precision =  0.0
       # 11.0  recall =     0.0
       # 11.0  F1 Measure = 0.0
# Class 15.0 precision =  0.0
       # 15.0  recall =     0.0
       # 15.0  F1 Measure = 0.0
       # 0     1     2     3     4     5     6     7     8    9    10    11   12   13   14   15   16   17   18  19  20
# 0   13631   250    60    56   188    87   115    44   186    5   83    11   19   11   15    3    4   20    0   0   1
# 1   26922  7082  2142  1335  2045  1269  1766  1068  1055  848  777   325  520  279  156   87   81   81   28  43  19
# 2    5208  2464  3255   206   715   858   498   108   196  729   63    59   83   25  120   12   19   14    5  20   3
# 3   14368  1933   463  3330  1275  1397  1537  1478   720  318  983  1350  981  872  280  378  341  182  129  64  35
# 4    1023    75    96    28   173   114    56    36    53   17    4     8   10    0    0    1    1    0    0   2   0
# 5    2539   182   158   195   446   404   139   215   204  110   25    64   63    6   30    0    7    3    2   7   3
# 6       0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 7    7604   214    96   183   410   166   148   559   221   53  106    38   61    8   20    3    8    4    3   3   0
# 8       0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 9       0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 10      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 11      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 12      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 13      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 14      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 15      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 16      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 17      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 18      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 19      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
# 20      0     0     0     0     0     0     0     0     0    0    0     0    0    0    0    0    0    0    0   0   0
