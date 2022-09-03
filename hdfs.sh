[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -ls /data/ml
Found 9 items
# -rw-r--r--   8 jsw93 supergroup       9777 2021-09-20 10:45 /data/ml/README.txt
# drwxr-xr-x   - jsw93 supergroup          0 2021-09-20 10:45 /data/ml/api
# -rw-r--r--   8 jsw93 supergroup   55998374 2021-09-20 10:45 /data/ml/attributes.json
# -rw-r--r--   8 jsw93 supergroup  344861061 2021-09-20 10:45 /data/ml/genome-scores.csv
# -rw-r--r--   8 jsw93 supergroup      18103 2021-09-20 10:45 /data/ml/genome-tags.csv                                    
# -rw-r--r--   8 jsw93 supergroup     989107 2021-09-20 10:45 /data/ml/links.csv                                          
# -rw-r--r--   8 jsw93 supergroup    2283410 2021-09-20 10:45 /data/ml/movies.csv                                         
# -rw-r--r--   8 jsw93 supergroup  709550327 2021-09-20 10:45 /data/ml/ratings.csv
# -rw-r--r--   8 jsw93 supergroup   27113729 2021-09-20 10:45 /data/ml/tags.csv

[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -ls -h /data/ml
Found 9 items
# -rw-r--r--   8 jsw93 supergroup      9.5 K 2021-09-20 10:45 /data/ml/README.txt                                         
# drwxr-xr-x   - jsw93 supergroup          0 2021-09-20 10:45 /data/ml/api                                                
# -rw-r--r--   8 jsw93 supergroup     53.4 M 2021-09-20 10:45 /data/ml/attributes.json                                    
# -rw-r--r--   8 jsw93 supergroup    328.9 M 2021-09-20 10:45 /data/ml/genome-scores.csv                                  
# -rw-r--r--   8 jsw93 supergroup     17.7 K 2021-09-20 10:45 /data/ml/genome-tags.csv                                    
# -rw-r--r--   8 jsw93 supergroup    965.9 K 2021-09-20 10:45 /data/ml/links.csv                                          
# -rw-r--r--   8 jsw93 supergroup      2.2 M 2021-09-20 10:45 /data/ml/movies.csv                                         
# -rw-r--r--   8 jsw93 supergroup    676.7 M 2021-09-20 10:45 /data/ml/ratings.csv                                        
# -rw-r--r--   8 jsw93 supergroup     25.9 M 2021-09-20 10:45 /data/ml/tags.csv                                           

[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$   



[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -ls /data/msd/audio/attributes/                                                                                                            
# Found 13 items                                                                                                                                                                                  
# -rw-r--r--   8 jsw93 supergroup       1051 2021-09-29 10:35 /data/msd/audio/attributes/msd-jmir-area-of-moments-all-v1.0.attributes.csv                                                         
# -rw-r--r--   8 jsw93 supergroup        671 2021-09-29 10:35 /data/msd/audio/attributes/msd-jmir-lpc-all-v1.0.attributes.csv                                                                     
# -rw-r--r--   8 jsw93 supergroup        484 2021-09-29 10:35 /data/msd/audio/attributes/msd-jmir-methods-of-moments-all-v1.0.attributes.csv                                                      
# -rw-r--r--   8 jsw93 supergroup        898 2021-09-29 10:35 /data/msd/audio/attributes/msd-jmir-mfcc-all-v1.0.attributes.csv                                                                    
# -rw-r--r--   8 jsw93 supergroup        777 2021-09-29 10:35 /data/msd/audio/attributes/msd-jmir-spectral-all-all-v1.0.attributes.csv                                                            
# -rw-r--r--   8 jsw93 supergroup        777 2021-09-29 10:35 /data/msd/audio/attributes/msd-jmir-spectral-derivatives-all-all-v1.0.attributes.csv                                                
# -rw-r--r--   8 jsw93 supergroup      12317 2021-09-29 10:35 /data/msd/audio/attributes/msd-marsyas-timbral-v1.0.attributes.csv                                                                  
# -rw-r--r--   8 jsw93 supergroup       9990 2021-09-29 10:35 /data/msd/audio/attributes/msd-mvd-v1.0.attributes.csv                                                                              
# -rw-r--r--   8 jsw93 supergroup       1390 2021-09-29 10:35 /data/msd/audio/attributes/msd-rh-v1.0.attributes.csv                                                                               
# -rw-r--r--   8 jsw93 supergroup      34913 2021-09-29 10:35 /data/msd/audio/attributes/msd-rp-v1.0.attributes.csv                                                                               
# -rw-r--r--   8 jsw93 supergroup       3942 2021-09-29 10:35 /data/msd/audio/attributes/msd-ssd-v1.0.attributes.csv                                                                              
# -rw-r--r--   8 jsw93 supergroup       9990 2021-09-29 10:35 /data/msd/audio/attributes/msd-trh-v1.0.attributes.csv                                                                              
# -rw-r--r--   8 jsw93 supergroup      28313 2021-09-29 10:35 /data/msd/audio/attributes/msd-tssd-v1.0.attributes.csv  

[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -cat "/data/msd/audio/attributes/*" | awk -F',' '{print $2}' | sort | uniq                                                                 
# NUMERIC                                                                                                                                                                                         
# real                                                                                                                                                                                            
# real                                                                                                                                                                                            
# string                                                                                                                                                                                          
# string                                                                                                                                                                                          
# STRING   


[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -ls /data/msd/genre
# Found 3 items
# -rw-r--r--   8 jsw93 supergroup   11625230 2021-09-29 10:35 /data/msd/genre/msd-MAGD-genreAssignment.tsv
# -rw-r--r--   8 jsw93 supergroup    8820054 2021-09-29 10:35 /data/msd/genre/msd-MASD-styleAssignment.tsv
# -rw-r--r--   8 jsw93 supergroup   11140605 2021-09-29 10:35 /data/msd/genre/msd-topMAGD-genreAssignment.tsv

[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -ls -R /data/msd > files.txt
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd
# SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 12.3 G   98.1 G                                 /data/msd/audio
# 30.1 M   241.0 M                                /data/msd/genre
# 174.4 M  1.4 G                                  /data/msd/main
# 490.4 M  3.8 G                                  /data/msd/tasteprofile
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio
# SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 103.0 K  824.3 K                                /data/msd/audio/attributes
# 12.2 G   97.8 G                                 /data/msd/audio/features
# 40.3 M   322.1 M                                /data/msd/audio/statistics
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/genre
# SIZE    DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 11.1 M  88.7 M                                 /data/msd/genre/msd-MAGD-genreAssignment.tsv
# 8.4 M   67.3 M                                 /data/msd/genre/msd-MASD-styleAssignment.tsv
# 10.6 M  85.0 M                                 /data/msd/genre/msd-topMAGD-genreAssignment.tsv
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/main
# SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 174.4 M  1.4 G                                  /data/msd/main/summary
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/tasteprofile
# SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 2.0 M    16.2 M                                 /data/msd/tasteprofile/mismatches
# 488.4 M  3.8 G                                  /data/msd/tasteprofile/triplets.tsv
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ for i in $(hdfs dfs -find /data/msd/audio/attributes -name '*.*')
> do
>   (hdfs dfs -cat $i | wc -l)
> done
# 21
# 21
# 11
# 27
# 17
# 17
# 125
# 421
# 61
# 1441
# 169
# 421
# 1177
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ for i in $(hdfs dfs -find /data/msd/audio/features -name 'msd-*')
> do
>    (hdfs dfs -cat $i/* | gunzip | wc -l)
> done
# 994623
# 994623
# 994623
# 994623
# 994623
# 994623
# 995001
# 994188
# 994188
# 994188
# 994188
# 994188
# 994188


[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$  /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00001.csv.gz
find: ‘/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00001.csv.gz’: No such file or directory
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00001.csv.gz
# SIZE   DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 8.2 M  65.9 M                                 /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00001.csv.gz
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00002.csv.gz
# SIZE   DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 8.2 M  65.9 M                                 /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00002.csv.gz
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00003.csv.gz
# SIZE   DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 8.2 M  65.9 M                                 /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00003.csv.gz
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00004.csv.gz
# SIZE   DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 8.2 M  65.9 M                                 /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00004.csv.gz
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00005.csv.gz
# SIZE   DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 8.2 M  65.9 M                                 /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00005.csv.gz
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00006.csv.gz
# SIZE   DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 8.2 M  65.9 M                                 /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00006.csv.gz
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -du -h -v /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00007.csv.gz
# SIZE   DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
# 7.9 M  63.0 M                                 /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00007.csv.gz


[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ for i in $(hdfs dfs -find /data/msd/genre -name '*.*'); do    (hdfs dfs -cat $i | wc -l); done
# 422714
# 273936
# 406427
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ for i in $(hdfs dfs -find /data/msd/main/summary -name '*.*'); do    (hdfs dfs -cat $i | wc -l); done
# 239762
# 480546
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ for i in $(hdfs dfs -find /data/msd/tasteprofile/mismatches -name '*.*'); do   (hdfs dfs -cat $i | wc -l); done
# 938
# 19094
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -cat /data/msd/tasteprofile/triplets.tsv/* | gunzip | wc -l
# 48373586


[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -ls /user/bpa78/outputs/msd
Found 6 items
drwxr-xr-x   - bpa78 bpa78          0 2022-06-02 12:52 /user/bpa78/outputs/msd/audioSimFinalData
drwxr-xr-x   - bpa78 bpa78          0 2022-05-21 00:12 /user/bpa78/outputs/msd/audioSimilarity
drwxr-xr-x   - bpa78 bpa78          0 2022-06-02 12:51 /user/bpa78/outputs/msd/mismatches
drwxr-xr-x   - bpa78 bpa78          0 2022-06-07 21:35 /user/bpa78/outputs/msd/songPopularity
drwxr-xr-x   - bpa78 bpa78          0 2022-05-23 00:34 /user/bpa78/outputs/msd/triplets
drwxr-xr-x   - bpa78 bpa78          0 2022-06-07 21:42 /user/bpa78/outputs/msd/userActivity
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -copyToLocal hdfs:///user/bpa78/outputs/msd/songPopularity/ /users/home/bpa78
[bpa78@canterbury.ac.nz@mathmadslinux2p ~]$ hdfs dfs -copyToLocal hdfs:///user/bpa78/outputs/msd/userActivity/ /users/home/bpa78