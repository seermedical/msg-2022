# Get Training Data

- [Go back to Main](README.md)

## Download Training Data

**Note**, the data is quite large. In total it is about `131 GB` when compressed  (`200 GB` uncompressed)

You can either download and extract the data manually, or use commands in the terminal to automate this process.


### Option 1. Download Manually

The training data can be downloaded using the following links. You can download them locally and unzip them into a `train` subdirectory.


- [https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1110_train.zip](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1110_train.zip) `20.3 GB` (`31.2 GB` unzipped )
- [https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1869_train.zip](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1869_train.zip) `16.6 GB` (`26.2 GB` unzipped)
- [https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1876_train.zip](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1876_train.zip) `20.4 GB` (`30.7 GB` unzipped)
- [https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1904_train.zip](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1904_train.zip) `32.7 GB` (`50.1 GB` unzipped)
- [https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1965_train.zip](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1965_train.zip) `29.3 GB` (`43 GB` unzipped)
- [https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/2002_train.zip](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/2002_train.zip) `12.2 GB` (`18.9 GB` unzipped)
- [https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/train_labels.csv](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/train_labels.csv) `5.3 MB`


### Option 2. Use Terminal Commands

```bash
# --------------------------------------------
# DOWNLOAD, EXTRACT, AND DELETE EACH ZIP FILE
# --------------------------------------------
cd /path/to/data_dir/

# Use "wget -c" instead of "curl -OC -" if you have wget installed instead of curl.
curl -OC - https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1110_train.zip &&\
    unzip 1110_train.zip "*/train/*" -d train  &&\
    mv train/1110_train/data/parquet/train/* train  &&\
    rm -r train/1110_train &&\
    rm 1110_train.zip

curl -OC - https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1869_train.zip &&\
    unzip 1869_train.zip "*/train/*" -d train  &&\
    mv train/1869_train/data/parquet/train/* train  &&\
    rm -r train/1869_train &&\
    rm 1869_train.zip

curl -OC - https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1876_train.zip &&\
    unzip 1876_train.zip "*/train/*" -d train  &&\
    mv train/1876_train/data/parquet/train/* train  &&\
    rm -r train/1876_train &&\
    rm 1876_train.zip

curl -OC - https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1904_train.zip &&\
    unzip 1904_train.zip "*/train/*" -d train  &&\
    mv train/1904_train/data/parquet/train/* train  &&\
    rm -r train/1904_train &&\
    rm 1904_train.zip

curl -OC - https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/1965_train.zip &&\
    unzip 1965_train.zip "*/train/*" -d train  &&\
    mv train/1965_train/data/parquet/train/* train  &&\
    rm -r train/1965_train &&\
    rm 1965_train.zip

curl -OC - https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/2002_train.zip &&\
    unzip 2002_train.zip "*/train/*" -d train  &&\
    mv train/2002_train/data/parquet/train/* train  &&\
    rm -r train/2002_train &&\
    rm 2002_train.zip

# DOWNLOAD TRAIN LABELS
curl -OC - https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/train_labels.csv
```

## Expected Directory structure

After downloading and extracting all the files, the file structure should follow the pattern as shown in the [directory structure page](directory_structure.md).



