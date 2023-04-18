# ChelseaYeh_2023_dcard_ml_intern_homework

本專案旨在利用文章發佈後前 6 小時的狀態及基本資訊，預測 24 小時後的愛心數。
其中包含三個模組，分別表示本作業流程，其用途及使用順序如下：

1. EDA
2. Model
3. Web Crawler

## Requirements

python >= 3.9

```
> cd path_to_project/
> pip install src/requirements.txt
```

## EDA

see `src/EDA/EDA.ipynb`

## Model

### Original

報告中 Original 的模型，使用了大部分的特徵

#### Usage

```
> cd path_to_project/
> python main_init.py

MLR (1): 153.65%
KNN (1): 105.29%
```

### Redesign Model (Final)

#### Usage example

```
> cd path_to_project/
> python main.py

MLR (1): 81.55%
MLR (2): 45.23%
MLR (3): 57.23%
MLR (4): 42.96%
KNN (1): 57.37%
KNN (2): 36.16%
KNN (3): 44.18%
KNN (4): 70.04%
```

## Web Crawler

### Step 1.

使用 google search 及 dcard 的搜尋欄，搜尋 title 與 created_at 符合 dataset 的文章

#### Usage example

```

> cd path_to_project/src/crawler/
> python crawler.py -h

usage: crawler.py [-h] [-p PATH]

Dcard Crawler

options:
-h, --help show this help message and exit
-p PATH, --path PATH path to dataset (default: ../../dataset/)

```

若使用預設值，會將爬到的文章 html 分別存入

```

dataset/private_test_raw
dataset/public_test_raw
dataset/train_raw

```

### Step 2.

將 html 處理成 csv 檔

#### Usage

```

> cd path_to_project/src/crawler/
> python rawprocess.py -h
> process the raw data from dcard crawler and save to csv

options:
-h, --help show this help message and exit
-p PATH, --path PATH path to dataset (default: ../../dataset/)

```

若使用預設值會將 csv 存在

```

dataset/crawler_public_test_dataset.csv
dataset/crawler_private_test_dataset.csv
dataset/crawler_train_dataset.csv

```
