# Mining Arguments in US Presidential Campaign Debates

## Project Description 

This project is dedicated to the problem of Argument Mining and is aimed at reproducing the [paper](https://aclanthology.org/P19-1463/) published by Haddadan et al. in 2019. We dataset is comprised of 39 speeches from the presidential debates in the US from the year 1960 to 2016. The corpus sentences are labels as claims, premises or none of above. 

Like the authors, we solve two binary classification tasks: 1) classification of argumentative vs. non-argumentative sentences; and 2) classification of claims vs. premises. 

The methodology of the project is entirely reflected by the original paper. We split the practical part into four experimental
settings. The splits are based on different methodological approaches. 

**Setting 1**: tf-idf word matrices for unigrams + SVM linear model

**Setting 2**: tf-idf word matrices for uni-, bi- and trigrams and a set of engineered features + SVM with rbf kernel

**Setting 3**: fasttext embeddings + LSTM 

**Setting 4**: tf-idf word matrices for uni-, bi- and trigrams and a set of engineered features + Feed Forward Network

The project is carried out as a part of PM "Opinion and Argument Mining", taught by Prof. Dr. Manfred Stede at the University of Potsdam, Winter Term 2021/22. Our final report can be found [here](https://github.com/pc852/Argument-Mining-Presidential-Debates/blob/main/final-report.pdf). 

*Team: Chen Peng, Mariia Poiaganova, Milena Voskanyan*

## Repository Structure 

|Folder   |Item   |Description   |
|---|---|---|
|:file_folder: **code**   |:file_folder: **experiments**|Notebooks for 4 experimental settings ×  2 tasks |
|   |linguistic_features.py |Documented functions for feature engineering |
|   |preprocessing.py |Code snippet for sentences preprocessing|
|   |requirements.txt   |Necessary Python frameworks and their versions   |
|:file_folder: **data**   |:file_folder: **raw**   |Debates' scripts along with respective annotation files   |
|   |sentence_db_candidate.csv   |Main dataset used   |
|:bookmark_tabs: **paper** ||Final report   |
|:file_folder: **presentations**   |dec-presentation.pdf   |Slides for the first presentation |
|   |feb-presentation.pdf   |Slides for the final presentation   |

## Replication 

:small_blue_diamond: To reproduce the project, start by cloning the repository: 

`git clone https://github.com/pc852/Argument-Mining-Presidential-Debates.git`

:small_blue_diamond: Next, we recommend running the experiments from Jupyter Notebook environment. If you are new to it, nice guideline can be found [here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).

:small_blue_diamond: You will need `Python` version 3.7 or higher, and the following packages: 

`pandas`, `scikit-learn`, `nltk`, `keras`, `tensorflow`, `gensim`, `numpy`, `spacy`, `vadersentiment`. Our versions for them can be found [here](https://github.com/pc852/Argument-Mining-Presidential-Debates/blob/main/code/requirements.txt).

:small_blue_diamond: Each notebook is tailored to be run independently, however, you might need some additional preparations for certain experimental settings. 

**For Settings 2 and 4, with engineered features**

In order to compile Part-of-Speech and NER engineering functions, you should download the [model](https://spacy.io/usage) `en_core_web_sm` to your system. 

**For Setting 3, with LSTM model**

You should download the pre-trained fasttext embeddings file `wiki-news-300d-1M.vec` and move it to project folder. [FastText](https://fasttext.cc/docs/en/english-vectors.html)


:white_circle: [Dataset](https://github.com/ElecDeb60To16/Dataset) is kept here, but for all the experiments you will need only [this](https://github.com/pc852/Argument-Mining-Presidential-Debates/blob/main/data/sentence_db_candidate.csv) file.




