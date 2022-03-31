# Mining Arguments in US Presidential Campaign Debates

## Project Description 

[Paper](https://aclanthology.org/P19-1463/)

## Repository Structure 

|Folder   |Item   |Description   |
|---|---|---|
|:file_folder: **code**   |:file_folder: **experiments**|Notebooks for 4 experimental settings ×  2 tasks |
|   |linguistic_features.py |Documented functions for feature engineering |
|   |preprocessing.py |Code snippet for sentences preprocessing|
|   |requirements.txt   |Necessary Python frameworks and their versions   |
|:file_folder: **data**   |:file_folder: **raw**   |Debates' scripts along with respective annotation files   |
|   |sentence_db_candidate.csv   |Main dataset used   |
|:file_folder: **paper**   |pdf   |Final report   |
|   |tex   |Tex code for the final report   |
|:file_folder: **presentations**   |dec-presentation.pdf   |Slides for the first presentation |
|   |feb-presentation.pdf   |Slides for the final presentation   |

## Replication 

To reproduce the project, start by cloning the repository: 
`$ git clone https://github.com/pc852/Argument-Mining-Presidential-Debates.git`

Next, we recommend running the experiments from Jupyter Notebook environment. 

To replicate the project, you will need `Python` version 3.7 or higher, and the following packages: `pandas`, `scikit-learn`, `nltk`, `keras`, `tensorflow`, `gensim`, `numpy`, `spacy`, `vadersentiment`. Our versions for them can be found [here](https://github.com/pc852/Argument-Mining-Presidential-Debates/blob/main/code/requirements.txt).

Each notebook is tailored to be run independently when needed, however, you might need some additional preparations for certain experimental settings. 



[FastText](https://fasttext.cc/docs/en/english-vectors.html)

[Dataset](https://github.com/ElecDeb60To16/Dataset) is kept here, but for all the experiments you will need only [this](https://github.com/pc852/Argument-Mining-Presidential-Debates/blob/main/data/sentence_db_candidate.csv) file.




