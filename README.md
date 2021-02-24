# Data Mining
This repository was created as part of the Data Mining course of the [Computer Science master’s program](https://www.th-koeln.de/en/academics/computer-science-masters-program_8263.php) at [TH Köln](https://www.th-koeln.de/en/).

## Description
This repository examines Kaggle's [Trending YouTube Video Statistics](https://www.kaggle.com/datasnaek/youtube-new) data set. For this purpose, the data is analyzed and, using various given and specially derived attributes, the period of time that a video needs to go trending after publication is predicted. The data set only contains videos that have actually been trending. Further information can be found in the [Business Understanding](./1_business_understanding/business_understanding.md) section.

The evaluation is primarily limited to the data relating to Germany. In the [evaluation section](./5_evaluation/) there is also a comparison with selected other regions.
Various algorithms are used for the predictions, whereby classifiers are in the foreground.

## Structure of this repository
- [``0_data/``](./0_data/)\
Contains the data processed in the project.
- [``1_business_understanding/``](./1_business_understanding/)\
Information about the [business understanding](./1_business_understanding/business_understanding.md).
- [``2_data_understanding/``](./2_data_understanding/)\
Information about the [data understanding](./2_data_understanding/data_understanding.md), as well as notebooks for the exploration and visual processing of the data.
- [``3_data_preparation/``](./3_data_preperation/)\
Notebooks to prepare the data for the modeling phase.
- [``4_modeling/``](./4_modeling/)\
Notebooks and scripts for the application of prediction models, as well as the optimization of the feature set and the model parameters.
- [``5_evaluation/``](./5_evaluation/)\
Notebooks for working out acquired insights and pecularities, as well as the visual representation of these.

## Requirements
**Programs**:
- [Python](https://www.python.org/) ``3.x``
- [Jupyter](https://jupyter.org/)

**Additional Python packages**:
- numpy
- pandas
- scipy
- mathplotlib
- seaborn
- pycountry
- sklearn
- xgboost

*For a reproducible environment:*
```
pip install -r requirements.txt
```