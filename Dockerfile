FROM jupyter/datascience-notebook 

RUN conda update seaborn && conda update -n base conda

