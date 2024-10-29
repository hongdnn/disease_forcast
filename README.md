# Disease Forecasting Project

## Overview

This Python project aims to forecast diseases based on patient information, illness types, geographical location, and temporal factors. 
The model utilizes machine learning techniques to analyze historical patient data and predict potential disease outbreaks, providing valuable insights for public health initiatives.

## Setup

Create/update environment contains dependencies using these commands:  
`conda env create -f environment.yml`  
`conda env update --file environment.yml --name dfenv`    

Run `train_model.py` to train AI model using train dataset.  
Run `handle_predict_data.py` to transform the list of patients data into ideal test data and export to csv file.  
Run `predict_model` to forcast disease using test data.   
