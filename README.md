
# Stock Performance Classification with a 1D CNN, Keras and Azure ML Workbench

## Overview
We recently worked with a financial services partner to develop a model to predict the future stock market performance of public companies in categories where they invest. The goal was to use select text narrative sections from publicly available earnings release documents to predict and alert their analytics to investment opportunities and risks. We developed a deep learning model using a one-dimensional convolutional neural network (a 1D CNN) based on text extracted from public financial statements from these companies to make these predictions. We used Azure Machine Learning Workbench to explore the data and develop the model. We modeled using the Keras deep learning Python framework with a Theano backend. The results demonstrate how a deep learning model trained on text in earnings releases and other sources could provide a valuable signal to the investment decision maker.

This initial result suggests that that deep learning models trained on text in earnings releases and other sources could prove a viable mechanism to improve the quality of the information available to the person making the investment decision, particularly in avoiding investment losses.  While the model needs to be improved with more samples, refinements of domain-specific vocabulary and text augmentation, this model suggests that providing this signal as another decision input for investmennt analyst would improve the efficiency of the firm’s analysis work.

The history of model training and testing is below, trained for 24 epochs.
![alt text](https://github.com/SingingData/StockPerformanceClassification/blob/master/images/Model_Training_accuracyandloss.png)
 
## Model Architecture
This graphic depicts the model architecture.
![alt text](https://github.com/SingingData/StockPerformanceClassification/blob/master/images/modelarchitecture.png)
 
This graphic gives you a summary of this model at each level.  
![alt text](https://github.com/SingingData/StockPerformanceClassification/blob/master/images/ModelSummary.png)

## Azure ML Workbench
We built the solution on the Azure ML Workbench python environment.  We found the following installs and upgrades were required.

Installs and Upgrades Required (Order is Important)
- conda install mkl-service
- conda install m2w64-toolchain
- conda install -c anaconda libpython=2.0
- pip install numpy==1.13.0
- pip install keras #Remember to change back0end to theano in the backend folder w/in keras, init file
- pip install theano
- pip install gensim
- pip install nltk
- pip install cloudpickle
- pip install --upgrade pandas
- pip install --upgrade gensim
- pip install importio
- pip install matplotlib
- pip install netwrokx
- pip install h5py
- pip install pyparsing==1.5.7
- pip install graphviz
- Install graphviz binaries linked in this issue https://stackoverflow.com/questions/36886711/keras-runtimeerror-failed-to-import-pydot-after-installing-graphviz-and-pyd
- pip install pydotplus
- In vis_utils.py comment out def _check_pydot(): lines 19 through 28


