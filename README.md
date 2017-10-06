# Overview of Project

Based on text in public earnings releases, Next Agenda would like to predict future stock performance.  For the purposes of an initial exploration, we bucketed the future stock performance as percentage change in stock price in 4 weeks from the price on the publication date.  To simplify our exploration further, we split the performance into three equal bins – 0,1,2 – representing low performance, middle performance and high performance.  Each bin contains one third of the population.    And as a final step, to narrow the subject matter, we modeled just on data from one industry, the biotechnology industry.

We used a 1D CNN in Keras, using word embeddings from the GloVe vector model published by Stanford.  (all links in code).  For purposes of initial exploration, we used the smallest word vector model, with 400,000 words.  We applied 39% dropout rate, and used the RMSProp optimizer with the Xavier initializer.  We took the documented Keras approach for working with custom word embeddings.  

Our output suggests there is some signal available to get above chance classification of future performance.  After 17 epochs of training, our model delivers 49.8% accuracy with a test score of 1.04.  While the sample is not enormous (943 total samples, training on 724 and testing on 219), this suggests more investigations are merited to determine if more sample, or alternative approaches can improve the signal.  

As next steps, the partner will experiment with LSTM and to incorporate sequential information in the release dates.  They will aslo replicate this model for different industries and across all industries together to determine if the signal persists.   

# Azure ML Workbench
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


