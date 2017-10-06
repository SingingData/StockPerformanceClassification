 
# StockPerformanceClassification
Keras 1D CNN on Azure ML Workbench to classify 4 week stock performance based on text in public earnings statements

Installs and Upgrades Required (Order is Important)
- installs and upgrades
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


