# Permutation Importance
Model agnostic feature importance implemented using pandas, and numpy. Can be used for any Sklearn API like model: Sklearn models, Xgboost, LightGBM, Catboost and even Keras. The only requirements are pandas and numpy. 

Permutation importance have several adventages over traditional feature importance based on number 
of splits in the trees for Tree based models:

* It is more intuitive since explains the affects on target variable directly. 
* It is more stable over diffirent splits of the dataset. 
* It is model agnostic, meaning one can use it to get feature importance for any Sklearn API 
    like model including Keras based neural networks.
* It is quick since it uses inference time of the model and not its training time. 
