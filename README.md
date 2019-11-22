# Permutation Importance
Model agnostic feature importance implemented using pandas, and numpy. It can be used for any Sklearn API like model: Sklearn models, Xgboost, LightGBM, Catboost and even Keras. The only requirements are pandas and numpy.

Permutation importance has several advantages over traditional feature importance based on the number of splits in the trees for Tree-based models:

* It is more intuitive since explains the effects on the target variable directly.
* It is more stable over different splits of the dataset.
* It is model agnostic, meaning one can use it to get feature importance for any Sklearn API like model including Keras based neural networks.
* It is quick since it uses the inference time of the model and not its training time.
