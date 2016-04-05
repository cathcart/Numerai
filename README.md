#Numerai
My best scoring submission to [Numerai](http://www.numer.ai) round 4, a data science tournament for stock predictions.
## Installation
```bash
pip install -r requirements.txt
```
## Results
The model produces a [log loss score](http://www.kaggle.com/wiki/LogarithmicLoss) of 0.691043 +/- 0.000324. This is calculted from the mean of 10 [k-folds](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html) splits of the training data. The estimated error is given by the [standard error.](https://en.wikipedia.org/wiki/Standard_error)

When submitted the model initially scored 0.69068, which later rose to 0.69184 with the close of the tournament.
