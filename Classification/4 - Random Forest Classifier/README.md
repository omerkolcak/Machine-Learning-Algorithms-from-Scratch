# Random Forest Classifier
## Resources
* https://www.youtube.com/watch?v=J4Wdy0Wc_xQ (Random Forest)
* https://www.youtube.com/watch?v=Xz0x-8-cgaQ (Bootstraping)
## Algorithm
Random forest algoirthm is an ensemble technique. Ensemble technique simply means combining different machine learning models. There are 2 categories of ensemble technique
as bagging and boosting. Random forest is considered as bagging ensemble technique where bunch of decision trees are trained on bootsraped datasets.
### Bagging
Bagging is also called as boostrap aggregation, because it involves both bootstraping and aggregation. For bagging technique, aim is the reducing the high variance.
#### Bootstraping
Bootstraping operation is randomly selecting data and allowing duplicate values. Also, it is called as sampling with replacement. If you have m number of sample on your
dataset, after the bootstraping you still have m number of sample on your bootstraped dataset with some duplicate samples.
#### Aggregation
If we think for random forest classifier, we train bunch of decision trees on bootsraped datasets. When it comes to prediction, all trees individually make predicitons
and the output is decided by the majority. This is called as aggregation. Predicions are made based on weighted majority
### Boosting
Even though the random forest is a bagging ensemble technique, having a rudimentary idea of boosting technique is beneficial. In boosting technique, different base 
learners are trained on the dataset where missclassified samples are highly prioritized. Aim is the reducing the high bias.
## Random Forest Steps
* Create bootstraped datasets
* Train each decision trees on bootstraped dataset by considering random subset of features (independent variables)
* Make predicitons by majority voting (aggregation)

Training decision trees on bootstraped datasets by considering random subset of features results wide variety of trees. Having the wide variety of trees, and combining
them is the power of random forest algoirthm.
