# 274481
TITLE: 
Credit Prediction

MEMBERS: 
Alberto de Leo (274481), Alessia Pazzaglia (269441), Matteo Policastro (276761)

DESCRIPTION: 
The aim of this project is to divide the users of the greatest financial company in the world into three credit score brackets: Poor, Standard and Good. This company has collected bank details and credit-related information from all over the globe, so our goal is to help it developing targeted products for each group.

APPROACHES:
The first things we should do is the "data engineering and labeling". 
We can assume that all the data are usable (we have the license to use and study them), reliable, sizeable (enough for our scope) and accessible. 
We also have to take care of how much noisy these data are, so we have to clean them by removing all the outliers (that are extreme values that differ from most other data points in our dataset) and the null\not available values (which will be replaced using the "Golden Rule").
Then we should give a look to our data, in order to see how values are correlated, their statistical information (mean, standard deviation etc.) in order to have a better understanding of the problem.
After having done the EDA, we can now start analysing all the possibilities that we have in order to solve the problem. Since we have to predict a target it is a classification problem. 
All the methods we can use are:
-	KNN
-	CART
-	Random Forest
-	Logistic Regression
-	Kernel Svc
-	Artificial Neural Network

We can not consider Logistic Regression, as between our data there is not a linar relationship. But in order to decide which one is better we should analyse all the features of each model (Performance, Metric, Interpretability, Tuning Parameters).

KNN has not a good performance as it is slower as it must compute the neighbours and all the distances between the data and the neighbours, metric and interpretability are good as in order to understand the output we should just look at its neighbirs. Tuning is awesome as we do not need to tune anything exceot for the number of neighbors.

CART has a good performance if the dataset is small which means that is a lot fast. It has a good metric and a fantastic interpretability as the only thing to do is to follow the splits and the output will be clear. Tuning is not so good, as we need to take care of a lot of parameters in order to have good metrics.

RANDOM FOREST, it seems that all the parameters that we are using are not so good as it is very slow as it has to prepare a lot of carts using a random method. Moreover, due to this fact, it is not even interpretable as the output will be the agglomeration of more decision trees. Unlikely in this model we have a lot to tune, as the depth of the tree for example, but this leads to an accurate model, this is why metrics are very good.

KERNEL SVC seems to be good mostly on the tuning as we do not really need to tune anything. On the other hand, it is not interpretable as the output comes from a formula, this is why we cannot understand it. We can also notice that this model has good metrics and it is really quick.

ARTIFICIAL NEURAL NETWORK are very slow as we need to let the method calculate the weights in the hidden layers and it is also not interpretable as we do not know the way the outputs are formed and why it is the output. We also need to tune some things, as the inputs, how many hidden layers, so it is not as good. Metric is really good as thanks to back-propagation the model adjusts the weights and the errors trying to minimize them. Interpretability is really really bad as we do not know how the weights are adjusted and why, so the final output cannot be explained in a clear way.

After having noticed all this, we can state that we should use as the aim of our problem is to be accurate and possibily in the less time possible:
-	CART
-	KERNEL
-	KNN

Then we started with the splitting between training, validation and test set. We have to split data in three parts:
-	Training, used to train the model. It is the set of data that is used to train and make the model learn the hidden features/patterns in the data. 70-80%
-	Validation, to tune the model. Not seen by the algorithm, used to estimate the performance of different algorithms. The validation set is a set of data, separate from the training set, that is used to validate our model performance during training. This validation process gives information that helps us tune the model’s hyperparameters and configurations accordingly. It is like a critic telling us whether the training is moving in the right direction or not. 10-15%
-	Test, to test its performance. Not seen by the algorithm, it evaluates the performance of the model. 10-15%
But partitioning is not that easy, as we cannot always use the randomization, in some cases we need the group partitioning in order not to train on the test or validation data. REMEMBER TO NEVER TRAIN ON THE TEST DATA!!!
Then, another important step that we should not forget is: Featuring! It is the process of transforming a raw example into a feature vector:
-	One-hot encoding, in order to convert categorical data into numerical values, do in not giving importance to data and also excluding dummy variables
-	Featuring scaling, bring every data to the same scale 

EXPERIMENTAL DESIGN:


