# a3barebones



Implemented naive Bayes, assuming a Gaussian distribution on each of the features under the class “GaussianNB”. It gives an accuracy of around 80.0 units for the dataset with 9 features and 100,000 samples.
 

On including the columns of ones;
When standard deviation becomes less than .0001 and the difference between the number and mean becomes less than .001, such a scenario leads to “divide by Zero error” causing the probability to be NaN (Not a number). Following code in the function “calculateprob” handles this error by returning 1 appropriately.
if stdev < 1e-3:
if math.fabs(x-mean) < 1e-2:
return 1.0
else:
return 0
This is not the case when we don’t include the column of ones.

B) Logistic regression:

Implemented Logistic regression in the iterative weight update rule in class “LogitReg”
It gives an accuracy of around 81.667 units for the dataset with 9 features and 100,000 samples.

 

C) Implemented a neural network with a single hidden layer in class “NeuralNet”
It gives an accuracy of around 68.667 units for the dataset with 9 features and 100,000 samples.
 


d) 

Naïve bayes: 
For the features (x) and the label y, it calculates a joint probability from the training data. It assumes all the features are conditionally independent and set theirs weights independently. In case some of the features are dependent on each other, the prediction might be poor. Works well even with less training data, because of the joint density function. When the training data size is less relative to the features, data on prior probabilities make things easier. Training has no optimization step.  You just calculate a count table for each feature. 

On multiple runs of NB my accuracy was always around 70. There wasn’t steep fall or raise in the accuracy for the dataset.    

In logistic regression, 
It estimates the probability(y/x) directly from the training data by minimizing error. Weights are set together such that the linear decision function for positive classes is high and for negative classes low.   It splits feature space linearly, it works well even if some of the variables are correlated. With the small training data, model estimates may over fit the data. 
Logistic regression outperforms Naïve bayes in terms of accuracy. Accuracy for LR is always better than NB.
 
 

Neural network are slow to converge and hard to set parameters but if done with care it work wells.
They underperformed in accuracy when compared to LR or NB. There accuracy was around 68 when LR gave around 77 and NB was around 75.  
  

Tuning NN is a very difficult task. May b this would be much more easy If we had prior scant knowledge about the dataset.


Implemented linear classifier with iterative weight update rule in class “modifiedLinear”. 
Here we have closed-form solution to ∇ll(w), Hence Hessian isn’t needed, which makes things much easy.  On multiple runs of the class, I get an accuracy of around 74
 
   
	Implemented LR with L1 and L2 regularization for a new DataSet “MADELON “ in class “LogitRegL1” and “LogitRegL2”
This when run on the data set with 500 features gives an accuracy of 58 without any regulrizer, L1 gives 58.6666667 and  L2 gives around 58.333333  Vanilla Logistic regression tends to give lower accuracy when compared to other regularize. L1 is outperforms L2 in accuracy. 

	Elastic net regression is a hybrid approach that blends both penalization of the L2 and L1 norms.

β̂  = argmax┬(β )⁡〖|(|(y-Xβ ||^2+ λ_2 |)|β)| |^2+λ_1 ||β||^1 〗

The  L1 part of the penalty generates a sparse model and L2 part of the penalty removes the limitation on the number of selected variables, leads to grouping effect and stabilizes the  l1 regularization path.

Gradient = Gradient + L2(penalty)+L1(penalty)
w(t+1) = w(t) + Alpha(Gradient/Hessian)
  
c.). Elastic Net, L2-regularization and L1-regularization are all techniques of finding solutions in a  linear system. L1 has good ability in finding sparse solution but introduces a large Mean Square Error (MSE) error. L2 is better at minimizing MSE. Elastic net regression is a hybrid approach that blends both penalization of the L2 and L1 norms. L1 is the first moment norm i.e, the absolute dıstance between two points where L2 is second moment norm corresponding to Eucledian Distance 
Another big difference is L1 is not differentiable thus for finding weigths, gradient based approach don’t help

L2-regularized loss function F(x)=f(x)+λ∥x∥2  
L1-regularized loss function F(x)=f(x)+λ∥x∥1 
Elastic net los function F(x)= f(x)+λ∥x∥2 +λ∥x∥1 

