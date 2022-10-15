# Support Vector Machine
## Resources
## Algorithm
Basically, support vector machine aims to seperate the data with the hyperplanes by maximazing the mariginal distance. 
This mariginal distance is defined as the nearest observations to the hyperplane. Also, this observations are called as support vectors. Formula of the
hyperplane is as following: </br>
![alt text for screen readers](images/hyperplane-formula.png "Hyperplane Formula")
### Mathematical Derivation of Optimization Function
In order to classify an obervation, we require the following inequalities with an additional error term that gives more flexibility for our model. That transforms
our algorithm from large mar,gin classifier to soft marigin classfier. In the later stpes this will be more clear. </br>
![alt text for screen readers](images/pn-samples.png "Classification Requirement")
If we introduce a new variable called t<sub>i</sub> as 1 if an observation is postive; -1 otherwise, we can combine positive and negative classification 
formula into 1 single formula.
![alt text for screen readers](images/combine-pn.png "Combined Requirement")
* If error term equals to 0, then sample is correctly classified and outside the marigin.
* If error is between the 0 and 1, then sample is correctly classfied and it is inside the marigin.
* If error is greater than 1, then it is missclassified.
Therefore, we can define an error term as follows. </br>
![alt text for screen readers](images/soft-marigin-error.png "Soft Marigin Error")
We want to maximize the marigin by minimizing the norm of the weight. Also, we wanto to minimize the soft marigin error. Therefore, we can define optimization
objective as follows, with the some constant regularization term C in front of the soft error. </br>
![alt text for screen readers](images/optimization-func.png "Optimization Objective")
Functions on some constraints can be optimized via the method of Lagrange Multipliers.
#### Method of Lagrange Multipliers
If the optimization function is rewritten as Lagrange Method required, we get the new following formula. </br>
![alt text for screen readers](images/lagrange.png "Lagrange Method")
After that, if we take the derivatives in terms of all the 4 parameters and set them to 0, we get the following: </br>
![alt text for screen readers](images/lagrange-subs.png "Lagrange Derivatives")
If we substitute the above on the lagrange function, finally we get the following optimization function. It is important to note that final optimization function depends
on the dot products of pairs of samples. This makes support vector machines so powerful, because we can use transformations on the observations so effectively.
![alt text for screen readers](images/final-optimization.png "Final Optimization Objective")
### Linearly Seperable Data
### Effect of Parameter C
### Not Linearly Seperable Data & Usage of Kernels
### Effect of Gaussian Kernel Parameters