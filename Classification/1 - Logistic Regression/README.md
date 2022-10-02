# Logistic Regression
## Resources Used
## Algorithm
### Hyphotesis Function
m = number of samples, n = number of features, x = training samples in a shape of (n,m), θ = weights in a shape of (n,1) </br>
sigmoid(x) = 1 / (1 + e<sup>-x</sup>) </br>
h<sub>θ</sub>(x) = sigmoid(θ<sup>T</sup>x) => 1 / (1 + e<sup>-θ<sup>T</sup>x</sup>) </br>
### Cost Function (Maximum Likelyhood)
![alt text for screen readers](images/cost-function.png "Loss Function")
We can analyze this cost function on 2 different states.
#### If the Actual Class Equals to 1
If the actual class equals to 1 then our cost function equals to -log(h<sub>θ</sub>(x)). If we look at the graph of -log(h<sub>θ</sub>(x)), we can infer that while the h(x) goes to 0, cost function converges to &infin;. This means we penalize the algorithm if it predicts as 0 when the actual class is 1, by adding huge numbers to the cost function.
![alt text for screen readers](images/negative_logx.png "-log(h(x))")
#### If the Actual Class Equals to 0
If the actual class equals to 0 then our cost function equals to -log(1 - h<sub>θ</sub>(x)). If we look at the graph of -log(1 - h<sub>θ</sub>(x)), we can infer that while the h(x) goes to 1, cost function converges to &infin;. This means we penalize the algorithm if it predicts as 1 when the actual class is 0, by adding huge numbers to the cost function.
![alt text for screen readers](images/negative_log_1-x.png "-log(1 - h(x))")
#### Combine it into 1 function
![alt text for screen readers](images/simplified-cost.png "Combine Cost Functions")
Nothing changes, we just combine the functions into 1. If y=1 cost function equals to -log(h<sub>θ</sub>(x)), else cost function equals to  -log(1 - h<sub>θ</sub>(x)).
### Gradient Descend
![alt text for screen readers](images/gradient-descend.png "Gradient Descend")
Update rule for gradient descend is identical with the linear regression. One thing differ and it is the definiont of the hypothesis. For the linear regression, it is θ<sup>T</sup>x, for the logistic regression it is 1 / (1 + e<sup>-θ<sup>T</sup>x</sup>).
### Regularization
