# Linear Regression
## Resources Used
* https://www.youtube.com/watch?v=nk2CQITm_eo&list=PLblh5JKOoLUIzaEkCLIUxQFjPIlapw8nU&index=2 (R<sup>2</sup> and F-Statistics)
* https://www.youtube.com/watch?v=pkJjoro-b5c&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=19 (Gradient Descend) 
* https://www.youtube.com/watch?v=N20rl2llHno (Degrees of Freedom)
## Algorithm
### 1 - Gradient Descent with Least Squares
### 2 - Calculate R<sup>2</sup> Score
x = feature, y = target variable, SS = sum of squares, mean = mean(y), n = number of data points, fit = fitted line to the data </br> </br>
SS(mean) => Sum of squres around the mean : (mean - y)<sup>2</sup> </br>
Var(mean) => Variation around the mean (average sum of squares per data point) : SS(mean) / n </br>
Var(fit) => Variation around the least squares line : SS(fit) / n </br> </br>
If Var(fit) < Var(mean) we can conclude that some of variation is explained the taking feature x into account. </br> </br>

R<sup>2</sup> = (Var(mean) - Var(fit)) / Var(mean) </br>
R<sup>2</sup> score shows us how much of the variation can be explained by taking feature x into account. </br> 
For example if R<sup>2</sup> is 0.80, we can say that our fitted line can reduce 80% of the variation.
#### Drawbacks for R<sup>2<sup>
  * Keep adding extra features into feature vector, will never perform worse in terms of R<sup>2</sup>
    * For example feature_vector_1 = [x1,x2,x3,x4] will never perform worse than the feature_vector_2 = [x1,x2]
    * This happens because if a silly feature has no effect on the predcition it will get the coefficent as 0, which means it has no effect. However, due to random small chances it may effect the prediction and this results as bigger R<sup>2</sup>.
    * As a result, the more parameters we add to the equation, the more opportunities we have for random events to reduce SS(fit) and result in better R<sup>2</sup>.
  * In order to overcome this issue, "adjusted r<sup>2</sup>" can be used. It scales the R<sup>2</sup> by the number of parameters.
### 3 - Calculate p-value for R<sup>2<sup>
If there are only 2 data points, there is always 1 straight line that connects 2 data points, and this results as 100% R<sup>2</sup> score which is perfect. However, this does not mean anything. We need a way to determine if the R<sup>2</sup> score statistically important. This is p-value by using F-Statistics. 
p<sub>fit</sub> = number of parameters for fitted line, p<sub>mean</sub> = number of parameters for the mean line
* First, calculate the F with the following formula: (SS(mean) - SS(fit) / (p<sub>fit</sub> - p<sub>mean</sub>)) / (SS(fit) / (n - p<sub>fit</sub>))
* Second, using F distribution and F score that is just calculated, find the p-value.
* If p-value <= 0.05 this indicates that R<sup>2</sup> is statistically significant, else it is not.
