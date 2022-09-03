# Linear Regression
## Resources Used
* https://www.youtube.com/watch?v=nk2CQITm_eo&list=PLblh5JKOoLUIzaEkCLIUxQFjPIlapw8nU&index=2
* https://www.youtube.com/watch?v=pkJjoro-b5c&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=19
## Algorithm
### 1 - Gradient Descent with Least Squares
### 2 - Calculate R^2 Score
x = feature, y = target variable, SS = sum of squares, mean = mean(y), n = number of data points, fit = fitted line to the data </br> </br>
SS(mean) => Sum of squres around the mean : (mean - y)<sup>2</sup> </br>
Var(mean) => Variation around the mean (average sum of squares per data point) : SS(mean) / n </br>
Var(fit) => Variation around the least squares line : SS(fit) / n </br>
If Var(fit) < Var(mean) we can conclude that some of variation is explained the taking feature x into account. </br> </br>

R<sup>2</sup> = (Var(mean) - Var(fit)) / Var(mean) </br>
R<sup>2</sup> score shows us how much of the variation can be explained by taking feature x into account. </br> 
### 3 - Calculate p-value for R^2
