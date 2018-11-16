
# Test for data scientist

1. Faça uma análise exploratória para avaliar a consistência dos dados e identificar
possíveis variáveis que impactam na qualidade do vinho.

|       | fixed acidity | volatile acidity | citric acid | residual sugar | chlorides   | free sulfur dioxide | total sulfur dioxide | density     | pH          | sulphates   | alcohol     | quality     |
|-------|---------------|------------------|-------------|----------------|-------------|---------------------|----------------------|-------------|-------------|-------------|-------------|-------------|
| count | 6497.000000   | 6497.000000      | 6497.000000 | 6497.000000    | 6497.000000 | 6497.000000         | 6497.000000          | 6497.000000 | 6497.000000 | 6497.000000 | 6497.000000 | 6497.000000 |
| mean  | 7.215307      | 0.339666         | 0.318633    | 5.443235       | 0.056034    | 30.525319           | 115.744574           | 1.710882    | 3.218501    | 0.531268    | 12.157179   | 5.818378    |
| std   | 1.296434      | 0.164636         | 0.145318    | 4.757804       | 0.035034    | 17.749400           | 56.521855            | 7.636088    | 0.160787    | 0.148806    | 33.946284   | 0.873255    |
| min   | 3.800000      | 0.080000         | 0.000000    | 0.600000       | 0.009000    | 1.000000            | 6.000000             | 0.987110    | 2.720000    | 0.220000    | 8.000000    | 3.000000    |
| 25%   | 6.400000      | 0.230000         | 0.250000    | 1.800000       | 0.038000    | 17.000000           | 77.000000            | 0.992340    | 3.110000    | 0.430000    | 9.500000    | 5.000000    |
| 50%   | 7.000000      | 0.290000         | 0.310000    | 3.000000       | 0.047000    | 29.000000           | 118.000000           | 0.994890    | 3.210000    | 0.510000    | 10.300000   | 6.000000    |
| 75%   | 7.700000      | 0.400000         | 0.390000    | 8.100000       | 0.065000    | 41.000000           | 156.000000           | 0.996990    | 3.320000    | 0.600000    | 11.300000   | 6.000000    |
| max   | 15.900000     | 1.580000         | 1.660000    | 65.800000      | 0.611000    | 289.000000          | 440.000000           | 103.898000  | 4.010000    | 2.000000    | 973.333000  | 9.000000    |

- Como foi a definição da sua estratégia de modelagem?
    
    First, I analyzed for each column all the statistics. After this, I realized that the data had very different distributions which makes the classification might be wrong. So, I had to perform tests to discretize the base at different intervals. The tests and their results are available in the file folder.

- Como foi definida a função de custo utilizada?  

    The cost function used was entropy, which, all decision tree of random forest use to build their model.

- Qual foi o critério utilizado na seleção do modelo final? 

    Random Forest is an algorithm ideal to construct models with a high variance, which is the case for the data used here in this test.

- Qual foi o critério utilizado para validação do modelo?  Por que escolheu utilizar este método?

    Several methods were tested, the model chosen was the one that obtained the best result in terms of acuracy.
    The methods tested were SVM(support vector machine); SVR(support vector regression); decision tree and random forest.

- Quais evidências você possui de que seu modelo é suficientemente bom?  

    To verify the quality of the model was calculated its acuracy, thus, this metric along with others was the evidence that we obtained.

