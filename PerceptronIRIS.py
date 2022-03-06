''' Implementation of a Perceptron Network to classify the Iris Dataset
    Developed by Tales Cabral Nogueira      @TalesNogueira
'''

# Important details of some already processed data -------------------------------------------------

'''     n = 10.000 | starting w = 0.5 | generations = 100000 
                    Best generation [5]     - neuron [1] weights: [  13.5   41.5  -51.5   -21.5]  0.00 % error
                    Best generation [1152]  - neuron [2] weights: [ 428.5-1216.5  515.5  -967.5] 28.00 % error
                    Best generation [174]   - neuron [3] weights: [-472.5 -250.5  584.5   488.5]  2.00 % error

        n = 01.000 | starting w = 0.5 | generations = 100000 
                    Best generation [10]    - neuron [1] weights: [   2.5    7.5  -10.     -3.8]   0.00 % error
                    Best generation [1108]  - neuron [2] weights: [  42.3 -120.2   50.3   -93.9]  27.33 % error
                    Best generation [106]   - neuron [3] weights: [ -39.   -15.8   46.5    37.5]   2.00 % error

        n = 00.100 | starting w = 0.5 | generations = 100000 
                    Best generation [6]     - neuron [1] weights: [  -0.07   0.59  -0.49   0.14]   0.00 % error
                    Best generation [1846]  - neuron [2] weights: [   4.92 -14.15   7.36 -13.74]  28.00 % error
                    Best generation [93]    - neuron [3] weights: [  -3.74 -1.21    4.2    3.69]   2.67 % error

        n = 00.010 | starting w = 0.5 | generations = 100000 
                    Best generation [36]    - neuron [1] weights: [  -0.107  0.356 -0.22   0.254]  0.00 % error
                    Best generation [2018]  - neuron [2] weights: [   0.49  -1.41   0.734 -1.371] 28.00 % error
                    Best generation [1019]  - neuron [3] weights: [  -0.879 -0.835  0.996  1.706]  2.67 % error

        n = 00.001 | starting w = 0.5 | generations = 100000 
                    Best generation [335]   - neuron [1] weights: [  -0.0984  0.2555 -0.1708  0.2768]  0.00 % error
                    Best generation [3723]  - neuron [2] weights: [   0.0501 -0.1432  0.0751 -0.1418] 28.00 % error
                    Best generation [1214]  - neuron [3] weights: [  -0.2441 -0.1928  0.2652  0.4354]  2.67 % error '''

# Import Packages ----------------------------------------------------------------------------------
from urllib.request import urlretrieve
import pandas as pd
import numpy as np

# Retrieve Iris Dataset ----------------------------------------------------------------------------
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris, 'IrisDataset.csv')

print('\n> Data:')
irisDataset = pd.read_csv(iris, header=None)
irisDataset.columns = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', "class"]
print(irisDataset.head())

# Define Perceptron --------------------------------------------------------------------------------
# Perceptron constructor
class Perceptron:
    def __init__(self, dataset, specie):
        self.x = dataset.to_numpy()
        self.x = np.delete(self.x, 4, 1)

        self.w = np.full(len(dataset.columns)-1, 0.5, dtype=float)

        self.t = np.zeros(len(dataset), dtype=int)
        for i in range(0, (len(dataset))):
            if(dataset['class'][i] == specie):
                self.t[i] = 1

        self.n = 1

        self.y = np.zeros(len(dataset)+1, dtype=float)

    # Perceptron train procedure
    def train(self, error):
        for i in range(0, self.x.shape[1]):
            self.w[i] += self.n  * (self.t[error] - self.y[error]) * self.x[error][i]

        for j in range(0, self.x.shape[0]):
            self.y[j] = 1
            for i in range(0, self.x.shape[1]):
                self.y[j] += self.x[j][i] * self.w[i]

            if(self.y[j] >= 1):
                self.y[j] = 1
            else:
                self.y[j] = 0

    # Perceptron testing data procedure
    def test(self, x, w, neuron):
        self.y[self.y.shape[0]-1] = 0
        
        for i in range(0, x.shape[0]):
            self.y[self.y.shape[0]-1] += x[i] * w[neuron][i] + 1
        
        if(self.y[self.y.shape[0]-1] >= 1):
            return 1
        else:
            return 0

# Main() -------------------------------------------------------------------------------------------
# Declare the three neurons of the Network
networkSize = 3

neuron = []
MCP = Perceptron(irisDataset, 'Iris-setosa')
neuron.append(MCP) 
MCP = Perceptron(irisDataset, 'Iris-versicolor')
neuron.append(MCP)
MCP = Perceptron(irisDataset, 'Iris-virginica')
neuron.append(MCP)

# Main loop for the generations
generation = 0

# Best values founded during the generations backup
maxHits = 0
bestHits = np.zeros(networkSize, dtype=int)
bestWeights = np.zeros([networkSize+1, len(irisDataset.columns)-1])
bestGeneration = np.zeros(networkSize, dtype=int)

# Start generation-loop
print('\n> Starting routine of generations')

while(generation < 1500):
    hits = [0, 0, 0]
    error = [len(irisDataset)+1, len(irisDataset)+1, len(irisDataset)+1]

    print(f'\n> Generation {generation+1} in process')

    # Verify the weights of the neural network and its errors (saving the first error problem)
    for i in range(0, networkSize):
        for j in range(0, len(irisDataset)):
            if(neuron[i].y[j] == neuron[i].t[j]):
                hits[i] += 1
            else:
                if(error[i] > hits[i]):
                    error[i] = hits[i]

    # Backup of best values
    for i in range(0, networkSize):
        if(bestHits[i] < hits[i]):
            print(f'     > Erro tax for neuron {i+1} is {100-(hits[i]*100/len(irisDataset)):.2f}% in the moment')
            bestHits[i] = hits[i]
            bestWeights[i] = np.copy(neuron[i].w)
            bestGeneration[i] = generation
    
    print()

    # Escape from the loop in case the neural network found the perfect weights to classify the dataset
    if(hits[0] == len(irisDataset) and hits[1] == len(irisDataset) and hits[2] == len(irisDataset)):
        print(f'> Generation {generation+1} was successful in training the neural network')
        break
    

    # Training process of the neurons who made mistakes
    for i in range(0, networkSize):
        if(hits[i] < len(irisDataset)):
            print(f'     > The neuron {i+1} needs more training')
            neuron[i].train(error[i])
    
    generation += 1

# Best data found
print(f'> We have found the best values of the weights')
for i in range(0, networkSize):
    print(f'        > In the generation {bestGeneration[i]} for the Neuron [{i+1}]: {bestWeights[i]} with exactly {bestHits[i]} hits ({100-(bestHits[i]*100/len(irisDataset)):.2f} % error)')

# Test the neural network with outsider data -------------------------------------------------------
while True:
    answer = ''

    print('> Do you wanna test the neural networt? (y/n)', end=' ')
    answer = input()

    if(answer == 'y'):
        print('     > Enter four values (float type) for, respectively, Sepal Length, Sepal Width, Petal Length and Petal Width', end=' ')
        
        x = np.zeros(4, dtype=float)

        try:
            for i in range(0, 4):
                x[i] = float(input())
        except ValueError:
            continue
        except TypeError:
            continue
        except EOFError:
            continue
        
        verify = []
        verify.append(neuron[0].test(x, bestWeights, 0))
        verify.append(neuron[1].test(x, bestWeights, 1))
        verify.append(neuron[2].test(x, bestWeights, 2))

        if(verify[0] == 1 and verify[1] == 0 and verify[2] == 0):
            print('> The type identified was "Iris-setosa"')
        if(verify[0] == 0 and verify[1] == 1 and verify[2] == 0):
            print('> The type identified was "Iris-versicolor"')
        if(verify[0] == 0 and verify[1] == 0 and verify[2] == 1):
            print('> The type identified was "Iris-virginica"')
        else:
            print('> The neural network could not classify this data')
        
    else:
        if(answer == 'n'):
            break