
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class genetic_algorithm():
    """
    Genetic Algorithm Implementation. 
    """

    def __init__(
        self, 
        data: pd.DataFrame,
        risk_free_rate: float, 
        population: int, 
        generations: int, 
        crossover_rate: float, 
        mutation_rate: float, 
        elite_rate: float) -> None:
        """
        Initialise the class with data and parameters.
        """
        self.data = data
        self.n_assets = len(self.data.columns)
        self.rf = risk_free_rate
        self.population = population
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.port_risk = self.find_series_sd()
        self.port_return = self.find_series_mean()
        self.port_cov = self.find_series_cov_matrix() 


    def optimise_weights(self) -> None:
        """
        Run the GA and optimise the weighting.
        """
        self.generate_weights()
        
        for i in range(0, self.generations):
            self.fitness_func()
            self.elitism()
            self.selection()
            self.crossover()
            self.mutation()
            self.avg_gen_result()
            print('Generation', i, ': Average Sharpe Ratio of', self.avg_sharpe, 'from', len(self.weights), 'chromosomes')
        
        self.fitness_func()
        self.optimal_solution()


    def generate_weights(self) -> None:
        """
        Generate random weights for each ticker.
        """        
        self.weights = self.normalise(arr = np.random.random(size = (self.population, self.n_assets)), type = "row")


    def fitness_func(self) -> None:
        """
        Evaluate weights by fitness function.
        """ 
        self.exp_ret = np.sum((self.weights * self.port_return), axis = 1) - self.rf
        self.sd = np.zeros(len(self.weights))
        for i in range(len(self.weights)): 
            self.sd[i] = np.sqrt(np.transpose(self.weights[i]) @ self.port_cov @ self.weights[i])
        self.sharpe = self.exp_ret / self.sd


    def elitism(self) -> None:   
        """
        Perform elitism step by finding n highest sharpe ratios.
        """ 
        n_elite = int(len(self.sharpe) * self.elite_rate)
        elite_index = (-self.sharpe).argsort()[:n_elite]
        self.non_elite_index = np.setdiff1d(range(len(self.sharpe)), elite_index).tolist()


    def selection(self) -> None:   
        """
        Perform selection step by selecting population for crossover and not crossover.
        """ 
        non_elite_sharpes = self.sharpe[self.non_elite_index]
        n_selections = int(len(non_elite_sharpes) / 2)
        self.crossover_index = np.array([])

        self.acc_sharpes = self.normalise(arr = np.cumsum(non_elite_sharpes), type = "cumsum") 

        for _ in range(n_selections):
            rw_prob = random.random()
            index = (np.abs(self.acc_sharpes - rw_prob)).argmin()
            self.crossover_index = np.append(self.crossover_index, index)


    def crossover(self) -> None:   
        """
        Perform crossover step with selected parents.
        """
        for i in range(0, int(len(self.crossover_index)/2), 2): 
            gen1, gen2 = self.crossover_index[i], self.crossover_index[i+1]
            gen1_weights, gen2_weights = self.uni_co(gen1, gen2)
            self.weights[int(gen1)] = self.normalise(gen1_weights, type = "array")
            self.weights[int(gen2)] = self.normalise(gen2_weights, type = "array")

        
    def uni_co(self, gen1: np.array, gen2: np.array) -> np.array:
        """
        Perform uniform crossover step.

        Parameters 
        ----------
        gen1 : first index of population to be crossovered
        gen2 : second index of population to be crossovered

        Returns
        ----------
        w_one : first array of population after crossover
        w_two : second array of population after crossover
        """
        w_one = self.weights[int(gen1)]
        w_two = self.weights[int(gen2)]

        prob = np.random.normal(1, 1, self.n_assets)

        for i in range(0, len(prob)):
            if prob[i] > self.crossover_rate:
                w_one[i], w_two[i] = w_two[i], w_one[i]  
                
        return w_one, w_two


    def mutation(self) -> None: 
        """
        Perform mutation step.
        """
        weight_n = len(self.crossover_index) * self.n_assets
        mutate_gens = int(weight_n * self.mutation_rate)
        
        if self.mutation_rate != 0:
            for _ in range(mutate_gens):
                rand_index = int(np.random.choice(self.crossover_index))
                generation = self.weights[rand_index]
                rand_asset = random.randint(0, (self.n_assets - 1))
                mu_gen = generation[rand_asset]
                mutated_ind = mu_gen * np.random.normal(0,1)
                generation[rand_asset] = abs(mutated_ind)
                generation = self.normalise(arr = generation, type = "array")
                self.weights[rand_index] = generation


    def find_series_mean(self) -> np.array:
        """
        Compute each tickers time series mean return.

        Returns
        ----------
        port_return : array of ticker mean return 
        """
        returns = np.log(self.data / self.data.shift(1))
        port_return = np.array(returns.mean() * 252)
        return port_return 


    def find_series_sd(self) -> np.array:
        """
        Compute each tickers time series standard deviation.

        Returns
        ----------
        port_risk : array of ticker standard deviations
        """
        return np.array(np.std(self.data, axis = 0))

    
    def find_series_cov_matrix(self) -> np.array:
        """
        Compute covariance matrix from each ticket.

        Returns
        ----------
        port_cov : matrix of standard deviations 
        """
        returns = np.log(self.data / self.data.shift(1))
        port_cov = returns.cov()
        return port_cov


    def normalise(self, arr: np.array, type: str) -> np.array:
        """
        Normalise an array.

        Parameters 
        ----------
        arr : array passed to be normalised
        type : on what axis 

        Returns
        ----------
        normal_arr : normalised array
        """
        if type == "cumsum":
            return arr / arr[len(arr) - 1]
        elif type == "row":
            return arr / np.sum(arr, axis = 1)[:, np.newaxis]
        elif type == "array":
            return arr / np.sum(arr)
        

    def optimal_solution(self) -> None:
        """
        Find the optimal solution.
        """
        optimal_index = np.argmax(self.sharpe)
        self.optimal_weight = self.weights[optimal_index]
        self.optimal_sharpe = self.sharpe[optimal_index]


    def avg_gen_result(self) -> float:
        """
        Compute average result from current population weights.

        """
        self.avg_sharpe = round(np.mean(self.sharpe), 2)


    def plot_efficient_frontier(self) -> None:
        """
        Plot the efficient frontier. 
        """
        cm = plt.cm.get_cmap('RdYlBu')

        fig = plt.scatter(self.exp_ret, self.sd, c = self.sharpe, cmap = cm)
        plt.colorbar(fig)
        plt.xlabel("Standard Deviation")
        plt.ylabel("Return")
        plt.title("Efficient Frontier")
        plt.show()
