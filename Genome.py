import numpy as np

CONNECTION_GENE = {}
INNOVATION = 0


class Genome:
    def __init__(self, inputs, outputs):
        self.inputs = inputs  # to save the inputs
        self.outputs = outputs  # to save the outputs
        self.hidden_keys = []  # the keys of the hidden neurons
        self.input_keys = []  # the keys of the input neurons
        self.output_keys = []  # the keys of the output neurons
        self.innov_num = 0  # keeps account of the innovation number only of a particular class
        # the total number of nodes till now
        self.num_nodes = len(self.inputs)+len(self.outputs)
        self.node_gene = {}  # the node gene
        self.connection_gene = {}  # the connection gene
        self.fitness = 0  # fitness score of the genome
        self.distance = {}  #this is used to store the distance of the neuron from the input neuron
        self.createGenome()

    def createGenome(self):
        '''
        this is the function to create the Genome structure such that the bias genome is always assigned 
        a key of 0 and the n inputs are provided the first n indexes(1-n) and the k indexes to output(n+1,n+k)

        the weights are generated randomly with gaussian distribution

        the bias neuron is given a value of 1 always 
        '''
        self.node_gene = {i: input for i, input in enumerate(
            np.concatenate(([1], self.inputs, self.outputs)))}
        keys = list(self.node_gene.keys())
        self.input_keys = keys[1:len(self.inputs)+1]
        self.output_keys = keys[len(self.inputs)+1:]

        weights = np.random.uniform(-1, 1,
                                    (len(self.outputs), len(self.inputs)))
        for i in self.output_keys:
            for j in self.input_keys:
                self.innov_num = self.innov_num+1
                enable = True
                self.connection_gene[(j, i)] = [
                    weights[i-len(self.inputs)-1, j-1], enable, self.innov_num]
        for key in keys:
            if key in self.output_keys:
                self.distance[key] = 1      #max distance of 1
            else:
                self.distance[key] = 0      #min distance of 0

    def compute_fitness(self, result):
        '''
        this funciton is used to compute fitness by comparing only the output nodes with the help of output keys 
        stored as a member variable in the class

        input: result i.e the value that reaches to the output node once propagation is done

        the result is stored in the fitness member variable

        '''
        total = sum([abs(self.node_gene[x]-result[x])
                     for x in self.output_keys])
        print(total)
        if total < 1e-3:
            total = 1e-3
        self.fitness = 1/total

    def propagate(self):
        '''
        this funciton takes no input
        
        this function first sorts the neuron according to the distance from the input neron
        so that dependency is taken care of and neuron activations are evaluated sequentially
        then it seperated the input keys along with the bias with key '0' since they 
        don't need to be evaluated
        
        fitness function is directly called from here 
        '''
        dir_keys = sorted([(x, y) for y, x in self.distance.items()])
        sortd_dict = dict([(x, y) for (y, x) in dir_keys])
        print(sortd_dict)
        temp = self.node_gene.copy()
        for i in list(sortd_dict.keys())[len(self.input_keys)+1:len(self.distance)]:
            print(i)
            temp[i] = 0
            for j in temp.keys():  # this considers all the nodes to check the connections input to the node at i
                print("Accesseing connection {} from {}".format(i, j))
                if (j, i) not in self.connection_gene.keys():
                    continue
                if not self.connection_gene[(j, i)][1]:
                    continue
                temp[i] += temp[j]*self.connection_gene.get((j, i), 0)[0]
                print(f"{temp[j]}x{self.connection_gene.get((j,i),0)[0]}={temp[i]}")
            temp[i] = self.activation(temp[i])
        self.compute_fitness(temp)

    def activation(self, x):
        '''
        This is the steep sigmoid function
        '''
        # return 1/(1+np.exp(-4.9*x))
        # momentarily the relu function
        return np.maximum(-0.01, x)

    def weight_mutate(self, rate=0.8):
        '''
        this function is used to mutate the weights of the connections randomly 
        at a rate of 0.8% by default
        '''
        if np.random.uniform(0, 1) < rate:
            print('success\n')
            for a, _ in self.connection_gene.items():
                weight = self.connection_gene[a][0]
                if np.random.uniform(0, 1) <= 0.9:
                    # random perturbation of weight
                    print('success1\n')
                    weight += np.random.uniform(-1, 1)
                else:
                    # complete change of weight
                    weight = np.random.uniform(-1, 1)
                    print('success2\n')
                self.connection_gene[a][0] = weight

    def node_mutate(self, rate=0.03):
        '''
        this function randomly inserts a new node between connections at the rate
        of 0.03% by default
        '''
        global INNOVATION
        # global CONNECTION_GENE
        # print(INNOVATION)
        temp = self.connection_gene.copy()
        for a, b in temp.items():
            if np.random.uniform(0, 1) <= rate and b[1]:
                # the rate must satisfy as well as the connection must be enabled
                # create variables to store the node numbers
                start, end = a
                # disable the conneciton in the connection gene
                self.connection_gene[a] = [b[0], False, b[2]]
                # find the key with the maximum value in the node gene and then increment it by one to get a new node
                # it has been initialized with 1
                center = max(self.node_gene.keys())+1
                self.node_gene[center] = 0
                #now here we also have to set the distance paramter as the midpoint
                self.distance[center]=(self.distance[start]+self.distance[end])/2
                # now in order to update the connection gene we have to check if the connection is novel or already
                # exits, this is necessary because we cannot make the innovation number redundant
                # the mating might be affected
                if (start, center) in CONNECTION_GENE.keys():
                    self.connection_gene[(start, center)] = [
                        1, True, CONNECTION_GENE[(start, center)]]
                else:
                    INNOVATION += 1
                    self.connection_gene[(start, center)] = [
                        1, True, INNOVATION]
                    # updating the CONNECTION GENE universal dicitonary as well
                    CONNECTION_GENE[(start, center)] = INNOVATION
                # also now i check for connection from center to end
                if (center, end) in CONNECTION_GENE.keys():
                    self.connection_gene[(center, end)] = [
                        b[0], True, CONNECTION_GENE[(center, end)]]
                else:
                    INNOVATION += 1
                    self.connection_gene[(center, end)] = [
                        b[0], True, INNOVATION]
                    CONNECTION_GENE[(center, end)] = INNOVATION

    def connection_mutate(self, rate=0.05):
        '''
        This function helps to form connection between the nodes at random 
        This does so by checking if the connection is already existing or if connection is viable
        '''
        global INNOVATION
        for i in self.node_gene.keys():
            if i in self.output_keys:
                continue
            for j in self.node_gene.keys():  # the loops are used to produce every combination of the nodes
                if i == j:  # eliminate the condition in which the connection between the same node is formed
                    continue
                # eliminate node formation to inputs but not from inputs
                if j in self.input_keys+[0]:
                    continue
                if (i, j) in self.connection_gene or (j, i) in self.connection_gene:
                    # this helps to eliminate the condition in which the connection is already exists
                    # the reverse and forward combination is treated as same
                    continue
                if self.distance[i]>=self.distance[j]:#this is to check condition that the destination node lies further away in network
                    continue
                if np.random.uniform(0, 1) < rate:
                    if (i, j) in CONNECTION_GENE:
                        self.connection_gene[(i, j)] = [np.random.uniform(
                            0, 1), True, CONNECTION_GENE[(i, j)]]
                    else:
                        INNOVATION += 1
                        self.connection_gene[(i, j)] = [
                            np.random.uniform(0, 1), True, INNOVATION]
                        CONNECTION_GENE[(i, j)] = INNOVATION

    def mutate(self):
        '''
        this funciton is used to call the differenty type of mutation function
        '''
        self.weight_mutate()
        self.node_mutate()
        self.connection_mutate()

    def tally(self):
        '''
        this function is used to initialize the CONNECTION_GENE which helps to
        keep account of all the possible connections later used for crossover
        '''
        for a, b in self.connection_gene.items():
            if not a in CONNECTION_GENE:
                CONNECTION_GENE[a] = b[-1]

    def print_obj(self):
        '''
        This is only a simple function to print the genes
        '''
        print(self.node_gene)
        print(self.connection_gene)
        print(CONNECTION_GENE)


