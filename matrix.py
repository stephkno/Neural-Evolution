import numpy
import random
import pickle
import copy
import json

class Utils():
    def __init__(self):
        super(Utils, self).__init__()
        self.prob_mutate = 0.8
        self.prob_add_gene = 0.01

    def mutation_curve(self,x):
        return x

    def combine(self, agent1, agent2, inputs, outputs, binary, hidden, n):

        if agent2.genome.shape[0] > agent1.genome.shape[0]:
            g_diff_size = agent2.genome.shape[0]-agent1.genome.shape[0]
            agent1.add_nodes(g_diff_size)

        elif agent1.genome.shape[0] > agent2.genome.shape[0]:
            g_diff_size = agent1.genome.shape[0]-agent2.genome.shape[0]
            agent2.add_nodes(g_diff_size)

        genome1 = agent1.genome
        genome2 = agent2.genome

        mask1 = ((numpy.random.random((agent1.size, agent1.size)) > n)*2.0)-1
        mask2 = mask1 * -1.0
        mask1 = (mask1 > 0.0)*1.0
        mask2 = (mask2 > 0.0)*1.0

        genome1 = genome1 * mask1
        genome2 = genome2 * mask2
        new_genome = genome1+genome2

        new_agent = copy.copy(agent1)
        new_agent.fitness = 0
        new_agent.lines_cleared = 0

        new_agent.genome = new_genome
        return new_agent

    def generate_agents(self, agents, best, num_species, inputs, outputs, bin, hidden, n_agents):

        new_agents = agents
        i = 0
        while len(new_agents) < n_agents:

            new_agent = self.combine(best, agents[-2], inputs, outputs, bin, hidden, i/n_agents-1)
            #new_agent = copy.copy(agents[-1])
            new_agent.fitness = 0

            
            if numpy.random.random() > self.prob_mutate:

                if numpy.random.random() > self.prob_add_gene:
                    new_agent.add_gene(self.mutation_curve(i))
                else:
                    new_agent.remove_gene(1)
            else:
                new_agent.mutate_gene(self.mutation_curve(i))
            i+=1
            num_species+=1

            new_agent.id = num_species
            new_agents.append(new_agent)

        return new_agents, num_species

    def export_genome(self, net):
        filehandler = open('Agent{}.obj'.format(net.id), 'wb')
        pickle.dump(net, filehandler)
        print("Saved agent as Agent{}.obj".format(net.id))

    def import_genome(self, filename):
        print("Loading {}...".format(filename))
        filehandler = open(filename, 'rb')
        net = pickle.load(filehandler)
        return net

class Network():
    def __init__(self, inputs, outputs, binary, id):
        super(Network, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.lines_cleared = 0
        self.render = False
        mat_size = inputs+outputs
        self.size = mat_size
        self.genome = numpy.zeros((mat_size, mat_size))
        self.nodes = numpy.zeros(mat_size)
        self.recurrence = numpy.zeros(mat_size)
        self.activation = numpy.tanh
        self.fitness = 0.0
        self.connections = 0
        self.binary = binary
        self.neuron_decay = 0.5
        self.recurrent_decay = 0.0
        self.id = id

    def set_fitness(self, score):
        if self.fitness == 0:
            self.fitness = score
        else:
            if score > self.fitness:
                self.fitness = score

    def get_genome(self):
        genome = self.genome.flatten()
        genome = {"size": [self.size], "genome": genome, "fitness": 0, "generation": 0}
        return genome

    def softmax(self, x):
        return numpy.exp(x) / sum(numpy.exp(x))
    
    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def evaluate(self, x):

        #clear node vectors
        self.nodes = self.nodes * self.neuron_decay
        self.nodes[-self.outputs:] *= 0.0

        recurr_mask = numpy.zeros(self.size)
        for i in range(self.inputs, (self.size-self.outputs)):
            recurr_mask[i] = 1.0

        #activate and add previous iteration's recurrence vector
        self.nodes += self.activation(self.recurrence)
        self.recurrence = (self.recurrence * recurr_mask) * self.recurrent_decay

        x = numpy.array(x)
        x = x.flatten()

        #fill input nodes
        for i, input in enumerate(x):
            self.nodes[i] = input

        #evaluate each node
        for i, node in enumerate(self.nodes):
            vec = self.activation(node)
            row = (self.genome[i] * vec) # multiply node input by weights

            #split row into two parts
            recurrent_row = row[:i]
            node_row = row[i:]

            #cat zeros to end
            a = numpy.zeros_like(recurrent_row)
            b = numpy.zeros_like(node_row)

            recurrent_row = numpy.concatenate((recurrent_row, b))
            node_row = numpy.concatenate((a, node_row))

            #add values to output nodes
            self.recurrence += recurrent_row

            self.nodes += node_row
           # #if i > self.size - self.outputs:
            #    print(self.genome[i][-2:])

        out = self.nodes[-self.outputs:]
        return self.sigmoid(out)

    def add_nodes(self, x):
        for _ in range(x):
            #insert new node into vectors
            self.nodes = numpy.insert(self.nodes, self.inputs, 0.0, axis=0)
            self.recurrence = numpy.insert(self.recurrence, self.inputs, 0.0, axis=0)
            #insert row and column into genome
            self.genome = numpy.insert(self.genome, self.inputs, 0.0, axis=0)
            self.genome = numpy.insert(self.genome, self.inputs, 0.0, axis=1)
            self.size += 1

    def add_gene(self, inp):
        for _ in range(inp):

            inputs = []
            outputs = []
            hiddens = []
            
            for i in range(self.inputs):
                inputs.append(i)
            for i in range(self.inputs, self.size-self.outputs):
                hiddens.append(i)
            for i in range((self.size-self.outputs), self.size):
                outputs.append(i)

            #choice 1: connect input to hidden to output
            #choice 2 connect hidden to hidden

            choice = random.choice(range(3))

            if choice == 0:
                x = random.choice(inputs)  # choose in or out node
                y = random.choice(hiddens)  # choose hidden node
                z = random.choice(outputs)  # choose output node
                self.genome[x][y] = random.random() * 2 - 1
                self.genome[y][z] = random.random() * 2 - 1
                if self.genome[y][z] > 1.0: self.genome[x][y] = 1.0
                if self.genome[y][z] < -1.0: self.genome[x][y] = -1.0
            if choice == 1:
                x = random.choice(hiddens)  # choose hidden
                y = random.choice(hiddens)  # choose hidden
                self.genome[x][y] = random.random() * 2 - 1
            if choice == 2:
                x = random.choice(inputs)  # choose input
                y = random.choice(outputs)  # choose output
                self.genome[x][y] = random.random() * 2 - 1

            #if self.binary:
            #    if r > 0.0:
            #        r = 1.0
            #    elif r <= 0.0:
            #        r = -1.0

            #if self.genome[x][y] > 1.0: self.genome[x][y] = 1.0
            #if self.genome[x][y] < -1.0: self.genome[x][y] = -1.0

            self.connections += 1

    def remove_gene(self, x):
        for _ in range(x):
            x = random.choice(range(len(self.genome)))
            choice = []
            for i, c in enumerate(self.genome[x]):
                if c != 0.0:
                    choice.append(i)

            if len(choice) == 0: break
            y = random.choice(choice)

            if x <= self.size - self.outputs:
                self.genome[x][y] = 0.0
                self.connections -= 1

    def mutate_gene(self, x):
        for _ in range(x):
            
            #choose which genes to mutate
            x = random.choice(range(len(self.genome)))
            choice = []
            for i, c in enumerate(self.genome[x]):
                if c != 0.0:
                    choice.append(i)

            if len(choice) == 0: break
            
            y = random.choice(choice)
            r = (random.random() * 2 - 1)

            if self.binary:
                if r > 0.0: r = 1.0
                elif r <= 0.0: r = -1.0

            d = r
            self.genome[x][y] += d

            if self.genome[x][y] > 1.0: self.genome[x][y] = 1.0
            if self.genome[x][y] < -1.0: self.genome[x][y] = -1.0

