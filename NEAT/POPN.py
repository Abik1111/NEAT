import DNA
from DNA import Genome
import numpy as np
class Population():
    def __init__(self,inputs,outputs,N,pop_list=None):
        #we have to enter the input, the output and the 
        #size of the population
        self.N=N
        self.inputs=inputs
        self.outputs=outputs
        if pop_list==None:
            self.population=[]  #the list to store all the genomes
            self.create_list(inputs,outputs,N)
        else:
            self.population=pop_list.copy()
            # for gnome in self.population:
            #     gnome.update_node_gene(self.inputs,self.outputs)
        self.rep_list=[]        #to define the key for each of the representatives of species
        self.species={}             #the actual dictionary holding the distinguishable species
        self.adjusted_fitness={}    #this is a dictionary storing the value of 
                                    #adjusted fitness with key as the representative gene
                                    #order as that of the species dictionary
        self.next_generation={}     #this gives the count of the genomes in the next
                                    #generation. the key is the representative genome
                                    #along with the count as the value




    def create_list(self,inputs,outputs,N):
        '''
        this function takes the input as the inputs, outputs of each genome and the number of 
        genomes that should be present in the population
        '''
        for _ in range(N):
            self.population.append(Genome(inputs,outputs))
        self.population[-1].tally()                         #this line helps the formation of the CONNECTION_GENE
        DNA.INNOVATION=max(DNA.CONNECTION_GENE.values())    #initialization of the INNOVATION variable

    def crossover(self, g1,g2):
        '''
        this function takes the input as the parent genomes and produces 
        a child more alike to the fit genome
        '''
        f1=g1.fitness
        f2=g2.fitness
        if f1>f2:
            # print("We took f1")#we copy the gene with higher fitness
            child=g1.copy()
            for a,b in child.connection_gene.items():
                flag=False                 #used to enable the already disabled gene
                if a in g2.connection_gene:#this if condition tallies all the homologous gene
                    if np.random.uniform(0,1)>0.5:
                        child.connection_gene[a]=g2.connection_gene[a].copy()
                    if not g2.connection_gene[a][1]:# if false, the condition to enable the disabled gene
                        if np.random.uniform(0,1)>0.75:
                            flag=True
                            child.connection_gene[a][1]=True
                if not flag and not b[1]:# this condtion works for the disjoint and the excess genes
                                         # used to enable the disabled gene
                    if np.random.uniform(0,1)>0.75:
                        child.connection_gene[a][1]=True                    

                    
        else:
            # print("We took f2")
            child=g2.copy()

            for a,b in child.connection_gene.items():
                flag=False                 #used to enable the already disabled gene
                if a in g1.connection_gene:#this if condition tallies all the homologous gene
                    if np.random.uniform(0,1)>0.5:
                        child.connection_gene[a]=g1.connection_gene[a].copy()
                    if not g1.connection_gene[a][1]:# if false, the condition to enable the disabled gene
                        if np.random.uniform(0,1)>0.75:
                            flag=True
                            child.connection_gene[a][1]=True
                if not flag and not b[1]:# this condtion works for the disjoint and the excess genes
                                         # used to enable the disabled gene
                    if np.random.uniform(0,1)>0.75:
                        child.connection_gene[a][1]=True
        return child

    def speciate(self):
        '''
        this function helps to group the population in to a number of species according to the compatibility distance
        this function takes no input and helps append into the variables : rep_list, species and adjusted_fitness
        '''
        threshold_compatibility=3.0
        self.rep_list.append(0)
        self.species[0]=[0]
        self.adjusted_fitness[0]=[self.population[0].fitness]
        for i in range(1,self.N): # we iterate through the component genomes of the population
            placed=False # this flag variable determines if we got any species that could encompass this particular genome
            for j in self.rep_list: # we iterate throught the representatives of the species
                comp_d=self.compatibility_distance(self.population[i],self.population[j])
                if comp_d<=threshold_compatibility:#the condition for a genome to belong in a species
                                                    # is that the representative of the species and that genome
                                                    # must lie within the threshold comp_dist
                    # print("Placing in the old representative",j)
                    self.species[j].append(i)
                    self.adjusted_fitness[j].append(self.population[i].fitness)
                    placed=True
                    break
            if not placed:
                # print("Creating new representative",i)
                # we create new representative and make that genome the representative of that particular species
                self.rep_list.append(i)
                self.species[i]=[i]
                self.adjusted_fitness[i]=[self.population[i].fitness]
        for a,b in self.adjusted_fitness.items():
            # print(self.adjusted_fitness)
            # here we divide by the number of genomes in the species to calculate
            # the actual adjusted fitness
            self.adjusted_fitness[a]=[x/len(b) for x in b]

    def members_in_nxt_gen(self):
        #to calculate the average adjusted fitness of the entire population 
        f_bar=sum([sum(x) for x in self.adjusted_fitness.values()])/self.N
        tots=0# this variable is used to check if the total counts of the genomes reach to N
        for a,b in self.adjusted_fitness.items():
            self.next_generation[a]=np.around(sum(b)/f_bar)
            tots+=np.around(sum(b)/f_bar)
        if tots<self.N:
            keyy=np.random.choice(list(self.adjusted_fitness.keys()),1)
            # print(keyy.item())
            self.next_generation[keyy.item()]+=self.N-tots
    




    def compatibility_distance(self,g1,g2):
        '''
        this function is used to calculate the distance between any two genomes
        '''
        g1_inn=[a[2] for a in g1.connection_gene.values()]#g1_inn stores all the innovation numbers of g1
        g2_inn=[a[2] for a in g2.connection_gene.values()]#g2stores all the innovation number of g2
        #the code below checks for the excess genes between two genomes
        e=0
        if max(g1_inn)>max(g2_inn):
            for i in g1_inn:
                if i>max(g2_inn):
                    e+=1
        elif max(g2_inn)>max(g1_inn):
            for i in g2_inn:
                if i >max(g1_inn):
                    e+=1
        else:
            e=0
        #the code below check for disjoint genes in two genomes
        d=0
        for i in g1_inn:
            if not i in g2_inn and i<max(g2_inn):
                d+=1

        for i in g2_inn:
            if i not in g1_inn and i<max(g1_inn):
                d+=1

        # the code below calculates the average difference of weights
        diff=[]
        for a,b in g1.connection_gene.items():
            if a in g2.connection_gene:
                diff.append(abs(b[0]-g2.connection_gene[a][0]))
        w=np.mean(diff)
        c1=1.0
        c2=1.0
        c3=0.4
        if len(g1_inn)>=len(g2_inn):
            N=len(g1_inn)
        else:
            N=len(g2_inn)
        if N<20:
            N=1
        #FINALLY the formula to calculate the compatibility distance
        comp_dist=c1*e/N+c2*d/N+c3*w
        return comp_dist
    def update_ipop(self,individual,inputs,outputs):
        individual.inputs=inputs
        individual.outputs=outputs
        for a,b in zip(individual.input_keys,individual.inputs):
            individual.node_gene[a]=b
        for a,b in zip(individual.output_keys,individual.outputs):
            individual.node_gene[a]=b

    def generate_new_population(self,rate=0.01,inputs=None,outputs=None):
        # count=0
        new_pop_list=[]
        for rep,counts in self.next_generation.items():
            loaded_genome_ids=self.species[rep]
            fitness_probs=[x/max(self.adjusted_fitness[rep]) for x in self.adjusted_fitness[rep]]
            # print(tuple(np.random.choice(loaded_genome_ids,2,fitness_probs)))
            first=True
            # print(rep)
            for _ in range(int(counts)):
                max_fit=np.argmax(fitness_probs)
                if first and len(self.population[loaded_genome_ids[max_fit]].node_gene)<5:
                    first=False
                    child=self.population[loaded_genome_ids[max_fit]].copy()
                    if inputs!=None and outputs!=None:
                        self.update_ipop(child,inputs,outputs)
                    # child.print_obj()
                    child.propagate()
                    new_pop_list.append(child)
                    continue
                # print(tuple(np.random.choice(loaded_genome_ids,2,fitness_probs)))
                parent1_idx,parent2_idx=tuple(np.random.choice(loaded_genome_ids,2,fitness_probs))
                parent1=self.population[parent1_idx]
                parent2=self.population[parent2_idx]
            
                child=self.crossover(parent1,parent2)
                if np.random.uniform(0,1)<rate:
                    child.mutate()

                if inputs!=None and outputs!=None:
                    self.update_ipop(child,inputs,outputs)
                # child.print_obj()
                child.propagate()
                new_pop_list.append(child)
        new_population=Population(inputs,outputs,self.N,new_pop_list)
        return new_population
    
    def max_fitness(self):
        ft=max([a.fitness for a in self.population])
        return ft


# p=Population([0,1],[1],10)
# print(p.population)
# p.speciate()
# p.members_in_nxt_gen()
# print(p.next_generation)
# p=p.generate_new_population()
# print(p.population)




