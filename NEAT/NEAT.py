import numpy as np
import copy
np.random.seed(42)
class CONNECTIONS:
    '''
    This class holds the information about the innovation number of each connection that
    has been formed throughout the history
    '''
    def __init__(self):
        self._connections={}#a dictionary to store the connections as their key and 
                            #the the innovation number as their values
        self._innovations=0 #keeps track of the latest innovation number
    def get_innovation(self,key):
        if key in self._connections:
            return self._connections[key]
        else:
            self._innovations+=1
            self._connections[key]=self._innovations
            return self._connections[key]
    def print_obj(self):
        print("The latest innovation is {} and the connections are:\n {}".format(self._innovations,self._connections))


class NODE_GENE:
    '''
    this is the class that represents the nodes and their corresponding values
    it stores all the possible information about the nodes
    '''
    def __init__(self,inputs,outputs):
        super().__init__()
        self.node_gene={}
        self.input_k=[]
        self.output_k=[]
        self.inputs=copy.deepcopy(inputs)
        self.outputs=copy.deepcopy(outputs)
        self.distance={}
        self.create_nodes(self.inputs,self.outputs)
        
    
    def create_nodes(self,inputs,outputs):
        '''
        This function takes the inputs and the outputs and finally sets it to the 
        nodes with minimal structure
        '''
        START=0
        END=100
        self.node_gene[0]=1
        self.distance[0]=START
        for i in inputs:
            self.input_k.append(len(self.node_gene))
            id=len(self.node_gene)
            self.node_gene[id]=i
            self.distance[id]=START
        for i in outputs:
            self.output_k.append(len(self.node_gene))
            id=len(self.node_gene)
            self.node_gene[id]=i
            self.distance[id]=END

    def update_node(self,inputs,outputs):
        '''
        This function is used to update the input and the output nodes so that the next 
        training example can be trained upon
        '''
        self.inputs=copy.deepcopy(inputs)
        self.outputs=copy.deepcopy(outputs)
        for a,b in zip(self.input_k,self.inputs):
            self.node_gene[a]=b
        for a,b in zip(self.output_k,self.outputs):
            self.node_gene[a]=b

        

class CONNECTION_GENE(NODE_GENE):
    '''
    This Class handles all the funciton related with creating a connection gene from the information that
    has been inherited from the class NODE_GENE
    '''
    def __init__(self,inputs,outputs):
        super().__init__(inputs,outputs)
        self.connection_gene={}
        self.init_connection()
    def init_connection(self):
        '''
        This function forms a minimal structure with all the inputs connected to all the outputs
        '''
        for x,y in zip(self.input_k,self.output_k*len(self.input_k)):
            self.connection_gene[(x,y)]=[np.random.uniform(-1,1),True,ACC.get_innovation((x,y))]



class GENOME(CONNECTION_GENE):
    '''
    A class that basically consists of node gene and connection gene representing an individual
    which can reproduce, mutate and so on
    '''
    def __init__(self,inputs,outputs):
        super(GENOME,self).__init__(inputs,outputs)
        self.fitness=0
        self.propagate()

    def activation(self, x):
        '''
        this is the activation function for each node
        '''
        return np.maximum(0, x)

    def compute_fitness(self, result):
        '''
        this is the function that determines the performance of NEAT
        '''
        total = sum([abs(self.node_gene[x]-result[x]) for x in self.output_k])
        if total ==0:
            total = 1e-9
        self.fitness = 1.0/total

    def propagate(self,predict=False,inputs=None):
        '''
        This is the function that can perform forward pass as well as prediction
        If the input "predict==true" then the prediction is performed instead of propagation
        '''
        if inputs !=None:#this block of code is executed when inputs are given
            self.inputs=inputs
            self.update_node(copy.deepcopy(self.inputs),copy.deepcopy(self.outputs))
        #then we sort the distances so that the dependencies of nodes are handled
        dir_keys = sorted([(x, y) for y, x in self.distance.items()])
        sortd_dict = dict([(x, y) for (y, x) in dir_keys])
        temp = copy.deepcopy(self.node_gene)
        #we only go through the hidden and the output nodes
        for i in list(sortd_dict.keys())[len(self.input_k)+1:len(self.distance)]:
            temp[i] = 0
            for j in temp.keys():  
                #check if the connection exists
                if (j, i) not in self.connection_gene.keys():
                    continue
                #check if the connection is enabled
                if not self.connection_gene[(j, i)][1]:
                    continue
                temp[i] += temp[j]*self.connection_gene.get((j, i), 0)[0]
                if self.distance[i]<self.distance[j]:
                    break
            temp[i] = self.activation(temp[i])
        if not predict:#for the block that propagatee
            self.compute_fitness(temp)
        else:#the block that predict and displays the output
            for i in self.output_k:
                print(temp[i])

    
    def weight_mutate(self):
        '''
        this function is used to mutate the weights of the connections randomly 
        '''
        for a in self.connection_gene.keys():
            if np.random.uniform(0, 1) < 0.9:
                # random perturbation of weight
                self.connection_gene[a][0]+=float(np.random.uniform(-1,1)/1000)
            else:
                # complete change of weight
                self.connection_gene[a][0] = np.random.uniform(-1,1)    

    def node_mutate(self):
        '''
        this part of the program is used to mutate i.e. add a node in the existing connections
        the existing connection is disabled and two new connections are added
        '''        
        keys=list(self.connection_gene.keys())
        id=np.random.choice(np.arange(len(keys)),1).item()#randomly select a connection gene
        start,end=tuple(keys[id])#get the start and the end of the particular connection
        if self.connection_gene[(start,end)][1]:#check if the connection is enabled
            prev_weight=self.connection_gene[(start,end)][0]#save previous weights
            self.connection_gene[(start,end)][1] = False
            center = max(self.node_gene.keys())+1
            self.node_gene[center] = 0
            self.distance[center]=(self.distance[start]+self.distance[end])/2
            self.connection_gene[(start, center)] = [1, True, ACC.get_innovation((start,center))]
            self.connection_gene[(center, end)] = [prev_weight, True, ACC.get_innovation((center,end))]


    def connection_mutate(self):
        '''
        This function helps to form connection between the nodes at random 
        This does so by checking if the connection is already existing or if connection is viable
        '''
        # print("\nCOnnection mutation\n")
        # print("Possibilities--->",list(self.node_gene.keys()))
        # print("the possibilities are=",self.node_gene.keys())
        node1=np.random.choice(list(self.node_gene.keys()),1).item()
        node2=np.random.choice(list(self.node_gene.keys()),1).item()
        # print("selected node=",node1," ",node2)
        if not self.distance[node1]==self.distance[node2]:#this is to check condition that the destination node lies further away in network
            if self.distance[node1]<self.distance[node2]:
                if not (node1, node2) in self.connection_gene:
                    self.connection_gene[(node1, node2)] = [np.random.uniform(-1, 1), True, ACC.get_innovation((node1,node2))]
            else:
                if not (node2, node1) in self.connection_gene:
                     self.connection_gene[(node2, node1)] = [np.random.uniform(-1, 1), True, ACC.get_innovation((node2,node1))]
                    


    def mutate(self,rate_w=0.9,rate_n=0.01,rate_c=0.09):
        '''
        this funciton is used to call the differenty type of mutation functions
        '''
        rd=np.random.uniform(0,1)
        if rd<rate_w:
            self.weight_mutate()
        elif rd<(rate_w+rate_c):
            self.connection_mutate()
            
        elif rd<(rate_w+rate_n+rate_c):
            self.node_mutate()


    def print_obj(self):
        '''
        This is only a simple function to print the genes
        '''
        print(self.node_gene)
        print(self.connection_gene)   

    def copy(self):
        cop=GENOME(self.inputs,self.outputs)
        cop.inputs=copy.deepcopy(self.inputs) 
        cop.outputs=copy.deepcopy(self.outputs) 
        cop.input_k=copy.deepcopy(self.input_k)
        cop.output_k =copy.deepcopy(self.output_k)
        cop.node_gene=copy.deepcopy(self.node_gene) 
        cop.connection_gene=copy.deepcopy(self.connection_gene) 
        cop.distance=copy.deepcopy(self.distance) 
        cop.fitness=self.fitness
        return cop 

        
    
    


'''
#testing part
ACC=CONNECTIONS()
g=GENOME([1,2,3],[4])
print(g.connection_gene)
print(g.node_gene)

print(g.inputs)
g.propagate()
print(g.fitness)
ACC.print_obj()
for _ in range(3):
    g.mutate()
    print(g.connection_gene)
    print(g.node_gene)
    g.propagate()
    print(g.fitness)
    ACC.print_obj()
g2=g.copy()
g2.connection_gene[(1,4)][0]=1111
print("g2=",g2.print_obj())
print("g=",g.print_obj())
g2.propagate(True)
'''


class SPECIES:
    def __init__(self,gnm,top_r,w_o_cross):
        super().__init__()
        self.rep=gnm
        self.genomes=[gnm]
        self.adj_fit=[]
        self.nxt_gen=0
        self.top_r=top_r
        self.w_o_cross=w_o_cross
    
    def add_member(self,new_gnm):
        self.genomes.append(new_gnm)

    def sort_members(self):
        temp=sorted(self.genomes,key=lambda x: x.fitness,reverse=True)
        self.genomes=copy.deepcopy(temp)

    def calculate_adjfit(self):
        self.adj_fit=[x.fitness/len(self.genomes) for x in self.genomes]

    def cut_out(self):
        if len(self.genomes)>1:
            id=int(np.around(self.top_r*len(self.genomes)))
            self.genomes=self.genomes[:id]
            self.adj_fit=self.adj_fit[:id]

    def children_number(self,popn_fitness):
        self.nxt_gen=int(np.around((sum(self.adj_fit)/popn_fitness)*(1-self.w_o_cross)))

class POPULATION:
    def __init__(self,inputs,outputs,N,threshold_comp=3.0,top_r=0.5,w_o_cross=0.25,mutr=0.001):
        super().__init__()
        self.N=N
        self.comp_threshold=threshold_comp
        self.inputs=inputs
        self.outputs=outputs
        self.species=[]
        self.sp_fitness=[]
        self.top_r=top_r
        self.w_o_cross=w_o_cross
        self.mutr=mutr
        # self.members=[GENOME(self.inputs,self.outputs) for _ in range(self.N)]
    
    def create_species(self,members=None):
        if members==None:
            members=[GENOME(self.inputs,self.outputs) for _ in range(self.N)]
        for genome in members:
            placed=False
            if self.species==[]:
                self.species.append(SPECIES(genome,self.top_r,self.w_o_cross))
                continue
            for individual in self.species:
                if self.compatibility_distance(individual.rep,genome)<self.comp_threshold:
                    individual.add_member(genome)
                    placed=True
                    break
            if not placed:
                self.species.append(SPECIES(genome,self.top_r,self.w_o_cross))
                    
        popn_fitness=np.mean([(x.fitness)/len(a.genomes) for a in self.species for x in a.genomes])
        # print("one time created species")
        for x in self.species:
            # print("the number placed originally is=",len(x.genomes))
            x.sort_members()
            x.calculate_adjfit()
            # print("First adj fit=\n",x.adj_fit)
            self.sp_fitness.append(sum(x.adj_fit))
            # print("One more species:",self.sp_fitness)
            x.children_number(popn_fitness)
            # print("The number of same species in next gen is=",x.nxt_gen)
            x.cut_out()
            # print([a.connection_gene for a in x.genomes[:2]])
            # print()
            

    
    def compatibility_distance(self,g1,g2,c1=1.0,c2=1.0,c3=0.4):
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
            if not i in g1_inn and i<max(g1_inn):
                d+=1

        # the code below calculates the average difference of weights
        diff=[]
        for a,b in g1.connection_gene.items():
            if a in g2.connection_gene:
                diff.append(abs(b[0]-g2.connection_gene[a][0]))
        w=np.mean(diff)
        
        if len(g1_inn)>=len(g2_inn):
            N=len(g1_inn)
        else:
            N=len(g2_inn)
        if N<20:
            N=1
        #FINALLY the formula to calculate the compatibility distance
        comp_dist=(c1*e)/N+(c2*d)/N+(c3*w)
        return comp_dist

    def crossover(self, g1,g2):
        '''
        this function takes the input as the parent genomes and produces 
        a child more alike to the fit genome
        '''
        f1=g1.fitness
        f2=g2.fitness
        # if f1==f2:
        #     print("\nsame fitness encountered\n")
        #     g1.print_obj()
        #     g2.print_obj()
        offspring=None
        if f1>=f2:
            # print("We took f1")#we copy the gene with higher fitness
            offspring=copy.deepcopy(g1)
            for a,b in offspring.connection_gene.items():
                flag=False                 #used to enable the already disabled gene
                if a in g2.connection_gene:#this if condition tallies all the homologous gene
                    if np.random.uniform(0,1)>0.5:
                        offspring.connection_gene[a][0]=g2.connection_gene[a][0]
                    if not g2.connection_gene[a][1]:# if false, the condition to enable the disabled gene
                        flag=True
                        if np.random.uniform(0,1)>0.75:
                            offspring.connection_gene[a][1]=True
                if not flag and not b[1]:# this condtion works for the disjoint and the excess genes
                                         # used to enable the disabled gene
                    if np.random.uniform(0,1)>0.75:
                        offspring.connection_gene[a][1]=True                    
            return copy.deepcopy(offspring)
                    
        else:
            # print("We took f2")
            offspring=copy.deepcopy(g2)

            for a,b in offspring.connection_gene.items():
                flag=False                 #used to enable the already disabled gene
                if a in g1.connection_gene:#this if condition tallies all the homologous gene
                    if np.random.uniform(0,1)>0.5:
                        offspring.connection_gene[a][0]=g1.connection_gene[a][0]
                    if not g1.connection_gene[a][1]:# if false, the condition to enable the disabled gene
                        flag=True
                        if np.random.uniform(0,1)>0.75:
                            offspring.connection_gene[a][1]=True
                if not flag and not b[1]:# this condtion works for the disjoint and the excess genes
                                         # used to enable the disabled gene
                    if np.random.uniform(0,1)>0.75:
                        offspring.connection_gene[a][1]=True
            return copy.deepcopy(offspring)
        

    def generate_new_popn(self,inputs,outputs):
        # print("\ngenerate called")
        members=[]
        # print(self.species)
        for sp in self.species:
            first_copy=True
            for _ in range(sp.nxt_gen):
                child=None
                if first_copy and len(sp.genomes[0].connection_gene)>5:
                    first_copy=False
                    child=copy.deepcopy(sp.genomes[0])              
                    child.update_node(inputs,outputs)
                    child.propagate()
                    members.append(copy.deepcopy(child))
                    continue
                first_copy=False
                # print("The adj fit from generate:",[el/sum(sp.adj_fit) for el in sp.adj_fit])
                par1,par2=tuple(np.random.choice(sp.genomes,2,p=[el/sum(sp.adj_fit) for el in sp.adj_fit]))
                # print("The two parents:",par1.connection_gene,par2.connection_gene)
                if self.similar(par1,par2):
                    # print("---same element detected---")
                    continue
                child=self.crossover(copy.deepcopy(par1),copy.deepcopy(par2))
                if np.random.uniform(0,1)<self.mutr:
                    child.mutate()
                child.update_node(inputs,outputs)
                child.propagate()
                # child.print_obj()
                members.append(copy.deepcopy(child))
                # print("Member contents::\n")
                # for i in members:
                #     i.print_obj()
        while(len(members)<self.N):
            child=None
            child_sp=None
            child_sp=copy.deepcopy(np.random.choice(self.species,1,p=[x/sum(self.sp_fitness) for x in self.sp_fitness]).item())
            
            child=copy.deepcopy(np.random.choice(child_sp.genomes,1,p=[x/sum(child_sp.adj_fit) for x in child_sp.adj_fit]).item())
            # if np.random.uniform(0,1)<self.mutr:
            child.mutate()
            child.update_node(inputs,outputs)
            child.propagate()
            # child.print_obj()
            members.append(child)
            # print("Member contents::\n")
            # for i in members:
            #     i.print_obj()
        # print("from new popn creator:",len(members))
        # print(members)
        # for i in members:
        #     i.print_obj()
        new_popn=POPULATION(copy.deepcopy(inputs),copy.deepcopy(outputs),self.N)
        
        new_popn.create_species(copy.deepcopy(members))
        return new_popn

    def get_fittest_genome(self):
        fittest=copy.deepcopy(self.species[0].genomes[0])
        for sp in self.species:
            if sp.genomes[0].fitness>fittest.fitness:
                fittest=copy.deepcopy(sp.genomes[0])

        return fittest
    
    def similar(self,g1,g2):
        if abs(g1.fitness-g2.fitness)<1e-5 and g1.distance==g2.distance and g1.connection_gene==g2.connection_gene:
            return True
        else:
            return False


# ACC=CONNECTIONS()
# p=POPULATION([1,2,3],[4],10)
# # p.members[0].print_obj()
# # p.members[1].print_obj()
# p.create_species()
# print("total fitness of speciess=",p.sp_fitness)
# for i in p.species:
#     print(i.nxt_gen)
#     print(i.adj_fit)
#     for x in i.genomes:
#         x.print_obj()
#         print(x.fitness)

# print("Now the real test")
# p=p.generate_new_popn([1,2,3],[4])
# print("total fitness of speciess=",p.sp_fitness)
# l=0
# for i in p.species:
#     print(i.nxt_gen)
#     print(i.adj_fit)
#     l+=len(i.genomes)
#     for x in i.genomes:
#         x.print_obj()
#         print(x.fitness)
# print("the total genomes is ===",l)
        
######3testing phase

ACC=CONNECTIONS()
inputs=[[1],[2],[3],[4],[5],[6],[7],[8],[9]]
outputs=[[2],[3],[4],[5],[6],[7],[8],[9],[10]]
# inputs=[[0,0],[0,1],[1,0],[1,1]]
# outputs=[[0],[1],[1],[0]]
gen_count=0
N=150
p=POPULATION(inputs[gen_count],outputs[gen_count],N)
p.create_species()

while gen_count<=10:
    gen_count+=1
    print("GENERATION:",gen_count)
    p=p.generate_new_popn(inputs[int(gen_count%len(inputs))],outputs[int(gen_count%len(outputs))])
    
fin_gnm=p.get_fittest_genome()
fin_gnm.print_obj()
for x in inputs:
    fin_gnm.propagate(True, x)


