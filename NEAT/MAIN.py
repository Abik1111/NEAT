import DNA
from DNA import Genome
import POPN
from POPN import Population
import numpy as np

INPUTS=[[1],[2],[3],[4],[5],[6],[7],[8],[9]]
# INPUTS=[[0,0],[0,1],[1,0],[1,1]]
# OUTPUTS=[[0],[1],[1],[0]]
OUTPUTS=[[2],[3],[4],[5],[6],[7],[8],[9],[10]]
population_size=1000
gen_count=0
MAXIMUM_FITNESS=1000
###############SETUP#####################
print(INPUTS[0])
print(OUTPUTS[0])
p=Population(INPUTS[0].copy(),OUTPUTS[0].copy(),population_size)

#########################################
print("Enter the dragon")
while gen_count<8:
    gen_count+=1
    print("present generation:::::",gen_count)
    print("INPUT->",p.inputs)
    print("OUTPUT->",p.outputs)
    p.speciate()
    check=p.max_fitness()
    print("MAX FITNESS->",p.max_fitness())
    p.members_in_nxt_gen()
    # print("List of representatives:::",p.rep_list)
    # print()
    # print("the dictionary of species:::",p.species)
    # print()
    # print("the adjusted fitnesses:::",p.adjusted_fitness)
    # print()
    # print("number of genomes in next generation::",p.next_generation)
    # print()
    
    p=p.generate_new_population(0.05,INPUTS[gen_count%9].copy(),OUTPUTS[gen_count%9].copy())
    
    if check>=MAXIMUM_FITNESS:
        break
print("end of the loop")
fitnesses=[a.fitness for a in p.population]
maxim=np.argmax(fitnesses)
p.population[maxim].print_obj()
print("next")
p.population[maxim].predict([1])
p.population[maxim].predict([8])
p.population[maxim].predict([11])
# p.population[maxim].predict([0,0])





