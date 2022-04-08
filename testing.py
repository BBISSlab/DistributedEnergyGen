import numpy as np
import random
random.seed(100)

def selection(population, fit_chromosomes, generation):
    fitnessscore=[]
    for chromosome in population:
        individual_fitness=100-abs(30-(chromosome[0]+2*chromosome[1]-3*chromosome[2]+chromosome[3]+4*chromosome[4]+chromosome[5]))
        fitnessscore.append(individual_fitness)

    total_fitness=sum(fitnessscore)
    print('Total fitness: ', total_fitness)
    score_card=list(zip(fitnessscore,population))
    score_card.sort(reverse=True)

    for individual in score_card:
        if individual[0]==100:
            if individual[1] not in fit_chromosomes:
                fit_chromosomes.append(individual[1])
    
    score_card=score_card[:4]
    score, population=zip(*score_card)
    return list(population)

def crossover(population):
    random.shuffle(population)
    fatherchromosome=population[:2]
    motherchromosome=population[2:]
    children=[]
    for i in range(len(fatherchromosome)):
        crossoversite=random.randint(1,5)
        fatherfragments=[fatherchromosome[i][:crossoversite],fatherchromosome[i][crossoversite:]]
        motherfragments=[motherchromosome[i][:crossoversite],motherchromosome[i][crossoversite:]]
        firstchild=fatherfragments[0]+motherfragments[1]
        children.append(firstchild)
        secondchild=motherfragments[0]+fatherfragments[1]
        children.append(secondchild)
    return children

def mutation(population):
    mutatedchromosomes=[]
    for chromosome in population:
        mutation_site=random.randint(0,5)
        chromosome[mutation_site]=random.randint(1,9)
        mutatedchromosomes.append(chromosome)
    return mutatedchromosomes

def get_fit_chromosomes(generations):
    population=[[random.randint(1,9) for i in range(6)] for j in range(6)]
    fit_chromosomes=[]
    for generation in range(generations):
        generation+=1
        print('Generation:', generation)
        population=selection(population, fit_chromosomes, generation)
        crossover_children=crossover(population)
        population=population+crossover_children
        mutated_population=mutation(population)
        population=population+mutated_population
        #random.shuffle(population)

    return fit_chromosomes

'''solution=get_fit_chromosomes(20)
print('-----------Solution-----------')
print(solution)
print(len(solution))'''

class StatePoint:
    
    def __init__(self, position, temperature=None, temperature_min=0, temperature_max=100, 
        pressure=1, enthalpy=None, fluid_type=None, mass_flowrate=None):
        self.position = position
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.pressure = pressure
        self.enthalpy = enthalpy
        self.fluid_type = fluid_type
        self.mass_flowrate = mass_flowrate
    
    def __repr__(self):
        attrs = ['position', 'fluid_type', 'mass_flowrate', 
            'temperature', 'temperature_min', 'temperature_max', 
            'pressure', 'enthalpy']
        return ('Statpoint: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr))
                                              for attr in attrs))




def abc_test():
    from ypstruct import structure

    p1 = structure()
    p2 = structure()
    p3 = structure()
    p4 = structure()
    p5 = structure()
    p6 = structure()
    p7 = structure()
    p8 = structure()
    p9 = structure()
    p10 = structure()
    p11 = structure()
    p12 = structure()
    p13 = structure()
    p14 = structure()
    p15 = structure()
    p16 = structure()
    p17 = structure()
    p18 = structure()
    p19 = structure()
    p20 = structure()
    p21 = structure()
    
    # Temperature
    p1.temperature = 80
    p4.temperature = 4
    p8.temperature = p1.temperature
    p13.temperature = None#T_dry_bulb
    p16.temperature = 25
    p18.temperature = 12
    p19.temperature = 6
    p20.temperature = 90.6
    p21.temperature = 85

    # Pressure
    '''P_g = Generator_.generator_pressure(p1.temperature)
    P_c = P_g
    P_e = Evaporator_.evaporator_pressure(p4.temperature)
    P_a = P_e'''

    # Insert known pressures
    upper_vessel = [p1, p2, p6, p7, p8, p9]
    lower_vessel = [p3, p4, p5, p10]
    for i in upper_vessel:
        i.pressure = None
    for i in lower_vessel:
        i.pressure = None
    p13.pressure = None
    p15.pressure = None

    # Insert known constraints
    water_liquid = [p2, p3, p11, p12, p14, p16, p17, p18, p19, p20, p21]
    water_vapor = [p1, p4]
    strongLiBr = [p8, p9, p10]
    weakLiBr = [p5, p6, p7]
    h_air = [p13, p15]
    for i in water_liquid:
        i.fluidtype = 'liquid_water'
        i.temp_min = 1
        i.temp_max = 100
    '''for i in water_vapor:
        i.fluidtype = 'vapor_water'
        i.temp_min = None
        i.temp_max = None
    for i in strongLiBr:
        i.fluidtype = 'liquid_water'
    for i in water_liquid:
        i.fluidtype = 'liquid_water'    
    p13.pressure = None
    p15.pressure = None'''

    print(p2)
    
value = (np.random.uniform(size = 1, low = 7.2, high = 7.5))
print(value[0])