import pandas as pd
import numpy
import random
import time

def AND(a,b):
    return (a & b)

def OR(a,b):
    return (a | b)

def XOR(a,b):
    return (a != b)

def NAND(a,b):
    return ~(a & b)

def NOR(a,b):
    return ~(a | b)

def XNOR(a,b):
    return (a == b)

def read__csv():
    df = pd.read_csv ('truth_table.csv')
    df = df.replace([True, False], [1,0])
    return df

def crossoverUtil(a,b,columns):
    line = random.randint(1,columns-1)
    t1 = []
    t2 = []
    for i in range(len(a)):
        if i < line:
            t1.append(a[i])
            t2.append(b[i])
        else:
            t1.append(a[i])
            t2.append(b[i])
    return t1,t2

def crossover(population,p,pc,pm,columns):
    new_population = []
    for i in range(int(len(population)/2)):
        index1,index2 = selection(population,p)
        if random.random() < pc:
            new1, new2 = crossoverUtil(population[index1],population[index2],columns)
            new1 = mutationUtil(new1,pm,columns)
            new2 = mutationUtil(new2,pm,columns)
            new_population.append(new1)
            new_population.append(new2)
        else:
            new_population.append(population[index1])
            new_population.append(population[index2])

    return new_population

def mutationUtil(a,pm,columns):
    if random.random() < pm:
        bit1 = random.randint(0,columns-1)
        b1 = random.randint(1,6)
        a[bit1] = b1
    return a

def generate_population(columns):
    population = []
    n = 200
    for i in range(n):
        temp = []
        for j in range(columns):
            temp.append(random.randint(1,6))
        population.append(temp)
    return population

def calc_fitnessUtil(row,x):
    arr = x.tolist()
    columns = len(row.columns) - 2
    inp = row['Input1']
    out = row['Output']
    temp = []
    for k in range (columns):
        temp.append(row[f'Input{k+2}'])
    i = 0
    for gate in arr:
        if gate == 1:
            inp = AND(inp,temp[i])
        if gate == 2:
            inp = OR(inp,temp[i])
        if gate == 3:
            inp = XOR(inp,temp[i])
        if gate == 4:
            inp = NAND(inp,temp[i])
        if gate == 5:
            inp = NOR(inp,temp[i])
        if gate == 6:
            inp = XNOR(inp,temp[i])
        i += 1

    return (inp == out).sum()

def calc_fitness(population,df):
    pop = pd.DataFrame(population)
    fit = pop.apply(lambda x: calc_fitnessUtil(df,x), axis=1)
    fitness = fit.tolist()
    return [x**3 for x in fitness]

def calc_p(fitness):
    p = []
    sum_fitness = sum(fitness)
    for i in fitness:
        p.append(i/sum_fitness)
    return p

def selection(population,pc):
    index = numpy.random.choice(numpy.arange(0, len(population)),2,replace=False, p=pc)
    return index

def main():
    df = read__csv()
    columns = len(df.columns) - 2
    population = generate_population(columns)
    fitness = calc_fitness(population,df)
    p = calc_p(fitness)
    pc = 0.85
    pm = 0.1
    i = 0
    prev = 0
    t1 = time.process_time()
    while True:
        population = crossover(population,p,pc,pm,columns)
        fitness = calc_fitness(population,df)
        p = calc_p(fitness)
        max_fitness = max(fitness)
        if prev == max_fitness:
            pc += 0.01
            pm += 0.01
        prev = max_fitness
        i += 1
        if 1024**3 == max_fitness:
            index = fitness.index(max_fitness)
            t2 = time.process_time()
            print("Answer found!")
            print("time: ", t2-t1)
            print("Generation: ",i)
            print("Gates: ",end="")
            for k in population[index]:
                if k == 1: print("AND ",end="")
                if k == 2: print("OR ",end="")
                if k == 3: print("XOR ",end="")
                if k == 4: print("NAND ",end="")
                if k == 5: print("NOR ",end="")
                if k == 6: print("XNOR ",end="")
            print()
            break
            
main()
