import random
import operator
import itertools
import numpy
import time
import math
import sys
from pyspark import SparkContext, SparkConf, StorageLevel
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#conf = SparkConf().setAppName("GP over Spark cluster")
#sc = SparkContext(conf=conf)

conf = SparkConf().setAppName("GP over Spark")
if not('sc' in globals()):
    sc = SparkContext(conf=conf)


## Preparing RDDs, cluster setup
TrainingRDD = sc.textFile('hdfs://83.212.76.230:9000/user/user/input/HIGGS.csv').map(lambda line : [float(x) for x in line.split(',')]).persist(StorageLevel.DISK_ONLY)
TestRDD = sc.textFile('hdfs://83.212.76.230:9000/user/user/input/HIGGS.csv').map(lambda line : [float(x) for x in line.split(',')]).persist(StorageLevel.DISK_ONLY)

## Preparing RDDs, single node setup, HIGGS.csv must be in the same dir as this file!
#TrainingRDD = sc.textFile('HIGGS.csv').map(lambda line : [float(x) for x in line.split(',')]).persist(StorageLevel.DISK_ONLY)
#TestRDD = sc.textFile('HIGGS.csv').map(lambda line : [float(x) for x in line.split(',')]).persist(StorageLevel.DISK_ONLY)


print("Training dataset size:{}".format(TrainingRDD.count()))
print("Test dataset size:{}".format(TestRDD.count()))

toolbox = base.Toolbox()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def evalHiggsBase(individual):
    func = toolbox.compile(expr=individual)
    result = sum(TrainingRDD.map(lambda line: bool(sigmoid(func(*line[1:])) > 0.5) is bool(line[0])).collect())
    return result,


# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


# Define a protected division function
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


# defined a new primitive set for strongly typed GP (28 float input attributes from Higgs dataset)
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 28), bool, "IN")

# Functions set

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)

# logic operators
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# constants float in [0,1] and boolean constants False, True
pset.addEphemeralConstant("rand1", lambda: random.random(), float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

# Set the best fitness as the max fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# other GP parameters : selection, mutation, ...
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalHiggsBase)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# Evaluating a population on a single exemplar
def evalPop(individuals, line):
    return [bool(sigmoid(i(*[float(v) for v in line[1:]])) > 0.5) is bool(float(line[0])) for i in individuals]


# redefinig the GP loop from DEAP to use Spark
def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    ii = [toolbox.compile(f) for f in invalid_ind]
    results = TrainingRDD.map(lambda line: evalPop(ii, line))
    fitnesses = results.reduce(lambda v1, v2: list(map(operator.add, v1, v2)))
    fitnesses = [tuple([vf]) for vf in fitnesses]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print
        logbook.stream
    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        ii = [toolbox.compile(f) for f in invalid_ind]
        results = TrainingRDD.map(lambda line: evalPop(ii, line))
        fitnesses = results.reduce(lambda v1, v2: list(map(operator.add, v1, v2)))
        fitnesses = [tuple([vf]) for vf in fitnesses]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(offspring)
        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print
            logbook.stream
    return population, logbook


# Redefinition
algorithms.eaSimple = eaSimple

def confusionMatrix(func,dataset):
    confusion_matrix = [[0.0,0.0],[0.0,0.0]]
    predictions=dataset.map(lambda line:[bool(sigmoid(func(*line[1:]))>0.5),bool(line[0])]).collect()
    for line in predictions:
        predicted = line[0]
        real = line[1]
        if(predicted == real and real):
            confusion_matrix[0][0]+=1
        elif(predicted == real and not(real)):
            confusion_matrix[1][1]+=1
        elif(predicted != real and real):
            confusion_matrix[1][0]+=1
        elif(predicted != real and not(real)):
            confusion_matrix[0][1]+=1
    return confusion_matrix


def main():
    log = open('higgs.log', 'a')
    print("logs are record in file higgs.log")
    try:
        random.seed()
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        popSize = 35
        crossover_prob = 1
        mutation_prob = 0.4
        nbGen = 5
        pop = toolbox.population(n=popSize)
        log.write('\nRun Started\n')
        startTime = time.time()
        p, lbook = algorithms.eaSimple(pop, toolbox, crossover_prob, mutation_prob, nbGen, stats, halloffame=hof, verbose=True)
        endTime = time.time()
        log.write(lbook.stream)
        log.write("\nLearning time : {} seconds".format(endTime - startTime))
        expr = hof[0]
        func = toolbox.compile(expr)
        confusion_matrix = confusionMatrix(func, TrainingRDD)
        resultTraining = confusion_matrix[0][0] + confusion_matrix[1][1]
        log.write("\n" + str(expr))
        log.write("\nfitness of best individual against total training set :{}/{}={}".format(resultTraining, numpy.sum(confusion_matrix), float(resultTraining) / numpy.sum(confusion_matrix)))
        cm = '\n'.join('\t'.join('%0.3f' % x for x in y) for y in confusion_matrix)
        log.write("\n" + cm)
        TP = confusion_matrix[0][0]
        TN = confusion_matrix[1][1]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        log.write("\naccuracy ={}, TPR={}, FPR={}".format((TP + TN) / (TP + TN + FP + FN), TP / (TP + FN), FP / (FP + TN)))
        confusion_matrix = confusionMatrix(func, TestRDD)
        resultTest = confusion_matrix[0][0] + confusion_matrix[1][1]
        log.write("\nfitness of best individual against test set :{}/{}={}".format(resultTest, numpy.sum(confusion_matrix),float(resultTest) / numpy.sum(confusion_matrix)))
        cm = '\n'.join('\t'.join('%0.3f' % x for x in y) for y in confusion_matrix)
        log.write("\n" + cm)
        TP = confusion_matrix[0][0]
        TN = confusion_matrix[1][1]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        log.write("\naccuracy ={}, TPR={}, FPR={}\n".format((TP + TN) / (TP + TN + FP + FN), TP / (TP + FN), FP / (FP + TN)))
        log.close()
        return pop, stats, hof
    except Exception as e:
        log.write("Exception !: {}\n".format(e))
        log.close()


if __name__ == "__main__":
    main()

