from pyspark import SparkContext
import networkx as nx
import os
from itertools import groupby, combinations
from operator import add
import time
import random
import sys
import numpy as np
from collections import defaultdict
from operator import itemgetter


class GirvanNewman:
    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.G = nx.Graph()

    def compute_betweeness(self):
        for node in self.G.nodes:
            start = time.time()
            self.compute_from_node(node)
            print("each_node",time.time()- start)


    def compute_from_node(self,node):
        ParentDict = defaultdict(set)
        EdgeBetweeness = defaultdict(int) # value -> (parents,level)
        visited =[]
        visited.append(frozenset([node]))
        cqueue = list(self.G.neighbors(node))
        for val in cqueue:
            ParentDict[val].add(node)
        pqueue = set(cqueue)
        cqueue = set()
        flag = True
        level = 1
        value = dict.fromkeys(self.G.nodes,1.0)

        while flag:
            for i in pqueue:
                for g in list(self.G.neighbors(i)):
                    if g not in pqueue and g not in visited[level-1]:
                        if g not in cqueue:
                            cqueue.add(g)
                        ParentDict[g].add(i)
            visited.append(frozenset(pqueue))
            if len(cqueue) == 0:
                flag = False
            else:
                pqueue = set().union(cqueue)
                cqueue = set()
            level += 1

        for i in range(level -1,-1,-1):
            if i == level -1:
                for j in visited[i]:
                    value[j] = 1
            else :
                for k in visited[i+1]:
                    temp = float(value[k])/len(ParentDict[k])
                    for j in ParentDict[k]:
                        value[j] += temp
                        EdgeBetweeness[tuple(sorted([j,k]))] += temp
        value[node] -=1
        #print("returning",node)
        return list(EdgeBetweeness.items())




    def run(self):
        start = time.time()
        sc = SparkContext("local[8]", "ratings")
        text = sc.textFile(inputFile)
        header = text.first() #extract header
        text = text.filter(lambda row : row != header)

        UsersMovies = text.map(lambda line: line.split(',')).map(lambda x:  (int(x[0]),int(x[1]))).groupByKey().sortByKey()
        UserDict = dict(UsersMovies.collect())

        EdgesForGraph =[]

        pairs = list(combinations(UserDict.keys(),2))
        for i in pairs:
            if len(set(UserDict[i[0]]) & set(UserDict[i[1]])) >= 9 :
                EdgesForGraph.append(sorted(i))

        print(len(EdgesForGraph))
        self.G.add_edges_from(EdgesForGraph)
        del(UserDict)
        print(len(self.G.nodes))

        Betweeness = UsersMovies.flatMap(lambda x: self.compute_from_node(x[0])).reduceByKey(add).mapValues(lambda v: float(v/2.0)).sortByKey().collect()
        Betweeness = sorted(Betweeness,key=lambda tup: tup[1],reverse=True)

        print("time for Betweeness calculation ", time.time() - start)


        #print(self.m2)
        Outdegree =[]
        nodelist =[]
        for i in range(1,672):
            Outdegree.append(self.G.degree[i])
            nodelist.append(i)

        Outdegree = np.array(Outdegree)
        self.m2 = sum(Outdegree)
        print("now ",self.m2)

        Outdegree = Outdegree.reshape(-1,1)*Outdegree.reshape(1,-1)
        Outdegree = Outdegree/float(self.m2)
        #adj = nx.adjacency_matrix(self)
        self.A =np.array(nx.to_numpy_matrix(self.G,nodelist)) - Outdegree
        print(self.A.shape)
        del(Outdegree)
        del(nodelist)
        no_removed = 0

        print("preprocessing", time.time() - start)

        self.max_components = list(nx.connected_components(self.G))
        self.max_modularity = -1.1
        prev_no_components = 1

        for no_removed in range(0,len(Betweeness)):
            self.G.remove_edge(Betweeness[no_removed][0][0],Betweeness[no_removed][0][1])
            components = list(nx.connected_components(self.G))
            no_components = len(components)
            if no_components > prev_no_components:
                curr_modularity = self.compute_modularity(components)
                #print("modularity",curr_modularity,no_components)
                if curr_modularity > self.max_modularity:
                    self.max_modularity,self.max_components = curr_modularity,components
                prev_no_components = no_components
            Betweeness = UsersMovies.flatMap(lambda x: self.compute_from_node(x[0])).reduceByKey(add).mapValues(lambda v: float(v/2.0)).sortByKey().collect()
            Betweeness = sorted(Betweeness,key=lambda tup: tup[1],reverse=True)



        print(len(self.max_components),self.max_modularity)
        sol = []
        for i in range(len(self.max_components)):
            current = sorted(list(self.max_components[i]))
            sol.append((min(current),current))
        sol.sort(key=itemgetter(0))
        with open('Tuhina_Kumar_Community.txt','w') as f:
            for i in range(len(sol)):
                str1='['
                for j in range(len(sol[i][1])):
                    str1 += str(sol[i][1][j]) +', '
                str1 = str1.strip(', ')
                str1 += ']'+'\n'
                f.write(str1)


    def compute_modularity(self, components):
        components = list(components)
        modularity = 0.0
        for c in range(len(components)):
            temp = np.array(sorted(list(components[c])))
            for i in range(len(temp)):
                temp2 = np.take(self.A[temp[i]-1],temp -1)
                modularity += np.sum(temp2)


        return float(modularity)/float(self.m2)

    '''
    def compute_modularity(self, components):
        modularity = 0.0
        for c in range(len(components)):
            comp_list = list(components[c])
            if len(comp_list) == 1:
                modularity += - float(self.Outdegree[comp_list[0]]*self.Outdegree[comp_list[0]])/self.m2
            else:
                temp = 0.0
                for i in range(0,len(comp_list)):
                    for j in range(i+1,len(comp_list)):
                        a = comp_list[i]
                        b = comp_list[j]
                        temp += self.A[i-1][j-1] - float(self.Outdegree[a]*self.Outdegree[b])/self.m2
                        #if self.A[a][b] != self.Gcopy.has_edge(a+1,b+1):
                        #    print('problem',a+1,b+1)
                modularity +=temp*2
        return float(modularity)/float(self.m2)
    '''




if __name__ == '__main__':
    start = time.time()
    inputFile = sys.argv[1]

    gn = GirvanNewman(inputFile)
    gn.run()
    print(time.time()- start)
