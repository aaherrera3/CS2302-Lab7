# Lab 7
# Programmed by Anthony Herrera
# Last modified April, 28, 2019

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import random
import time


def DisjointSetForest(size):
    return np.zeros(size, dtype=np.int) - 1


def dsfToSetList(S):
    # Returns aa list containing the sets encoded in S
    sets = [[] for i in range(len(S))]
    for i in range(len(S)):
        sets[find(S, i)].append(i)
    sets = [x for x in sets if x != []]
    return sets


def find(S, i):
    # Returns root of tree that i belongs to
    if S[i] < 0:
        return i
    return find(S, S[i])


def find_c(S, i):  # Find with path compression
    if S[i] < 0:
        return i
    r = find_c(S, S[i])
    S[i] = r
    return r


def union(S, i, j):
    # Joins i's tree and j's tree, if they are different
    ri = find(S, i)
    rj = find(S, j)
    if ri != rj:
        S[rj] = ri


def union_c(S, i, j):
    # Joins i's tree and j's tree, if they are different
    # Uses path compression
    ri = find_c(S, i)
    rj = find_c(S, j)
    if ri != rj:
        S[rj] = ri


def union_by_size(S, i, j):
    # if i is a root, S[i] = -number of elements in tree (set)
    # Makes root of smaller tree point to root of larger tree
    # Uses path compression
    ri = find_c(S, i)
    rj = find_c(S, j)
    if ri != rj:
        if S[ri] > S[rj]:  # j's tree is larger
            S[rj] += S[ri]
            S[ri] = rj
        else:
            S[ri] += S[rj]
            S[rj] = ri


def draw_dsf(S):
    scale = 30
    fig, ax = plt.subplots()
    for i in range(len(S)):
        if S[i] < 0:  # i is a root
            ax.plot([i * scale, i * scale], [0, scale], linewidth=1, color='k')
            ax.plot([i * scale - 1, i * scale, i * scale + 1], [scale - 2, scale, scale - 2], linewidth=1, color='k')
        else:
            x = np.linspace(i * scale, S[i] * scale)
            x0 = np.linspace(i * scale, S[i] * scale, num=5)
            diff = np.abs(S[i] - i)
            if diff == 1:  # i and S[i] are neighbors; draw straight line
                y0 = [0, 0, 0, 0, 0]
            else:  # i and S[i] are not neighbors; draw arc
                y0 = [0, -6 * diff, -8 * diff, -6 * diff, 0]
            f = interpolate.interp1d(x0, y0, kind='cubic')
            y = f(x)
            ax.plot(x, y, linewidth=1, color='k')
            ax.plot([x0[2] + 2 * np.sign(i - S[i]), x0[2], x0[2] + 2 * np.sign(i - S[i])],
                    [y0[2] - 1, y0[2], y0[2] + 1], linewidth=1, color='k')
        ax.text(i * scale, 0, str(i), size=20, ha="center", va="center",
                bbox=dict(facecolor='w', boxstyle="circle"))
    ax.axis('off')
    ax.set_aspect(1.0)

def draw_maze(walls,maze_rows,maze_cols,cell_nums=False):
    fig, ax = plt.subplots()
    for w in walls:
        if w[1]-w[0] ==1: #vertical wall
            x0 = (w[1]%maze_cols)
            x1 = x0
            y0 = (w[1]//maze_cols)
            y1 = y0+1
        else:#horizontal wall
            x0 = (w[0]%maze_cols)
            x1 = x0+1
            y0 = (w[1]//maze_cols)
            y1 = y0
        ax.plot([x0,x1],[y0,y1],linewidth=1,color='k')
    sx = maze_cols
    sy = maze_rows
    ax.plot([0,0,sx,sx,0],[0,sy,sy,0,0],linewidth=2,color='k')
    if cell_nums:
        for r in range(maze_rows):
            for c in range(maze_cols):
                cell = c + r*maze_cols
                ax.text((c+.5),(r+.5), str(cell), size=10,
                        ha="center", va="center")
    ax.axis('off')
    ax.set_aspect(1.0)

def wall_list(maze_rows, maze_cols):
    # Creates a list with all the walls in the maze
    w =[]
    for r in range(maze_rows):
        for c in range(maze_cols):
            cell = c + r*maze_cols
            if c!=maze_cols-1:
                w.append([cell,cell+1])
            if r!=maze_rows-1:
                w.append([cell,cell+maze_cols])
    return w


def NumberOfSets(S):# will count the number of -1 that show diffrent sets
    count = 0
    for x in S:
        if x == -1:
            count += 1
    return count


class Graph:                       # creates a graph object with vertexes
    def __init__(self,vertices):
        self.vertices = vertices
        self.graph = []
        for v in range(vertices):  # append unvisited vertexes
            self.graph.append([])


def add_edge(G,v1,v2):      # add edges to the graph
    G.graph[v1].append(v2)


def BFS(G,start):
    visited = [False] * (len(G.graph))    # unvisited array
    queue = []
    queue.append(start)      # Queue off unvisited vertexes
    visited[start] = True
    while queue:
        start = queue.pop(0)
        print(start,'-->', end='')
        for i in G.graph[start]:     # add unvisited vertex to queue 
            if visited[i] == False:
                queue.append(i)
                visited[i] = True


def DFS(G,start):
    visited = []  # creates array for vested vertexes
    stack = [start]  # stack that hold vertexes to be added
    while stack:
        start = stack.pop()
        visited.append(start)
        print(start,'-->', end='')
        for i in G.graph[start]:  # append unvisited vertexes
            stack.append(i)


def DFSR(G,start,visited=None):
    if visited == None:  # creates array of vertexes that have been visited
        visited = []
    visited.append(start)    # appends vertex to visited list
    for i in G.graph[start]:  # traverse and add unvisited vertexes
        if i not in visited:
            DFSR(G,i,visited)
    return visited











plt.close("all")
maze_rows = int(input('Enter number of x axis.\n'))
maze_cols = int(input('Enter number of y axis.\n'))
graph = Graph(maze_rows*maze_cols)
print('The number of cells (n) in the maze is ',maze_cols*maze_rows)
wall_remover = int(input('Enter the number of walls (m) to remove.\nA path from source to destination is not guaranteed to exist (when m < n − 1)\nThe is a unique path from source to destination (when m = n − 1)\nThere is at least one path from source to destination (when m > n − 1)\n'))
wall_remover += 1
walls = wall_list(maze_rows,maze_cols)

draw_maze(walls,maze_rows,maze_cols,cell_nums=True)

disjoint_set_forest = DisjointSetForest(maze_rows*maze_cols)

print('Maze using union')
timer_0 = time.time()
while wall_remover > 0:                                                                # will create the maze using stander union of sets
    choice = random.choice(walls) # selects random wall from list
    index = walls.index(choice)  # return index of wall
    if find(disjoint_set_forest,choice[0]) != find(disjoint_set_forest,choice[1]):
        walls.pop(index) # deletes the wall selected
        union(disjoint_set_forest,choice[0],choice[1]) # add the wall to a set using union
        #print('V1=',choice[0],'V2=',choice[1])
        add_edge(graph,choice[0],choice[1])
    wall_remover -= 1 # decreases the number of sets by one
timer_1 = time.time()
print('Total time ', timer_1-timer_0)
draw_maze(walls,maze_rows,maze_cols)


print(graph.graph)

print('Breath-First Search')
timer_0 = time.time()
BFS(graph,0)
timer_1 = time.time()
print('Total time ', timer_1-timer_0)

print('\nDepth-First Search')
timer_0 = time.time()
DFS(graph,0)
timer_1 = time.time()
print('Total time ', timer_1-timer_0)

print('\nDepth-First Search Recursion')
timer_0 = time.time()
print(DFSR(graph,0))
timer_1 = time.time()
print('Total time ', timer_1-timer_0)

plt.show()




