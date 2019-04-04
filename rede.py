import math
import numpy as np
import pandas as pd

qtInter = 100
v = np.random.uniform(-1,1,(784,qtInter))
w = np.random.uniform(-1,1,(qtInter,10))
df = pd.read_csv('csvTest.csv')
delK = np.zeros(10)
inDel = np.zeros(100)
delX = np.zeros(100)
target = np.zeros(10)

def trataTarget(t) :
    retTarget = np.zeros(10)
    if   (t == 0) : retTarget = [1,0,0,0,0,0,0,0,0,0]
    elif (t == 1) : retTarget = [0,1,0,0,0,0,0,0,0,0]
    elif (t == 2) : retTarget = [0,0,1,0,0,0,0,0,0,0]
    elif (t == 3) : retTarget = [0,0,0,1,0,0,0,0,0,0]
    elif (t == 4) : retTarget = [0,0,0,0,1,0,0,0,0,0]
    elif (t == 5) : retTarget = [0,0,0,0,0,1,0,0,0,0]
    elif (t == 6) : retTarget = [0,0,0,0,0,0,1,0,0,0]
    elif (t == 7) : retTarget = [0,0,0,0,0,0,0,1,0,0]
    elif (t == 8) : retTarget = [0,0,0,0,0,0,0,0,1,0]
    elif (t == 9) : retTarget = [0,0,0,0,0,0,0,0,0,1]
    return retTarget

def propag (inp, pesos) :
    soma =0
    qtInput = len(inp)
    inX = np.zeros(np.size(pesos,axis=1))
    for j in range (np.size(pesos,axis=1)):
        for i in range (qtInput):
            soma +=(inp[i]/2550) * pesos[i][j]
        inX[j] = round(soma,4)
        soma =0
    return inX
    
def funcAtivacao(x) :
    Ze = np.zeros(len(x))
    for i in range(len(x)):
        Ze[i] = round ( (2/1+ np.exp(-x[i])) -1 ,4)
    return Ze
  
def deltaK(targetK,Yk,YinK) :
        erro = targetK - Yk
        deriv = YinK * (1 - YinK)
        return round(erro * deriv,4)
     
def deltaW(alpha,dK,Ze):
    delW = np.zeros((100,10))
    for j in range(10):
        for i in range(qtInter):
            delW[i][j] = round (alpha * delK[j] * Z[i],4)
    return delW

#separa csv em dois arrays, a = input, target = target.
for row in df.itertuples(index=False):
    a = list(row)
    tar = a.pop(0)

target = trataTarget(tar)    
    
#forward
inZ = propag(a,v)

Z = funcAtivacao(inZ)

inY = propag(Z,w)

Y = funcAtivacao(inY)

#calculo de erro
for i in range (10) :
    delK[i] = deltaK(target[i],Y[i],inY[i])
    
#backpropagation    
deltaW(0.2,delK,Z)

#continua
w = deltaW(0.2,delK,Z)
