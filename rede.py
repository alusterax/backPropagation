import math
import numpy as np
import pandas as pd

qtInter = 100
v = np.random.uniform(-1,1,(784,qtInter))
w = np.random.uniform(-1,1,(qtInter,10))
df = pd.read_csv('csvTest.csv')
delK = np.zeros(10)
delW = np.zeros((100,10))
inDel = np.zeros(100)
DeltaDelta = np.zeros(100)

def propag (inp, pesos) :
    soma =0
    qtInput = len(inp)
    inX = np.zeros(np.size(pesos,axis=1))
    for j in range (np.size(pesos,axis=1)):
        for i in range (qtInput):
            soma +=(inp[i]/2550) * v[i][j]
        inX[j] = soma
        soma =0
    return inX
    
def funcAtivacao(x) :
    Ze = np.zeros(len(x))
    for i in range(len(x)):
        Ze[i] = (2/1+ np.exp(-x[i])) -1
    return Ze
  
def deltaK(targetK,Yk,YinK) :
        erro = targetK - Yk
        deriv = YinK * (1 - YinK)
        return erro * deriv
     
def deltaW(alpha,dK,Ze):
    for j in range(10):
        for i in range(qtInter):
            delW[i][j] = alpha * delK[j] * Z[i]

#separa csv em dois arrays, a = input, target = target.
for row in df.itertuples(index=False):
    a = list(row)
    target = a.pop(0)

#forward
inZ = propag(a,v)

Z = funcAtivacao(inZ)

inY = propag(Z,w)

Y = funcAtivacao(inY)

#calculo de erro
for i in range (10) :
    delK[i] = deltaK(1,Y[i],inY[i])
#backpropagation    
deltaW(0.2,delK,Z)

#continua
