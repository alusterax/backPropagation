import numpy as np
import pandas as pd

qtInter,qtOut,qtInput,e,alpha,epoca = 100,10,784,0,0.1,0
dataset = pd.read_csv('mnist_test.csv')
v = np.random.randn(qtInput, qtInter) / np.sqrt(qtInput)
w = np.random.randn(qtInter, qtOut) / np.sqrt(qtInter)

def algoritmo(isTreino,qtEpoca):
    global epoca
    for i in range (qtEpoca):
        readLine(dataset,isTreino)
        epoca+=1
        
def readLine(dSet,isTreino):
    global e
    r,e = 1,0
    for row in dSet.itertuples(index=False):
        linha = list(row)
        target = targetVetor(linha.pop(0))
        linha = trataLinha(linha)
        forward(linha,isTreino,target,r)
        r+=1
        
def forward(linha,isTreino,target,r):
    global e
    inZ = inputZ(linha)
    Z = funcAtivacao(inZ,'relu')
    inY = inputY(Z,w)
    Y = funcAtivacao(inY,'relu')
    print (target)
    print (Y)
    if isTreino:
        if (not verificaAcerto(target,Y)):
            e+=1
            backPropagation(inZ,Z,inY,Y,target,linha)
    print (f'Linha: {r} Epoca: {epoca} Erros: {e} ______________')
    
def backPropagation(inZ,Z,inY,Y,target,linha):
    delK,inJ,delJ =[],[],[]
    delK = deltaK(target,Y,inY,'relu') 
    inJ  = deltainJ(delK,w)
    delJ = deltaJ(inJ,inZ,'relu')
    atualizaPesos(delK,Z,linha,delJ)
    
def trataLinha(inp):
    retorno = np.array(inp)
    return retorno/255

def targetVetor(labels, num_classes=10):
    retTarget = np.eye(num_classes)[labels]
    return retTarget

def inputZ(inp):
    inX = np.zeros(qtInter)
    for j in range (qtInter):
        inX[j] = np.sum(inp * np.array(v[:,j]))
    return inX

def inputY(inp,pesos):
    inX = np.zeros(qtOut)
    for k in range (qtOut):
        inX[k] = np.sum(inp * np.array(pesos[:,k]))
    return inX

def deltainJ (dK,peso):
    inJ = np.zeros(qtInter)
    for j in range (qtInter):
        for k in range (qtOut):
            inJ[j] += dK[k] * peso[j][k]
    return inJ

def corrigePeso(peso,delta):
    return np.add(peso,delta)

def funcAtivacao(x,func) :
    if (func == 'relu') : 
        return np.maximum(x,0) 
    elif (func == 'sig') : 
        return (1/(1+np.exp(-x)))

def derivada(x,func):        
    if (func == 'relu') : 
        return derivRelu(x) 
    elif (func == 'sig') : 
        return (x * (1 - x))

def verificaAcerto(tar,out):
    for i in range (qtOut):
        if (tar[i] != out[i]):
            return False
    return True

def deltaK(targetK,Yk,YinK,func) :
    dK = np.zeros(qtOut)
    for i in range (qtOut) :
        erro = targetK[i] - Yk[i]
        deriv = derivada(YinK[i],func)
        dK[i] = erro * deriv
    return dK

def deltaJ(inJ,inZ,func):
    delJ = np.zeros(qtInter)
    for j in range (qtInter):
        deriv = derivada(inZ[j],func)
        delJ[j] = inJ[j] * deriv
    return delJ

def atualizaPesos(delK,Z,linha,delJ):
    global w
    global v
    correcaoW = deltaW(delK,Z)
    correcaoV = deltaV(linha,delJ)
    w = corrigePeso(w,correcaoW)
    v = corrigePeso(v,correcaoV)

def deltaW(dK,Ze):
    dW = np.empty_like(w)
    for j in range(qtInter):
        for k in range(qtOut):
            dW[j][k] = alpha * dK[k] * Ze[j]
    return dW

def deltaV(linha,dJ):
    dV = np.empty_like(v)
    for i in range(qtInput):
        for j in range (qtInter):
            dV[i][j] = alpha * dJ[j] * linha[i]
    return dV
    
def derivRelu(x):
    return 0 if x<0 else 1

algoritmo(True,2)
