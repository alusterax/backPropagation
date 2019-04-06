import math
import numpy as np
import pandas as pd

qtInter = 100
qtOut = 10
alpha = 0.5
v = np.round(np.random.uniform(-1,1,(784,qtInter)),4)
w = np.round(np.random.uniform(-1,1,(qtInter,10)),4)
df = pd.read_csv('mnist_test.csv')
delK = np.zeros(qtOut)
inDel = np.zeros(qtInter)
delX = np.zeros(qtInter)
target = np.zeros(qtOut)

epoca = 0
erros=[]

def trataTarget(t) :
    retTarget = np.zeros(qtOut)
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

def forward (inp, pesos) :
    soma =0
    qtInput = len(inp)
    inX = np.zeros(np.size(pesos,axis=1))
    for j in range (np.size(pesos,axis=1)):
        for i in range (qtInput):
            soma +=(inp[i]) * pesos[i][j]
        inX[j] = soma
        soma =0
    return inX

def hiperbolica(x):
    hip = np.zeros(len(x))
    for i in range(len(x)):
        hip[i] = ((np.exp(x[i]) - np.exp(-x[i])) / (np.exp(x[i]) + np.exp(-x[i])))
    return hip

def sigmoide(x):
    sig = np.zeros(len(x))
    for i in range(len(x)):
        sig[i] = (1/1+ np.exp(-x[i]))
    return sig

def softmax(x):
    smax = np.exp(x - np.max(x))
    return smax / smax.sum()

def funcAtivacao(x,func) :
    if (func == 'relu') : 
        return np.maximum(x,0) 
    elif (func == 'sig') : 
        return sigmoide(x) 
    elif (func == 'hip') : 
        return hiperbolica(x)
    elif (func == 'softmax'):
        return softmax(x)
    else: 
        print('parametro invalido!')

def deltaK(targetK,Yk,YinK) :
    dK = np.zeros(qtOut)
    for i in range (qtOut) :
        erro = targetK[i] - Yk[i]
        deriv = YinK[i] * (1 - YinK[i])
        dK[i] = erro * deriv
    return dK

def deltaW(dK,Ze):
    delW = np.zeros((qtInter,qtOut))
    for j in range(qtInter):
        for k in range(qtOut):
            delW[j][k] = alpha * delK[k] * Ze[j]
    return delW

def deltainJ (dK,peso):
    inJ = np.zeros(qtInter)
    for k in range (qtOut) :
        for j in range (qtInter):
            inJ[j] += delK[k] * w[j][k]
    return inJ

def deltaJ(inJ,inZ):
    delJ = np.zeros(qtInter)
    for j in range (qtInter):
        deriv = inZ[j] * (1 - inZ[j])
        delJ[j] = inJ[j] * deriv
    return delJ

def deltaV(linha,delJ):
    qtInput = len(linha)
    delV = np.zeros((qtInput,qtInter))
    for i in range(qtInput):
        for j in range (qtInter):
            delV[i][j] = alpha * delJ[j] * linha[i]
    return delV

def trataInput(inp,limiar):
    for i in range(len(inp)) :
        if (inp[i]>limiar):
            inp[i] = 1
        else: 
            inp[i] = 0
    return inp

def trata255(inp):
    for i in range(len(inp)):
        if (inp[i]>0):
            inp[i] = round(inp[i]/255,5)
        else:
            inp[i] = inp[i]
    return inp

def validaSaida(tar,out):
    for i in range (qtOut):
        if (tar[i] != out[i]):
            return False
    return True

def backprop (treino,ep):
    global epoca
    global erros

    for e in range (ep):
        global err
        global v
        global w
        err = 0
        i = 0
        for row in df.itertuples(index=False):
            linha = list(row) #linha = trataInput(linha) #linha = trata255(linha)
            tar = linha.pop(0)
            linha = trata255(linha)
            target = trataTarget(tar)
            inZ = forward(linha,v)
            Z = funcAtivacao(inZ,'relu')
            inY = forward(Z,w)
            Y = funcAtivacao(inY,'hip')
            i+=1
            if treino:
                acertou = validaSaida(target,Y)
                if (not acertou):
                    err+=1
                    delK = deltaK(target,Y,inY)
                    inJ = deltainJ(delK,w)
                    delJ = deltaJ(inJ,inZ)
                    w = deltaW(delK,Z)
                    v = deltaV(linha,delJ)
            print (f'ja foram {i} linhas da epoca {epoca}')
            #print (f'V: {v}')
            #print (f'W: {w}')
            print(f'Y:{Y}')
            print (f'erros: {err}')
            #break
        erros.append(err)
        epoca +=1
        
 backprop(True,1)
