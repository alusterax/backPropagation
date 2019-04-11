#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

qtInter,qtOut,qtInput,e,alpha,epoca = 100,10,784,0,0.1,0
dataset = pd.read_csv('mnist_train.csv')
v = np.random.randn(qtInput, qtInter) / np.sqrt(qtInput)
w = np.random.randn(qtInter, qtOut) / np.sqrt(qtInter)
ativacao = 'sig'
limiteLinhas = 60001
#Recomendado colocar 10000 pois o algoritmo está lento, e pode demorar para testar
#Leva em media 10-15 minutos para rodar
#Com 10000 linhas ele pega 87-89% de precisão
#Com todas 60000 chega proximo a 94%


# In[ ]:


grafico = np.array([[0]*2])
graficoTreino = np.array([[0]*2])


# In[ ]:


def export_weights(filenameV,filenameW):
    if (not os.path.exists(fV) and not os.path.exists(fW)):
        exW = pd.DataFrame(w)
        exV = pd.DataFrame(v)
        exW.to_csv('WCSV.csv', index = False, header = False)
        exV.to_csv('VCSV.csv', index = False, header = False) 
        
def import_weights():
    global v, w
    recV = pd.read_csv('VCSV.csv',sep=',',header = None,dtype=float)
    recW = pd.read_csv('WCSV.csv',sep=',',header = None,dtype=float)
    v = recV.values
    w = recW.values


# In[ ]:


#export_weights('VtesteCSV.csv','WtesteCSV.csv')
#import_weights()


# In[ ]:


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

def funcAtivacao(x,func) :
    if (func == 'relu') : 
        return np.maximum(x,0) 
    elif (func == 'sig') : 
        return (1/(1+np.exp(-x)))
    elif (func == 'tanh'):
        return np.tanh(x)

def derivada(x,func):        
    if (func == 'relu') : 
        return (0 if x<0 else 1)
    elif (func == 'sig') : 
        return (x * (1 - x))
    elif (func == 'tanh') :
        return (1 - np.power(x,2))

def verificaAcerto(tar,out):
    indTar = np.unravel_index(np.argmax(tar, axis=None), tar.shape)
    indOut = np.unravel_index(np.argmax(out, axis=None), out.shape)
    if (indTar == indOut):
        return True
    else :
        return False

def deltaK(targetK,Yk,func) :
    dK = np.zeros(qtOut)
    for i in range (qtOut) :
        erro = targetK[i] - Yk[i]
        deriv = derivada(Yk[i],func)
        dK[i] = erro * deriv
    return dK

def deltaJ(inJ,inZ,func):
    delJ = np.zeros(qtInter)
    for j in range (qtInter):
        deriv = derivada(inZ[j],func)
        delJ[j] = inJ[j] * deriv
    return delJ

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


# In[ ]:


def algoritmo(isTreino,qtEpoca):
    global epoca
    for i in range (qtEpoca):
        readLine(dataset,isTreino)
        epoca+=1


# In[ ]:


def readLine(dSet,isTreino):
    global e
    r,e = 1,0
    for row in dSet.itertuples(index=False):
        if (r<limiteLinhas):
            linha = list(row)
            target = targetVetor(linha.pop(0))
            linha = trataLinha(linha)
            forward(linha,isTreino,target,r)
            r+=1


# In[ ]:


def forward(linha,isTreino,target,r):
    
        global e
        global grafico
        global graficoTreino
        inZ = inputZ(linha)
        Z = funcAtivacao(inZ,ativacao)
        inY = inputY(Z,w)
        Y = funcAtivacao(inY,ativacao)
        if (not verificaAcerto(target,Y)):
            e+=1

        a = 100 - ((e/r)*100)
        if (r!=0 and r%100 == 0):
            #print (target)
            #print (Y)
            print (f'Linha: {r} Epoca: {epoca} Erros: {e}  Acerto: {a} %' + '\n' + '_________' )
        if isTreino:
            backPropagation(inZ,Z,inY,Y,target,linha)
            graficoTreino = np.append(graficoTreino,[[r,a]],axis=0)
        if (not isTreino) :
            grafico = np.append(grafico,[[r,a]],axis=0)


# In[ ]:


def backPropagation(inZ,Z,inY,Y,target,linha):
    delK,inJ,delJ =[],[],[]
    delK = deltaK(target,Y,ativacao) 
    inJ  = deltainJ(delK,w)
    delJ = deltaJ(inJ,Z,ativacao)
    atualizaPesos(delK,Z,linha,delJ)


# In[ ]:


def atualizaPesos(delK,Z,linha,delJ):
    global w
    global v
    correcaoW = deltaW(delK,Z)
    correcaoV = deltaV(linha,delJ)
    w = corrigePeso(w,correcaoW)
    v = corrigePeso(v,correcaoV)


# In[ ]:


def corrigePeso(peso,delta):
    return np.add(peso,delta)


# In[ ]:


def testar():
    global dataset
    dataset = pd.read_csv('mnist_test.csv')
    algoritmo(False,1)
    plt.plot(grafico[:,0],grafico[:,1])
    plt.xlabel('Linhas')
    plt.ylabel('Porcentagem de Acerto')
    plt.title('Teste da rede | Sigmoid *Teste*')
    plt.show()


# In[ ]:


def treinar():
    algoritmo(True,1)
    plt.plot(graficoTreino[:,0],graficoTreino[:,1])
    plt.xlabel('Linhas')
    plt.ylabel('Porcentagem de Acerto')
    plt.title('Aprendizado da rede | Sigmoid *Treino* ')
    plt.show()


# In[ ]:


treinar()


# In[ ]:


testar()

