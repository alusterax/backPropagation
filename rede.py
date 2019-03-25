import math
import numpy as np
#TO-DO: 
# Criar matriz input e outputs (e fazer abrir do csv)
# Completar função de propagação (de preferencia que funciona de X para Z e Z para Y)
# Criar função para atribuir pesos aleatórios (de -1 a 1)
# Definir quantos neuronios na camada intermediária (Max sugeriu começar com 100)
#___________

#matriz Input [728] = Recebe csv;
#matriz Output[10 ] = Recebe csv;

#Legendas: 
# X - Input, Z - Intermediária, Y - Output, target - Valor alvo de saída desejado, e que é fornecido 
# velAprend - Velocidade de aprendizado, erro - diferença entre output e target
# deltaK - Fator de distrib. de erro Z <- Y (backpropagation,ajuste de peso)
# deltaJ - Fator de distrib. de erro X <- Z     - || -

X = np.array([ [-1],[-1],[-1],[-1] ])
V = np.array([ [0.3, -0.4],
               [0.1,  0.6],
               [-0.2, 0.1],
               [0,   -0.5] ])
Z = np.array([[0],[0]])   #vetor 100            
def propag() :
    for i in range(3):
        for j in range (1):
           Z[j] +=   X[i] * V[i][j]
    #foreach in Input xi, Pesos vij, Xi*Vij;
def funcAtivacao(x) :
    return 1/(1+math.exp(-x))
def deltaK(targetK,Yk,YinK) :
    erro = targetK - Yk
    deriv = YinK * (1-YinK)
    return erro*deriv
def deltaWjk(velAprend,delK,Zj) :
    return velAprend * delK * Zj


print (deltaK(-1,0.459,-0.164))
print (deltaK(-1,0.629,-0.529))
print (deltaWjk(0.2,-0.362,0.59))
print (X)

propag()