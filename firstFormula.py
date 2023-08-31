import numpy as np

j=[0,1]
theta=[0,1]
alphaVectors=[0]

#Transition Probabilities
p0=np.eye(2)
p1=np.full((2,2), 0.5)
p2=np.full((2,2), 0.5)


#Observation Probabilities
r0=np.array([[0.85, 0.15], [0.15, 0.85]])
r1=np.full((2,2), 0.5)
r2=np.full((2,2), 0.5)


#Immediate Rewards
w0=[-1,-1]
w1=[-100,10]
w2=[10,-100]



# Pi(belief space)
piList=np.linspace(0,1,101)



def getReward(i,a):
    result=0
    
    if(a==0):
        p=p0;r=r0;w=w0
    if(a==1):
        p=p1;r=r1;w=w1
    if(a==2):
        p=p2;r=r2;w=w2
        
#double summation        
    for ele in j:
        for ele2 in theta:
            result+=(p[i,ele])*(r[ele,ele2])*(w[i])
        
    return result    






def itrateFun(pi,a,theta,t):  
    
    summation=0
    argmaxArray=[]
            
    if(a==0):
        p=p0;r=r0
    if(a==1):
        p=p1;r=r1
    if(a==2):
        p=p2;r=r2
        
    
    for k in range(len(alphaVectors[t])):     
        if (t==1):
            if(pi[0]>=0.6):
                k=2
#             elif(pi[0]<=0.9 and pi[0]>=0.1):
#                 k=0
#             else:
#                 k=1
        for j in range(2):
            for i in range(2):

                summation+=pi[i]*p[i,j]*r[j,theta]*alphaVectors[t][k][j]
        argmaxArray.append(summation)
    
    maxK=max(argmaxArray)    

    

    
    
    return argmaxArray.index(maxK)
     


    
def generalSum(j,theta,t,a,i,pi):
    
        
    if(a==0):
            p=p0;r=r0
    if(a==1):
            p=p1;r=r1
    if(a==2):
            p=p2;r=r2

    result=0
    if(t!=1):
        for ele in j:
            for ele2 in theta:
                result+=(p[i,ele])*(r[ele,ele2])*alphaVectors[t-1][itrateFun(pi,a,ele2,t-1)][ele]
    return result        




actionsList=[]
for a in range(3):
    statesList=[]
    for i in range(2):
        train=getReward(i,a)+generalSum(j,theta,1,a,i,[1,0])        
        statesList.append(train)
    actionsList.append(statesList)
    
alphaVectors.append(actionsList)


# Belief_States_t1=[piList[10:91],piList[0:11],piList[90:101]]




actionsList=[]
beliefStates=[]
for a in range(3):
    for beliefPoint in piList:
        statesList=[]
        for i in range(2):
            train=getReward(i,a)+generalSum(j,theta,2,a,i,[beliefPoint,1-beliefPoint]) 
            statesList.append(train)
        if(not statesList in actionsList):
            actionsList.append(statesList)
            beliefStates.append(beliefPoint)
#             print(a)
alphaVectors.append(actionsList)







for eles in alphaVectors:
    print(f"T{alphaVectors.index(eles)}:",eles)
# print(beliefStates)



