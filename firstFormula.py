import numpy as np

j=[0,1]
theta=[0,1]

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



def getReward(i,a):
    resulst=0
    if(a==0):
        p=p0;r=r0;w=w0
    if(a==1):
        p=p1;r=r1;w=w1
    if(a==2):
        p=p2;r=r2;w=w2
        
    for ele in j:
        for ele2 in theta:
            resulst+=(p[i,ele])*(r[ele,ele2])*(w[i])
        
    return resulst    
    
print(getReward(1,2))


