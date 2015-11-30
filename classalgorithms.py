from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
import random
import string

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = 0.01
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes; need to complete the inherited learn and predict functions """
   
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
            
class GaussianNB(Classifier):
    #naive Bayes, assuming a Gaussian distribution on each of the features
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
    
    def divide(self,ds):
	dividedDS = [(utils.mean(x), utils.stdev(x)) for x in zip(*ds)]
	del dividedDS[-1]
	return dividedDS
        
    def learn(self, Xtrain, ytrain):
        temp = {}
        dividedDS = {}
        for i in range(Xtrain.shape[0]):
            trial = Xtrain[i]
            if (ytrain[i] not in temp):
                temp[ytrain[i]]=[]
            temp[ytrain[i]].append(trial)  
        for val, i in temp.items():
            dividedDS[val] = self.divide(i)
        return dividedDS
               
	
    def probbydivide(self,dividedDS, Xtest):
	prob = {}
	for val, var in dividedDS.items():
	    prob[val] = 1
	    for i in range(len(var)):
		mean, stdev = var[i]
		x = Xtest[i]
		probability=utils.calculateprob(x, mean, stdev)
		prob[val]=prob[val]* probability
	return prob
    
    def GetMaxLabel(self,dividedDS, Xtest):
	Label=None
	Prob =-1
	prob = self.probbydivide(dividedDS, Xtest)
	for val, var in prob.items():
	    if Label is None or var > Prob:
	        Prob = var
		Label = val
	return Label

    def Predict(self,dividedDS,Xtest):
	pred = []
	for i in range(len(Xtest)):
	    result = self.GetMaxLabel(dividedDS, Xtest[i])
	    pred.append(result)
	return pred
            
class modifiedLinear(Classifier):
    
    lamda=0.0
    alpha=0.0
    def newActivefun(self, number):
        squareterm=number**2
        squareroot=math.sqrt(1+squareterm)
        return (1/2)*(1+(number/squareroot))
        
    def __init__( self, params=None ):
        self.weights = None
        self.lamda=0.001
        self.alpha=0.01
        
    def sign(self, weights):
        signval=np.zeros(weights.shape[0])
        for i in range (weights.shape[0]):
            if weights[i]>0:
                signval[i]=1
            elif weights[i]<0:
                signval[i]=-1
            else:
                signval[i]=0
        return signval
        
    def temp(self,num):
        squareterm=num**2
        squareroot=math.sqrt(1+squareterm)
        return(1/squareroot)
        
    def learn(self, Xtrain, ytrain):
        p=range(Xtrain.shape[0])
        productterm=range(Xtrain.shape[0])
        RHS=range(Xtrain.shape[0])
        oldweights=np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        NewWeights=np.zeros(oldweights.shape[0])
        tolerance=0.1
        n=0
        while self.getTolerance(NewWeights, oldweights)>tolerance and n<20:
            n=n+1
            NewWeights=oldweights
            for i in range(ytrain.shape[0]):
                p[i]=self.newActivefun(np.dot(-oldweights.T,Xtrain[i]))
                productterm[i]=self.temp(np.dot(-oldweights.T,Xtrain[i]))
            #for k in range(ytrain.shape[0]):
            RHS=np.multiply(productterm,ytrain-p)  
            temp1=NewWeights
            signvals=self.sign(temp1)
            Gradient=np.dot(Xtrain.T, RHS)-self.lamda*signvals
            oldweights=NewWeights - self.alpha*Gradient
        self.weights=oldweights
             
    def getTolerance(self,NewWeights, oldweights):
        sum=0
        for i in range(NewWeights.shape[0]):
            sum = sum + (NewWeights[i] - oldweights[i])**2
        #print math.sqrt(sum)
        return math.sqrt(sum)

    def predict(self,Xtest):
        ytest= []
        for i in range(Xtest.shape[0]):
            threshold=self.newActivefun(np.dot(-self.weights.T,Xtest[i]))
            if threshold >= 0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        return ytest
    
    
class LogitReg(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """
    
    def sigmoid(self, number):
        return 1.0/(1.0+np.exp(number))

    def __init__( self, params=None ):
        self.weights = None
    
        
    def learn(self, Xtrain, ytrain):
        #print Xtrain.shape[1],ytrain.shape[0]
        p=range(Xtrain.shape[0])
        oldweights=np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        #print oldweights.shape
        NewWeights=np.zeros(oldweights.shape[0])
        tolerance=0.00001
        while self.getTolerance(NewWeights, oldweights)>tolerance:
            NewWeights=oldweights
            for i in range(ytrain.shape[0]):
                p[i]=self.sigmoid(np.dot(-oldweights.T,Xtrain[i]))
            P=np.diag(p);
            IdentityMat=np.identity(len(P))
            Hessian=np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,P),np.subtract(IdentityMat,P)), Xtrain))
            Gradient=np.dot(Xtrain.T, np.subtract(ytrain, p))
            oldweights=NewWeights + np.dot(Hessian,Gradient)
        self.weights=oldweights
        
            
    def getTolerance(self,NewWeights, oldweights):
        sum=0
        for i in range(NewWeights.shape[0]):
            sum = sum + (NewWeights[i] - oldweights[i])**2
        
        return math.sqrt(sum)

    def predict(self,Xtest):
        ytest= []
        for i in range(Xtest.shape[0]):
            threshold=self.sigmoid(np.dot(-self.weights.T,Xtest[i]))
            if threshold >= 0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        return ytest
   
    
class NeuralNet(Classifier):
    lcount=0
    shape=None
    weights=[]
    """ Two-layer neural network; need to complete the inherited learn and predict functions """
    
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid
        lsize=(self.ni,self.nh,self.no)
        self.lcount=len(lsize)-1
        self.layerIP=[]
        self.layerOP=[] 
        self.shape=lsize
        
        # Set step-size
        self.stepsize = 0.1

        # Number of repetitions over the dataset
        self.reps = 5

        for(i,j) in zip(lsize[:-1], lsize[1:]):
            self.weights.append(np.random.normal(scale=0.1,size=(j,i)))
        
            
    def activate(self,x,Derivative=False):
        if not Derivative:
            sigmoid= 1/(1+np.exp(-x))
            return sigmoid
        else:
            out=self.activate(x)
            dsigmoid= out*(1-out)
            return dsigmoid

    def learn(self, Xtrain, ytrain):
        """ Incrementally update neural network using stochastic gradient descent """        
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                self.update(Xtrain[samp,:],ytrain[samp])         
            
    def evaluate(self, inputs):
        """ Including this function to show how predictions are made """
        self.layerIP=[]
        self.layerOP=[] 
        
        for i in range(self.lcount):
            if i==0:
                LIP=np.dot(self.weights[0],inputs.T)
            else:
                LIP=np.dot(self.weights[i],self.layerOP[-1])
            
            self.layerIP.append(LIP)
            self.layerOP.append(self.activate(LIP))
        return self.layerOP[-1].T

    def update(self, inp, out):
        """ This function needs to be implemented """   
        delta=[]
        self.evaluate(inp) 
        prev_delta=[]
        
        for i in reversed(range(self.lcount)):
            if i==self.lcount-1:
                next_delta=self.layerOP[i]-out.T
                error=np.sum(next_delta**2)
                delta.append(next_delta*self.activate(self.layerIP[i],True))

            else:
                prev_delta=self.weights[i+1].T.dot(delta[-1])
                delta.append(prev_delta*self.activate(self.layerIP[i],True))
            
    
        for i in range(self.lcount):        
            if i==0:
                layerOutput=inp.T
            else:
                layerOutput=self.layerOP[i-1]             
            for i in range(len(self.weights)):
                layer = np.atleast_2d(layerOutput[i])
                deltas = np.atleast_2d(delta[i])
                self.weights[i] += self.stepsize * np.dot(layer.T,deltas)
        return error
        
    def predict(self,Xtest):
        ytest=[]
        n = len(Xtest)
        ret=np.ones((n,1))
        for p in range (Xtest.shape[0]):
            ret[p,:]=self.evaluate(Xtest[p,:])
            threshold=ret[p,:]
            if threshold >= 0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        #print ret
        return ytest
    
        
class LogitRegL2(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """
    lamda=0.0
    
    def sigmoid(self, number):
        return 1.0/(1.0+np.exp(number))

    def __init__( self, params=None ):
        self.weights = None
        self.lamda=0.00001
        #self.alpha=0.01
                
    def learn(self, Xtrain, ytrain):
        #print('\n R : {0}').format(self.lamda)
        p=range(Xtrain.shape[0])
        oldweights=np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        NewWeights=np.zeros(oldweights.shape[0])
        tolerance=0.01
        n=0
        while self.getTolerance(NewWeights, oldweights)>tolerance:
            n+=1
            #print('\n Iter : {0}').format(iter)
            NewWeights=oldweights
            for i in range(ytrain.shape[0]):
                p[i]=self.sigmoid(np.dot(-oldweights.T,Xtrain[i]))
            P=np.diag(p);
            IdentityMat=np.identity(len(P))
            Hessian=np.linalg.pinv(np.dot(np.dot(np.dot(Xtrain.T,P),np.subtract(IdentityMat,P)), Xtrain))
            Gradient=np.dot(Xtrain.T, np.subtract(ytrain, p))-2*self.lamda*NewWeights
            oldweights=NewWeights+np.dot(Hessian,Gradient)
        self.weights=oldweights
        
            
    def getTolerance(self,NewWeights, oldweights):
        sum=0
        for i in range(NewWeights.shape[0]):
            sum = sum + (NewWeights[i] - oldweights[i])**2
        #print math.sqrt(sum)
        return math.sqrt(sum)

    def predict(self,Xtest):
        ytest= []
        for i in range(Xtest.shape[0]):
            threshold=self.sigmoid(np.dot(-self.weights.T,Xtest[i]))
            if threshold >= 0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        return ytest
   
class LogitRegL1(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """
    lamda=0.0001
    
    def sigmoid(self, number):
        return 1.0/(1.0+np.exp(number))

    def __init__( self, params=None ):
        self.weights = None
        self.lamda=0.01
        #self.alpha=0.001
        
    def sign(self, weights):
        signval=np.zeros(weights.shape[0])
        for i in range (weights.shape[0]):
            if weights[i]>0:
                signval[i]=1
            elif weights[i]<0:
                signval[i]=-1
            else:
                signval[i]=0
        return signval
            
            
    def learn(self, Xtrain, ytrain):
        p=range(Xtrain.shape[0])
        oldweights=np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        NewWeights=np.zeros(oldweights.shape[0])
        tolerance=0.01
        n=0
        while self.getTolerance(NewWeights, oldweights)>tolerance and n<20:
            n=n+1
            NewWeights=oldweights
            for i in range(ytrain.shape[0]):
                p[i]=self.sigmoid(np.dot(-oldweights.T,Xtrain[i]))
            P=np.diag(p);
            IdentityMat=np.identity(len(P))
            temp1=NewWeights
            signvals=self.sign(temp1)
            Hessian=np.linalg.pinv(np.dot(np.dot(np.dot(Xtrain.T,P),np.subtract(IdentityMat,P)), Xtrain))
            Gradient=np.dot(Xtrain.T, np.subtract(ytrain, p))-self.lamda*signvals
            #print np.dot(Hessian,Gradient).shape
            oldweights=NewWeights+np.dot(Hessian,Gradient)
        self.weights=oldweights
        
            
    def getTolerance(self,NewWeights, oldweights):
        sum=0
        for i in range(NewWeights.shape[0]):
            sum = sum + (NewWeights[i] - oldweights[i])**2
        
        return math.sqrt(sum)

    def predict(self,Xtest):
        ytest= []
        for i in range(Xtest.shape[0]):
            threshold=self.sigmoid(np.dot(-self.weights.T,Xtest[i]))
            if threshold >= 0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        return ytest

class ElasticNet(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """
    lamda1=0.00001
    lamda2=0.01
    
    def sigmoid(self, number):
        return 1.0/(1.0+np.exp(number))

    def __init__( self, params=None ):
        self.weights = None
        self.lamda1=.01
        self.lamda2=.00001
        
    def sign(self, weights):
        signval=np.zeros(weights.shape[0])
        for i in range (weights.shape[0]):
            if weights[i]>0:
                signval[i]=1
            elif weights[i]<0:
                signval[i]=-1
            else:
                signval[i]=0
        return signval
            
            
    def learn(self, Xtrain, ytrain):
        p=range(Xtrain.shape[0])
        oldweights=np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        NewWeights=np.zeros(oldweights.shape[0])
        tolerance=0.1
        n=0
        while self.getTolerance(NewWeights, oldweights)>tolerance and n<100:
            n=n+1
            NewWeights=oldweights
            for i in range(ytrain.shape[0]):
                p[i]=self.sigmoid(np.dot(-oldweights.T,Xtrain[i]))
            P=np.diag(p);
            IdentityMat=np.identity(len(P))
            Hessian=np.linalg.pinv(np.dot(np.dot(np.dot(Xtrain.T,P),np.subtract(IdentityMat,P)), Xtrain))
            temp1=NewWeights
            signvals=self.sign(temp1)
            Gradient=np.dot(Xtrain.T, np.subtract(ytrain, p))-self.lamda1*signvals-2*self.lamda2*NewWeights
            oldweights=NewWeights+ np.dot(Hessian,Gradient)
        self.weights=oldweights
        
            
    def getTolerance(self,NewWeights, oldweights):
        sum=0
        for i in range(NewWeights.shape[0]):
            sum = sum + (NewWeights[i] - oldweights[i])**2
        #print math.sqrt(sum)
        return math.sqrt(sum)

    def predict(self,Xtest):
        ytest= []
        for i in range(Xtest.shape[0]):
            threshold=self.sigmoid(np.dot(-self.weights.T,Xtest[i]))
            if threshold >= 0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        return ytest
   