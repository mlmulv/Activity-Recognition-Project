from hmmlearn import hmm

class HMM:
    def __init__(self, 
                 trainFeatures,
                 trainLabels,
                 testFeatures,
                 hiddenStates = 4, 
                 numItt = 100):

        self.trainFeatures = trainFeatures
        self.trainLabels = trainLabels
        self.testFeatures = testFeatures
        self.hiddenStates = hiddenStates
        self.numItt = 100

    def train(self, features):
        """
        Inputs:
            features - training data used for the HMM model (numpy array)
        
        """
    
        model = hmm.GaussianHMM(n_components=self.hiddenStates, covariance_type='full', n_iter=self.numItt)
        model.fit(features)

        return model

    @staticmethod
    def logLikelihood(model, observation):
        log_likelihood = model.score(observation.reshape(1, -1))
        return log_likelihood

    def test(self):
        featStand = self.trainFeatures[self.trainLabels == 0].values
        featWalk = self.trainFeatures[self.trainLabels == 1].values
        featRun = self.trainFeatures[self.trainLabels == 2].values
        featTest = self.testFeatures.values
        
        modelStand = self.train(featStand)
        modelWalk = self.train(featWalk)
        modelRun = self.train(featRun)
    
        predLabels = []
        
        for i in range(len(featTest)):
            obs = featTest[i]
            ll_stand = self.logLikelihood(modelStand, obs)
            ll_walk = self.logLikelihood(modelWalk, obs)
            ll_run = self.logLikelihood(modelRun, obs)
            
            # Choose the activity with the highest likelihood
            likelihoods = {
                0: ll_stand,
                1: ll_walk,
                2: ll_run
            }
            
            predLabels.append(max(likelihoods, key=likelihoods.get))
    
        return predLabels