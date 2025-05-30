from hmmlearn import hmm

# The HMM class will be used for sequence classification of accelerometer data using given training and testing data

class HMM:
    def __init__(self, 
                 trainFeatures,
                 trainLabels,
                 testFeatures,
                 randomState,
                 classProb,
                 hiddenStates = 4, 
                 numItt = 150):

        self.trainFeatures = trainFeatures
        self.trainLabels = trainLabels
        self.testFeatures = testFeatures
        self.randomState = randomState
        self.classProb = classProb
        self.hiddenStates = hiddenStates
        self.numItt = numItt

    def train(self, features):
        """
        Description:  Helper function to train a HMM function
           
        Inputs:
        features - training data used for the HMM model (numpy array)
        randomState - the random state to keep same initialization of variables within the model initialization
        hiddenStates - number of hidden states used for the HMM model (default of 4)
        numItt - number of iterations used for training the HMM model (default of 100)

        Outputs:
        model - trained HMM model (hmm object)
        """
        model = hmm.GaussianHMM(n_components=self.hiddenStates, covariance_type='full', n_iter=self.numItt, random_state = self.randomState)
        model.fit(features) 
        return model

    @staticmethod
    def logLikelihood(model, observation, prior):
        """
        Description: Helper function to calculate the Log Likelihood of a given observation for a HMM model

        Inputs:
            model - hmm model object
            observation - the observation at a given instance
            prior - prior distribution of the class

        Outputs:
            log_likelihood - calculated log likelihood 
        """
        log_likelihood = model.score(observation.reshape(1, -1)) + prior
        return log_likelihood

    def test(self):
        """
        Description: This model will calculated the predicted classes for the test data. This is done by training a HMM model for each class given the training 
        data. Then using the training data, we will input the observation at each time instance to each HMM model. We assign the predicted class based on the            HMM model with the highest likelihood

        Output:
        predLabels - the predicted classes calculated
        """

        # Strafity training data per class
        featStand = self.trainFeatures[self.trainLabels == 0].values
        featWalk = self.trainFeatures[self.trainLabels == 1].values
        featRun = self.trainFeatures[self.trainLabels == 2].values
        featTest = self.testFeatures.values

        # Train HMM model for each class
        modelStand = self.train(featStand)
        modelWalk = self.train(featWalk)
        modelRun = self.train(featRun)

        # Initalize List for predicted labels
        predLabels = []

        # Iterate for each observation
        for i in range(len(featTest)):
            obs = featTest[i]

            # Calculate log likelihoods
            ll_stand = self.logLikelihood(modelStand, obs, self.classProb[0])
            ll_walk = self.logLikelihood(modelWalk, obs, self.classProb[1])
            ll_run = self.logLikelihood(modelRun, obs, self.classProb[2])
            
            # Choose the activity with the highest likelihood
            likelihoods = {
                0: ll_stand,
                1: ll_walk,
                2: ll_run
            }

            # Take maximum of the likelihoods and append to list
            predLabels.append(max(likelihoods, key=likelihoods.get))
    
        return predLabels