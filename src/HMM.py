from hmmlearn import hmm

def train(features, hiddenStates = 6, numItt = 500):
    """
    Inputs:
        features - training data used for the HMM model (numpy array)
        
    """

    model = hmm.GaussianHMM(n_components=hiddenStates, covariance_type='full', n_iter=numItt)
    model.fit(features)
    
    return model

def logLikelihood(model, observation):
    # Wrap observation in sequence format
    log_likelihood = model.score(observation.reshape(1, -1))
    return log_likelihood

def test(featStand, featWalk, featRun, featTest):

    modelStand = train(featStand)
    modelWalk = train(featWalk)
    modelRun = train(featRun)

    predLabels = []
    
    for i in range(len(featTest)):
        obs = featTest[i]
        ll_stand = logLikelihood(modelStand, obs)
        ll_walk = logLikelihood(modelWalk, obs)
        ll_run = logLikelihood(modelRun, obs)
        
        # Choose the activity with the highest likelihood
        likelihoods = {
            0: ll_stand,
            1: ll_walk,
            2: ll_run
        }
        
        predLabels.append(max(likelihoods, key=likelihoods.get))

    return predLabels