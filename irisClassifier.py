import numpy                as  np
import pandas               as  pd
import matplotlib.pyplot    as  plt
import seaborn              as  sn
import os

# This class contains information about each sample. A feature vector, the actual class and it's vector
class SampleVector():

    # Runs when a SampleVector() object is made
    def __init__( self, features, classVector,  classLabel ):

        # Initialize values
        self.featuresVector     =   features
        self.classVector        =   classVector
        self.className          =   classLabel


        # Make a dictionary mapping the index of features to the feature name
        self.featureNames   = \
            {
                "Sepal length"  :   0,
                "Sepal width"   :   1,
                "Petal length"  :   2,
                "Petal width"   :   3
            }

    # Function that removes a specified feature from the sample
    def removeFeature( self, feature ):

        # Index of the removed feature
        removedIndex    =   self.featureNames[ feature ]

        # Make a new empty list to be filled with all features except the one to be deleted
        newFeaturesVector   =   []


        if removedIndex == len( self.featureNames ) - 1:
            newFeaturesVector   =   self.featuresVector[ : removedIndex ]
        else:
            newFeaturesVector.extend( self.featuresVector[ : removedIndex ] )
            newFeaturesVector.extend( self.featuresVector[ removedIndex + 1 : ] )


        # Make the new list into a numpy array
        self.featuresVector =   np.array( newFeaturesVector )

        # Delete the feature from the dictionary
        del self.featureNames[ feature ]

        # Set new indexes
        for featureName in self.featureNames:

            if self.featureNames[ featureName ] > removedIndex:
                self.featureNames[ featureName ] -= 1

# This class contains information about a specific prediction made
class Prediction( ):

    # This function runs automatically when a Prediction() object is made
    def __init__( self, predVector ):

        # Set the predition vector. For example [ 0.56, 0.76, 0.54 ]
        self.vector             = predVector

        # The feature vector the prediction was made out of. For example [ 1.2, 5.3, 2.3, 4.5 ]
        self.featureVector      = None

        # The predicted class vector. The highest index of the prediction vector decides the class vector
        # For example if the prediction vector is [ 0.54, 0.32, 0.81 ] the class vector is [ 0, 0, 1 ]
        self.classVector        = None

        # The name of the class corresponding to the predicted class vector.
        # For ex. [ 0, 0, 1 ] = Iris virginica
        self.className          = None

        # The correct class vector corresponding to the feature vector
        self.actualClassVector  = None

        # The name of the correct class corresponding to the feature vector
        self.actualClassName    = None

# This class contains the sorted datasets
class Dataset():

    # When a class object is initialized
    def __init__( self , trainingSetSize):

        # Fetch data from the text files and sort it into a training set and a testing set
        self.__fetchData()
        self.makeSets( trainingSetSize )

        # Keep track of how many classes and features there are
        self.classes        =   3
        self.features       =   4

        # List with names of the classes
        self.classNames     =   [ "Iris setosa", "Iris versicolour", "Iris virginica" ]

    # Read raw data from textfiles
    def __fetchData(self):

        # Open the data files
        with    open( "irisData/class_1" ) as setosa        ,\
                open( "irisData/class_2" ) as versicolour   ,\
                open( "irisData/class_3" ) as virginica     :

            # Makes a list with each line as a string
            setosa              =   setosa.readlines()
            versicolour         =   versicolour.readlines()
            virginica           =   virginica.readlines()

            # Initialize empty lists to be filled with feature vectors
            setosaVectors       =   []
            versicolourVectors  =   []
            virginicaVectors    =   []

            # Iterate through each line for all the classes
            for lineSetosa, lineVersicolour, lineVirginica in zip( setosa, versicolour, virginica ):

                # Split up the line by ',', remove the '\n' from the end and create a numpy array of the values
                # Append this new vector to setosaVectors
                lineSetosa          =   lineSetosa[ :-1 ].split( "," )
                setosaVector        =   np.array( [ float( feature ) for feature in lineSetosa ] )
                setosaVectors.append( setosaVector )

                # Split up the line by ',', remove the '\n' from the end and create a numpy array of the values
                # Append this new vector to versicolourVectors
                lineVersicolour     =   lineVersicolour[ :-1 ].split( "," )
                versicolourVector   =   np.array( [ float( feature ) for feature in lineVersicolour ] )
                versicolourVectors.append( versicolourVector )

                # Split up the line by ',', remove the '\n' from the end and create a numpy array of the values
                # Append this new vector to virginicaVectors
                lineVirginica       =   lineVirginica[ :-1 ].split( "," )
                virginicaVector     =   np.array( [ float( feature ) for feature in lineVirginica ] )
                virginicaVectors.append( virginicaVector )

            # Make member variables containing each array of feature vectors for each class
            self.setosa             =   np.array( setosaVectors )
            self.versicolour        =   np.array( versicolourVectors )
            self.virginica          =   np.array( virginicaVectors )

    # Sort the arrays of feature vector into a training set and a testing set
    def makeSets(self, trainingSetSize):

        # Initialize empty lists
        trainingSet                 =   []
        testingSet                  =   []

        # Slice each list of vectors to the trainingSetSize argument to make the training sets
        self.setosa_Training        = self.setosa     [ :trainingSetSize ]
        self.versicolour_Training   = self.versicolour[ :trainingSetSize ]
        self.virginica_Training     = self.virginica  [ :trainingSetSize ]

        # Create a sampleVector object containing sample data, class vector and class label for each feature vector
        #  in the training set.
        # Also add 1 to the end of every vector as vaguely described on page 15 of the classification compendium
        for setosa, versicolour, virginica in zip( self.setosa_Training, self.versicolour_Training, self.virginica_Training ):

            trainingSet.append( SampleVector( np.append( setosa     , [ 1 ] ), np.array( [ 1, 0, 0 ] ), "setosa"      ) )
            trainingSet.append( SampleVector( np.append( versicolour, [ 1 ] ), np.array( [ 0, 1, 0 ] ), "versicolour" ) )
            trainingSet.append( SampleVector( np.append( virginica  , [ 1 ] ), np.array( [ 0, 0, 1 ] ), "virginica"   ) )

        # Slice each list of vectors from the trainingSetSize argument to make the testing sets
        self.setosa_Testing         = self.setosa     [ trainingSetSize: ]
        self.versicolour_Testing    = self.versicolour[ trainingSetSize: ]
        self.virginica_Testing      = self.virginica  [ trainingSetSize: ]

        # Create a sampleVector object containing sample data, class vector and class label for each feature vector
        # in the testing set.
        # Also add 1 to the end of every vector as vaguely described on page 15 of the classification compendium
        for setosa, versicolour, virginica in zip( self.setosa_Testing, self.versicolour_Testing, self.virginica_Testing ):

            testingSet.append( SampleVector( np.append( setosa     , [ 1 ] ), np.array( [ 1, 0, 0 ] ), "setosa"     ) )
            testingSet.append( SampleVector( np.append( versicolour, [ 1 ] ), np.array( [ 0, 1, 0 ] ), "versicolour") )
            testingSet.append( SampleVector( np.append( virginica  , [ 1 ] ), np.array( [ 0, 0, 1 ] ), "virginica"  ) )

        # Create member variables of the two sets
        self.trainingSet    =   trainingSet
        self.testingSet     =   testingSet

    # Removes a specified feature from each feature vector in both sets
    def removeFeature( self, feature ):

        # Run the member function removeFeature() of the SampleVector class for every SampleVector object
        for trainingSampleVector in self.trainingSet:

            trainingSampleVector.removeFeature( feature )

        for testingSampleVector in self.testingSet:
            testingSampleVector.removeFeature ( feature )

        # Decrement the features count by 1
        self.features   -=  1

# The linear classifier
class IrisClassifier():

    # Runs when a IrisClassifier() object is made
    def __init__(self, trainingSetSize ):

        # Fetch the dataset by making a Dataset() object from the Dataset() class
        self.__dataset              =   Dataset( trainingSetSize )

    # Train the classifier. Runs through a list of functions a specified number of times.
    # The weighting matrix is reset every time this function is called, meaning all previous training is
    # nullified.
    def trainClassifier(self, iterations, stepFactor, stopTraining = False):

        # Initialize lists for MSE and error rate pr iteration. Make these private member variables.
        # This means they can only be accesed inside the definition of thsi class. Doesn't really matter, I
        # just thinks it makes the code more tidy.
        self.__MSEprIterationTraining           =   []
        self.__MSEprIterationTesting            =   []
        self.__errorRatePrIterationTrainingSet  =   []
        self.__errorRatePrIterationTestingSet   =   []

        # Initialize an empty weighting matrix with dimension ( nr.classes x nr.features + 1 )
        # We must add 1 to the features because of the 1 we added to the end of all feature vectors.
        # Again, this is vaguely desccribed on page 15 in the classification compendium.
        self.weightingMatrix        =   np.zeros( ( self.__dataset.classes, self.__dataset.features + 1 ) )

        # Set the step factor as a private member variable
        self.__alpha                =   stepFactor

        # Create a member variable that let's the classifier know if the MSE has converged. For lower stepFactor values, the
        # MSE will just keep getting smaller and smaller for every iteration, but it will converge. Even though it decreases it
        # may just decrease by 10^-9 pr iteration for iterations over 3000, so there's really no benefit to keep iterating.
        # Therefore we will stop the training when the MSE ha reached a certain point.
        self.finishedTraining       =   False

        # Run through a set of functions a specified number of times
        for i in range( iterations ):

            if i != 0:
                # Update the weighting matrix based on the predictions
                self.__updateWeightingMatrix()

            # Get predictions (make Prediction() objects) for each feature vector in both training and testing sets
            self.__getPredictions()

            # Test classifier by finding the mean squared error for the testing set and error rates for both the training
            # and testing sets
            self.__getMSE( stopTraining )
            self.__getErrorRate()

            if self.finishedTraining:

                break

    # Based on the current weighting matrix, calculate the predictions for each sample vector in the training
    # and testing sets.
    def __getPredictions( self ):

        # This function will multiply our sample vector with our weighting matrix.
        # Since we need our resulting vector values to be between 0 and 1, we apply the sigmoid function to
        # the results of each matrix multiplication.
        # See equation (3.20) from the compendium on classification

        # The function will return two arrays of Prediction() objects, each corresponding to their input sample vector
        # In mathematical terms, each prediction vector is calculated according to the formula:

        #                               1
        #           -----------------------------------------
        #            -(sample vector * weighting matrix)
        #           e                                    + 1

        # The resulting vectors should be in the form of [0<x<1, 0<y<1, 0<z<1]^T, where x, y, and z corresponds to the
        # different classes/species of the Iris flower. The greatest value decides what Iris flower the sample will be
        # classified as.

        # Caluclate the prediction vector of each sample in the training set and make a list of Prediction() objects
        exponentials    =   np.array( [ np.exp( - ( np.matmul( self.weightingMatrix, sample.featuresVector ) ) )
                            for sample in self.__dataset.trainingSet ] )
        denominators    =   exponentials + 1

        self.__trainingPredictions  =    np.array( [ Prediction( 1 / denominator ) for denominator in denominators ] )

        # Caluclate the prediction vector of each sample in the testing set and make a list of Prediction() objects
        exponentials                =   np.array( [ np.exp( - ( np.matmul( self.weightingMatrix, sample.featuresVector ) ) )
                                        for         sample      in         self.__dataset.testingSet ]                     )
        denominators                =   exponentials + 1

        self.__testingPredictions   =    np.array( [ Prediction( 1 / denominator ) for denominator in denominators ] )

        # Iterate through all prediction vectors in the training set and round them up to a predicted class vector
        for trainingPrediction,trainingSample in  zip( self.__trainingPredictions, self.__dataset.trainingSet ):

            # Round up the prediction objects vector and set it as its classVector
            trainingPrediction.classVector  =   np.array( [ i == np.argmax( trainingPrediction.vector )
                                                for         i
                                                in          range( len( trainingPrediction.vector ) ) ], dtype = np.uint8 )

            # Fetch the class name and set the prediction object's className as this
            trainingPrediction.className    =   self.__dataset.classNames[ np.argmax( trainingPrediction.classVector ) ]

            # Fecth the actual class vector, the actual class name, and the feature vector the prediction was made from
            trainingPrediction.actualClassVector    =   trainingSample.classVector
            trainingPrediction.actualClassName      =   trainingSample.className
            trainingPrediction.featureVector        =   trainingSample.featuresVector

        # Iterate through all prediction vectors in the testing set and round them up to a predicted class vector
        for testingPrediction, testingSample in zip( self.__testingPredictions, self.__dataset.testingSet ):

            # Round up the prediction objects vector and set it as its classVector
            testingPrediction.classVector = np.array( [ i == np.argmax( testingPrediction.vector )
                                            for         i
                                            in          range( len( testingPrediction.vector ) ) ], dtype=np.uint8 )

            # Fetch the class name and set the prediction object's className as this
            testingPrediction.className = self.__dataset.classNames[np.argmax(testingPrediction.classVector)]

            # Fecth the actual class vector, the actual class name, and the feature vector the prediction was made from
            testingPrediction.actualClassVector     =   testingSample.classVector
            testingPrediction.actualClassName       =   testingSample.className
            testingPrediction.featureVector         =   testingSample.featuresVector

    # Update the weighting matrix for the next training iteration
    def __updateWeightingMatrix( self ):

        #                                    Implements eq. (3.22) from the compendium on classification
        #
        #             len(samples)
        #                 ___
        #     ∇_W MSE =   \                                                                                                    T
        #                 /__  [ (predictions[k] - rightanswers[k]) ⚬ predictions[k] ⚬ (1 - predictions[k]) ] * sampleVectors[k]
        #                 k=1
        #
        # This calculates the gradient of the Mean Square Error with respect to W. Result will be a vector pointing in the
        # direction of greatest ascent. Here x[]] is sample vector[k]


        grad_g_MSE  =   np.array( [ prediction.vector - prediction.actualClassVector
                        for         prediction
                        in          self.__trainingPredictions ] )

        grad_z_g    =   np.array( [ prediction.vector * ( 1 - prediction.vector )
                        for         prediction
                        in          self.__trainingPredictions ] )

        grad_W_z    =   np.array( [ np.reshape( prediction.featureVector, ( 1, len( prediction.featureVector ) ) )
                        for         prediction
                        in          self.__trainingPredictions ] )

        grad_W_MSE  =   np.sum(     np.matmul( np.reshape( grad_g_MSE[ k ] * grad_z_g[ k ], ( self.__dataset.classes, 1 ) ), grad_W_z[ k ] )
                        for         k
                        in          range( len( grad_g_MSE ) ) )

        #                                    Implements eq. (3.23) from the compendium on classification
        #
        #           updated Weigthing matrix    =   current weigthing matrix  - ( alpha * delta_W MSE )
        #
        # This takes the current weigthing matrix and nudges it in the oposite direction of the MSE gradient. How much it
        # is shifted is decided by the step factor alpha. Set the weighting matrix as a public member variable.

        self.weightingMatrix    =   self.weightingMatrix - ( self.__alpha * grad_W_MSE )

    # Calculate the mean squared error for the testing set for the current iteration
    def __getMSE( self, stopTraining = False ):

        # Calculate the mean squared error for each prediction in the testing set
        predictedVectorsTesting    =   np.array( [ prediction.vector            for prediction in self.__testingPredictions ] )
        actualValuesTesting        =   np.array( [ prediction.actualClassVector for prediction in self.__testingPredictions ] )

        currentMSETesting          =   ( ( predictedVectorsTesting - actualValuesTesting ) ** 2 ).mean()

        # Calculate the mean squared error for each prediction in the training set
        predictedVectorsTraining   =   np.array( [ prediction.vector            for prediction in self.__trainingPredictions ] )
        actualValuesTraining       =   np.array( [ prediction.actualClassVector for prediction in self.__trainingPredictions ] )

        currentMSETraining         =   ( ( predictedVectorsTraining - actualValuesTraining ) ** 2 ).mean()

        # Stops training when MSE for the testing set converges if stopTraining parameter is True.
        if stopTraining:

            if len( self.__MSEprIterationTesting ) > 1:

                # Check if MSE is reduced by less than 0.1%. If so, stop the training.
                if  (   currentMSETesting   < self.__MSEprIterationTesting[ -1 ]
                and     currentMSETesting   > self.__MSEprIterationTesting[ -1 ] - ( self.__MSEprIterationTesting[ -1 ] * 0.0001 )
                and     self.__MSEprIterationTesting[ -1 ]  > self.__MSEprIterationTesting[ -2 ] - ( self.__MSEprIterationTesting[ -2 ] * 0.0001 ) ):

                    self.finishedTraining   =   True

        # Add the MSE for the current iteration to the list of MSE pr iteration
        self.__MSEprIterationTraining.append( currentMSETraining )
        self.__MSEprIterationTesting.append ( currentMSETesting  )

    # Caluclate the error rate for the training set and the testing set for the current iteration
    def __getErrorRate( self ):

        nrSamplesTrainingSet    =   len( self.__trainingPredictions )
        nrSamplesTestingSet     =   len( self.__testingPredictions  )
        errorsTraining          =   0
        errorsTesting           =   0

        for prediction in self.__testingPredictions:

            if not np.array_equal( prediction.classVector, prediction.actualClassVector):

                errorsTesting   +=  1

        for prediction in self.__trainingPredictions:

            if not np.array_equal( prediction.classVector, prediction.actualClassVector):

                errorsTraining   +=  1

        errorRateTrainingSet    =   errorsTraining / nrSamplesTrainingSet
        errorRateTestingSet     =   errorsTesting  / nrSamplesTestingSet

        self.__errorRatePrIterationTestingSet.append ( errorRateTestingSet  )
        self.__errorRatePrIterationTrainingSet.append( errorRateTrainingSet )

    # Calculate the confusion matrix for the current iteration
    def __getConfusionMatrix( self, weightingMatrix ):

        # Set the weighting matrix to the argument input
        self.weightingMatrix    =   weightingMatrix

        # Fetch predictions based on the weighting matrix
        self.__getPredictions()

        # Initialize empty lists for each confusion matrix
        confusionMatrixTraining = []
        confusionMatrixTesting  = []

        # Iterate through the class vectors
        for predClass in [ np.array( [ 1, 0, 0 ] ),
                           np.array( [ 0, 1, 0 ] ),
                           np.array( [ 0, 0, 1 ] ) ]:

            collumnTesting =   []
            rowTesting     =   \
                {
                    "setosa"        :   0,
                    "versicolour"   :   0,
                    "virginica"     :   0
                }

            collumnTraining =   []
            rowTraining     =   \
                {
                    "setosa"        :   0,
                    "versicolour"   :   0,
                    "virginica"     :   0
                }

            for prediction in self.__trainingPredictions:

                if np.array_equal( prediction.classVector, predClass ):

                    rowTraining[prediction.actualClassName] += 1

            collumnTraining.append         ( rowTraining[ "setosa"      ] )
            collumnTraining.append         ( rowTraining[ "versicolour" ] )
            collumnTraining.append         ( rowTraining[ "virginica"   ] )
            confusionMatrixTraining.append( collumnTraining )

            for prediction in self.__testingPredictions:

                if np.array_equal( prediction.classVector, predClass ):

                    rowTesting[prediction.actualClassName] += 1

            collumnTesting.append         ( rowTesting[ "setosa"      ] )
            collumnTesting.append         ( rowTesting[ "versicolour" ] )
            collumnTesting.append         ( rowTesting[ "virginica"   ] )
            confusionMatrixTesting.append( collumnTesting )

        return np.array( confusionMatrixTraining), np.array( confusionMatrixTesting )

    # Plot the confusion matrix for either the training set or the testing set
    def plotConfusionMatrix( self, type = "regular" ):

        # If type is "regular", calculate the confusion matrices based on the current weighting matrix
        if type == "regular":

            trainingMatrix, testingMatrix   =   self.__getConfusionMatrix( self.weightingMatrix )

            confusionMatrixTrainingDf = pd.DataFrame(trainingMatrix,
                                                     index=self.__dataset.classNames,
                                                     columns=self.__dataset.classNames)
            confusionMatrixTestingDf = pd.DataFrame(testingMatrix,
                                                    index=self.__dataset.classNames,
                                                    columns=self.__dataset.classNames)

            # Create two subfigures with confusion matrix for training and for testing
            fig, ( trainingMatrix, testingMatrix ) = plt.subplots( ncols = 2, figsize = ( 10, 5 ) )

            sn.heatmap(confusionMatrixTrainingDf, annot=True, ax=trainingMatrix)
            sn.heatmap(confusionMatrixTestingDf , annot=True, ax=testingMatrix)

            testingMatrix.set_title(f"Confusion matrix for the testing set, α = {self.__alpha}")
            trainingMatrix.set_title(f"Confusion matrix for the training set, α = {self.__alpha}")

        # Else, calculate confusion matrices for optimal and least optimal value of alpha
        else:

            optTrainingMatrix   , optTestingMatrix      =   self.__getConfusionMatrix( self.optAndSubOptWMatrix[ 0 ] )
            suboptTrainingMatrix, suboptTestingMatrix   =   self.__getConfusionMatrix( self.optAndSubOptWMatrix[ 1 ] )

            optConfusionMatrixTrainingDf    = pd.DataFrame( optTrainingMatrix,
                                                            index=self.__dataset.classNames,
                                                            columns=self.__dataset.classNames)
            optConfusionMatrixTestingDf    = pd.DataFrame( optTestingMatrix,
                                                            index=self.__dataset.classNames,
                                                            columns=self.__dataset.classNames)
            suboptConfusionMatrixTrainingDf = pd.DataFrame( suboptTrainingMatrix,
                                                            index=self.__dataset.classNames,
                                                            columns=self.__dataset.classNames)
            suboptConfusionMatrixTestingDf = pd.DataFrame( suboptTestingMatrix,
                                                            index=self.__dataset.classNames,
                                                            columns=self.__dataset.classNames)

            # Create two subfigures with confusion matrix for training and for testing
            fig, [( optTraining, optTesting ), ( suboptTraining, suboptTesting ) ] = plt.subplots( ncols = 2, nrows = 2, figsize = ( 10, 5 ) )

            sn.heatmap(optConfusionMatrixTrainingDf   , annot=True, ax = optTraining    )
            sn.heatmap(optConfusionMatrixTestingDf    , annot=True, ax = optTesting     )
            sn.heatmap(suboptConfusionMatrixTrainingDf, annot=True, ax = suboptTraining )
            sn.heatmap(suboptConfusionMatrixTestingDf , annot=True, ax = suboptTesting  )

            optTesting.set_title(f"Confusion matrix for the testing set with optimal α = {self.__optSuboptAlpha[ 0 ]}")
            optTraining.set_title(f"Confusion matrix for the training set with optimal α = {self.__optSuboptAlpha[ 0 ]}")
            suboptTesting.set_title(f"Confusion matrix for the testing set with lest optimal α = {self.__optSuboptAlpha[ 1 ]}")
            suboptTraining.set_title(f"Confusion matrix for the training set with least optimal α = {self.__optSuboptAlpha[ 1 ]}")



        plt.show()

    # Change the size of the training set and the testing set
    def changeSetSize( self, trainingSetSize ):

        self.__dataset.makeSets( trainingSetSize )

    # Remove a feature from the dataset
    def removeFeature( self, feature ):

        self.__dataset.removeFeature( feature )

    # Plot error rates  and MSE pr. iteration for the training set and the testing set.
    # Also finds what alpha value gives the lowest error rates at convergence
    def plotMSEandError( self, iterations, alphaValues = [ 0.01, 0.0075, 0.005, 0.0025 ] ):

        # Make lists to be filled with values pr iteration for each alpha value
        errorRateprAlphaTraining    =   []
        errorRateprAlphaTesting     =   []
        MSEprAlphaTraining          =   []
        MSEprAlphaTesting           =   []

        # Make a list with the weighting matrixes at convergence for each alpha value.
        # Later we find the optimal alpha value and use that weighting matrix to plot confusion matrices
        weightingMatrices           =   []

        alphasFinished  =   1
        # Iterate through each alpha value
        for stepFactor in alphaValues:

            # Train the classifier with the current alpha value. Stop training when mean squared error converges
            self.trainClassifier( iterations, stepFactor, stopTraining = True )

            # Add the values pr. iteration to the values pr. alpha lists
            errorRateprAlphaTraining.append( self.__errorRatePrIterationTrainingSet )
            errorRateprAlphaTesting.append ( self.__errorRatePrIterationTestingSet )
            MSEprAlphaTraining.append      ( self.__MSEprIterationTraining )
            MSEprAlphaTesting.append       ( self.__MSEprIterationTesting)
            weightingMatrices.append( self.weightingMatrix )

            # Print out in readable format
            #print(f"Alphas trained: {alphasFinished} / {len(alphaValues)}\t-\t{len(self.__errorRatePrIterationTrainingSet)} iterations\t-\tAlpha: {stepFactor}")
            #alphasFinished += 1

            # Print out in latex table format. noConverge if training doesn't converge within the number of iterations set
            if len( self.__errorRatePrIterationTrainingSet ) == iterations:
                print(f"{stepFactor}    \t&\t{'No convergence'} \t&\t"
                      f"{round(self.__MSEprIterationTesting[- 1], 4)}\t&\t"
                      f"{round(self.__errorRatePrIterationTrainingSet[-1], 4)}\t&\t"
                      f"{round(self.__errorRatePrIterationTestingSet[-1], 4)}\t \\\\ \hline")
            else:
                # Print out in latex table format
                print(f"{stepFactor}    \t&\t{ len( self.__errorRatePrIterationTrainingSet ) } \t&\t"
                      f"{ round( self.__MSEprIterationTesting[ - 1 ], 4 ) }\t&\t"
                      f"{ round( self.__errorRatePrIterationTrainingSet[ -1 ], 4 ) }\t&\t"
                      f"{ round( self.__errorRatePrIterationTestingSet[ -1 ], 4 ) }\t \\\\ \hline")

            alphasFinished += 1

        # Create a figure with four subplots in 2x2 grid
        allPlots, [( errorRatesTraining, errorRatesTesting), (MSEsTraining, MSEsTesting )] = plt.subplots( ncols = 2, nrows = 2, figsize = ( 10, 5 ) )

        # Create a figure with just the MSEs
        justMSEs, ( justMSEsTraining   , justMSEsTesting ) =   plt.subplots( ncols = 2, figsize = ( 10, 5 ) )

        # Create a figure with just the error rates
        justErrorRates, ( justErrorRatesTraining   , justErrorRatesTesting )   =   plt.subplots( ncols = 2, figsize = ( 10, 5 ) )

        # Create a figure with four subplots in 2x2 grid
        optSuboptPlots, [ ( optSuboptErrorRatesTraining, optSuboptErrorRatesTesting ),
                          (optSuboptMSEsTraining       , optSuboptMSEsTesting       ) ] = plt.subplots( ncols = 2, nrows = 2, figsize = ( 10, 5 ) )


        # Iterate through the indexes in the values pr. alpha value lists
        for index in range( len( alphaValues ) ):

            # Fetch the values pr. iteration for the current alpha value
            errorRateTraining   =   errorRateprAlphaTraining[ index ]
            errorRateTesting    =   errorRateprAlphaTesting [ index ]
            MSETraining         =   MSEprAlphaTraining      [ index ]
            MSETesting          =   MSEprAlphaTesting       [ index ]
            alpha               =   alphaValues             [ index ]

            errorRatesTraining.plot( range( len( errorRateTraining ) ), errorRateTraining, label='$\\alpha={' + str(alpha) + '}$' )
            errorRatesTesting.plot ( range( len( errorRateTesting  ) ), errorRateTesting , label='$\\alpha={' + str(alpha) + '}$' )

            MSEsTraining.plot      ( range( len( MSETraining       ) ), MSETraining      , label='$\\alpha={' + str(alpha) + '}$' )
            MSEsTesting.plot       ( range( len( MSETesting        ) ), MSETesting       , label='$\\alpha={' + str(alpha) + '}$' )

            justMSEsTraining.plot  ( range( len( MSETraining       ) ), MSETraining      , label='$\\alpha={' + str(alpha) + '}$' )
            justMSEsTesting.plot   ( range( len( MSETesting        ) ), MSETesting       , label='$\\alpha={' + str(alpha) + '}$' )

            justErrorRatesTraining.plot( range( len( errorRateTraining ) ), errorRateTraining, label='$\\alpha={' + str(alpha) + '}$' )
            justErrorRatesTesting.plot ( range( len( errorRateTesting  ) ), errorRateTesting , label='$\\alpha={' + str(alpha) + '}$' )

        # Make a list with the minimum values for each alpha value
        minValuesPrAlpha    =   []

        # Initialize variables to find lowest and highest errors as the convergence error of the first alpha value
        # Second index in the list is number of iterations, third index is the index of the alpha value
        lowestTestingError      =   [ 1, 0, 0 ]
        lowestTrainingError     =   [ 1, 0, 0 ]
        highestTestingError     =   [ 0, 0, 0 ]
        highestTrainingError    =   [ 0, 0, 0 ]

        # Iterate through each list in the values pr. alpha lists
        index   =   0

        for trainingErrors, testingErrors, MSETraining, MSETesting \
        in  zip( errorRateprAlphaTraining, errorRateprAlphaTesting, MSEprAlphaTraining, MSEprAlphaTesting):

            # For this bulk of code we will find what alpha values provides the lowest errors with the least
            # amount of iterations
            # Check if error rates for testing at convergence is lower than lowestTestingError.
            # Also make sure the training actualy as reached convergence
            if testingErrors[ -1 ] <= lowestTestingError[ 0 ] and len( testingErrors ) != iterations:

                # Check if the error rates are equal
                if testingErrors[ - 1 ] == lowestTestingError[ 0 ]:

                    #Check if the training error rate is lower than lowestTrainingError
                    if trainingErrors[ - 1 ] <= lowestTrainingError[ 0 ]:

                        # Check if the error rates are equal:
                        if trainingErrors[ - 1 ] == lowestTrainingError[ 0 ]:

                            # Check which one had the least amount of iterations
                            if len( trainingErrors ) < lowestTrainingError[ 1 ]:

                                lowestTestingError  =   [ testingErrors [ -1 ], len( testingErrors  ), index ]
                                lowestTrainingError =   [ trainingErrors[ -1 ], len( trainingErrors ), index ]

                            # If not, do nothing
                            else:

                                pass

                        # If not, replace the lowest variables with the error rate and number of iterations for training and testing
                        else:

                            # Replace the lowest variables with the error rate and number of iterations for testing and training
                            lowestTestingError  = [ testingErrors [ - 1 ], len( testingErrors  ), index ]
                            lowestTrainingError = [ trainingErrors[ - 1 ], len( trainingErrors ), index ]

                    # If not, do nothing
                    else:

                        pass

                else:

                    # Replace the lowest variables with the error rate and number of iterations for testing and training
                    lowestTestingError  = [ testingErrors [ - 1 ], len( testingErrors  ), index ]
                    lowestTrainingError = [ trainingErrors[ - 1 ], len( trainingErrors ), index ]

            # Check if error rates for testing at convergence is higher than highestTestingError.
            # Also make sure the training actualy as reached convergence
            if testingErrors[ -1 ] >= highestTestingError[ 0 ] and len( testingErrors ) != iterations:

                # Check if the error rates are equal
                if testingErrors[ - 1 ] == highestTestingError[ 0 ]:

                    #Check if the training error rate is higher than highesteTrainingError
                    if trainingErrors[ - 1 ] >= highestTrainingError[ 0 ]:

                        # Check if the error rates are equal:
                        if trainingErrors[ - 1 ] == lowestTrainingError[ 0 ]:

                            # Check which one had the highest amount of iterations
                            if len( trainingErrors ) > highestTrainingError[ 1 ]:

                                highestTestingError  =   [ testingErrors [ -1 ], len( testingErrors  ), index ]
                                highestTrainingError =   [ trainingErrors[ -1 ], len( trainingErrors ), index ]

                            # If not, do nothing
                            else:

                                pass

                        # If not, replace the lowest variables with the error rate and number of iterations for training and testing
                        else:

                            # Replace the lowest variables with the error rate and number of iterations for testing and training
                            highestTestingError  = [ testingErrors [ - 1 ], len( testingErrors  ), index ]
                            highestTrainingError = [ trainingErrors[ - 1 ], len( trainingErrors ), index ]

                    # If not, do nothing
                    else:

                        pass

                else:

                    # Replace the lowest variables with the error rate and number of iterations for testing and training
                    highestTestingError  = [ testingErrors [ - 1 ], len( testingErrors  ), index ]
                    highestTrainingError = [ trainingErrors[ - 1 ], len( trainingErrors ), index ]


            # Convert to numpy arrays
            trainingErrors  =   np.array( trainingErrors )
            testingErrors   =   np.array( testingErrors  )
            MSETraining     =   np.array( MSETraining    )
            MSETesting      =   np.array( MSETesting     )

            # Find index with the lowest value in each list
            minTrainingErrorIndex   =   np.argmin( trainingErrors )
            minTestingErrorIndex    =   np.argmin( testingErrors  )
            minMSETrainingIndex     =   np.argmin( MSETraining    )
            minMSETestingIndex      =   np.argmin( MSETesting     )

            # Find the minimum values
            minTrainingError        =   round( trainingErrors[ minTrainingErrorIndex ], 4 )
            minTestingError         =   round( testingErrors [ minTestingErrorIndex  ], 4 )
            minMSETraining          =   round( MSETraining   [ minMSETrainingIndex   ], 4 )
            minMSETesting           =   round( MSETesting    [ minMSETestingIndex    ], 4 )

            # Find the iteration number off the lowest values
            minTrainingErrorIteration   =   minTrainingErrorIndex + 1
            minTestingErrorIteration    =   minTestingErrorIndex  + 1
            minMSETrainingIteration     =   minMSETrainingIndex   + 1
            minMSETestingIteration             =   minMSETestingIndex    + 1

            # Append to the list as a string to be printed
            minValuesPrAlpha.append(
                f"Alpha:\t{alphaValues[ index ]}\n\n"
                f"\t\tIterations until convergence              :\t{len( trainingErrors )}\n"
                f"\t\tError rate for training set at convergence:\t{round( trainingErrors[ - 1 ], 4 )}\n"
                f"\t\tError rate for testing set at convergence :\t{round( testingErrors[ - 1 ], 4 )}\n"
                f"\t\tMSE for training set at convergence       :\t{round( MSETraining[ -1 ], 4 )}\n"
                f"\t\tMSE for testing set at convergence        :\t{round( MSETesting[ -1 ], 4 )}\n\n"
                f"\t\tMin. error rate training set              :\t{minTrainingError}\t\tafter {minTrainingErrorIteration} iterations.\n"
                f"\t\tMin. error rate testing set               :\t{minTestingError}\t\tafter {minTestingErrorIteration} iterations.\n"
                f"\t\tMin. MSE for training set                 :\t{minMSETraining}\t\tafter {minMSETrainingIteration} iterations.\n"
                f"\t\tMin. MSE for testing set                  :\t{minMSETesting}\t\tafter {minMSETestingIteration} iterations.\n"
                f"\n" )


            index   +=  1

        # Print readable data to console
        #for i in minValuesPrAlpha:
            #print( i )

        # Make a graph with just the optimal and least optimal alpha value
        # Iterate through the indexes in the values pr. alpha value lists
        its     =   0
        for index in [ lowestTrainingError [ 2 ], highestTrainingError[ 2 ] ]:

            if its == 0:
                type = "Optimal"
            else:
                type = "LeastOptimal"

            # Fetch the values pr. iteration for the current alpha value
            errorRateTraining   =   errorRateprAlphaTraining[ index ]
            errorRateTesting    =   errorRateprAlphaTesting [ index ]
            MSETraining         =   MSEprAlphaTraining      [ index ]
            MSETesting          =   MSEprAlphaTesting       [ index ]
            alpha               =   alphaValues             [ index ]

            optSuboptErrorRatesTraining.plot( range( len( errorRateTraining ) ), errorRateTraining, label='${' + type + '}\\ alpha={' + str(alpha) + '}$' )
            optSuboptErrorRatesTesting.plot ( range( len( errorRateTesting  ) ), errorRateTesting , label='${' + type + '}\\ alpha={' + str(alpha) + '}$' )

            optSuboptMSEsTraining.plot      ( range( len( MSETraining       ) ), MSETraining      , label='${' + type + '}\\ alpha={' + str(alpha) + '}$' )
            optSuboptMSEsTesting.plot      ( range( len( MSETesting        ) ), MSETesting       , label='${' + type + '}\\ alpha={' + str(alpha) + '}$' )

            its += 1

        # Make variable with optimal and least optimal weighting matrix
        self.optAndSubOptWMatrix    =   [ weightingMatrices[ lowestTestingError[ 2 ] ], weightingMatrices[ highestTestingError[ 2 ] ] ]

        # Make variable with optimal and least optimal alpha value
        self.__optSuboptAlpha       =   [ alphaValues[ lowestTrainingError[ 2 ] ], alphaValues[ highestTrainingError[ 2 ] ] ]

        # Print the optimal and least optimal alpha values
        print( f"Least optimal alpha value: {alphaValues[ highestTrainingError[ 2 ] ] }" )
        print( f"Optimal alpha value      : {alphaValues[ lowestTrainingError [ 2 ] ] }" )

        # Set the weighting matrix to the one at convergence for the optimal alpha value
        self.weightingMatrix    =   weightingMatrices[ lowestTestingError[ 2 ] ]

        errorRatesTraining.set_xlabel("Iteration number")
        errorRatesTraining.set_ylabel("Error rate")
        errorRatesTraining.set_title("Error rates pr. iteration for the training set")

        errorRatesTesting.set_xlabel("Iteration number")
        errorRatesTesting.set_ylabel("Error rate")
        errorRatesTesting.set_title("Error rates pr. iteration for the testing set")

        justErrorRatesTraining.set_xlabel("Iteration number")
        justErrorRatesTraining.set_ylabel("Error rate")
        justErrorRatesTraining.set_title("Error rates pr. iteration for the training set")

        justErrorRatesTesting.set_xlabel("Iteration number")
        justErrorRatesTesting.set_ylabel("Error rate")
        justErrorRatesTesting.set_title("Error rates pr. iteration for the testing set")

        MSEsTraining.set_xlabel("Iteration number")
        MSEsTraining.set_ylabel("Mean Squared Error")
        MSEsTraining.set_title("MSE pr iteration for the training set")

        MSEsTesting.set_xlabel("Iteration number")
        MSEsTesting.set_ylabel("Mean Squared Error")
        MSEsTesting.set_title("MSE pr iteration for the testing set")

        justMSEsTraining.set_xlabel("Iteration number")
        justMSEsTraining.set_ylabel("Mean Squared Error")
        justMSEsTraining.set_title("MSE pr iteration for the training set")

        justMSEsTesting.set_xlabel("Iteration number")
        justMSEsTesting.set_ylabel("Mean Squared Error")
        justMSEsTesting.set_title("MSE pr iteration for the testing set")

        optSuboptErrorRatesTraining.set_xlabel("Iteration number")
        optSuboptErrorRatesTraining.set_ylabel("Error rate")
        optSuboptErrorRatesTraining.set_title("Error rates pr. iteration the training set")

        optSuboptErrorRatesTesting.set_xlabel("Iteration number")
        optSuboptErrorRatesTesting.set_ylabel("Error rate")
        optSuboptErrorRatesTesting.set_title("Error rates pr. iteration for the testing set")

        optSuboptMSEsTraining.set_xlabel("Iteration number")
        optSuboptMSEsTraining.set_ylabel("Mean Squared Error")
        optSuboptMSEsTraining.set_title("MSE pr iteration for the training set")

        optSuboptMSEsTesting.set_xlabel("Iteration number")
        optSuboptMSEsTesting.set_ylabel("Mean Squared Error")
        optSuboptMSEsTesting.set_title("MSE pr iteration for the testing set")

        errorRatesTraining.legend(fontsize = 7)
        errorRatesTesting.legend(fontsize = 7)
        MSEsTraining.legend(fontsize = 7)
        MSEsTesting.legend(fontsize = 7)
        justMSEsTraining.legend()
        justMSEsTesting.legend()
        justErrorRatesTesting.legend()
        justErrorRatesTraining.legend()
        optSuboptMSEsTesting.legend()
        optSuboptMSEsTraining.legend()
        optSuboptErrorRatesTesting.legend()
        optSuboptErrorRatesTraining.legend()

        # Plot confusion matrices for that alpha value at convergence
        self.plotConfusionMatrix( "optSubopt")


        plt.figure(allPlots)
        plt.show()
        plt.figure(justMSEs)
        plt.show()
        plt.figure(justErrorRates)
        plt.show()
        plt.figure(optSuboptPlots)
        plt.show()

    # Sets the training set as the testing set and vica versa
    def switchSets( self ):

        training    =   self.__dataset.testingSet
        testing     =   self.__dataset.trainingSet

        self.__dataset.testingSet   =   testing
        self.__dataset.trainingSet  =   training

