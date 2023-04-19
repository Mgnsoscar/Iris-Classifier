from    irisClassifier  import  *

if __name__ == '__main__':

    # Create a list with 20 different alpha values
    lst = []
    # append alpha values from 0.01 to 0.001 with 0.001 increments
    for i in range(10):
        num = round(0.01 - (i * (0.01 / 10)), 4)
        lst.append(num)

# Task 1a, 1b, 1c)
    clas = IrisClassifier(30)
    #clas.plotMSEandError(3000, lst)

    # Plot confusion matrices for alpha = 0.009
    #clas.trainClassifier(3000, 0.009, stopTraining= True)
    #clas.plotConfusionMatrix("regular")


# Task 1d)
    # Change training set size to first 20, and switch training and testing sets
    #clas.changeSetSize( 20 )
    #clas.switchSets()
    #clas.plotMSEandError( 3000, lst )

    # Plot the confusion matrices for the optimal alpha value from task 1a-1c)
    #clas.trainClassifier(3000, 0.009, stopTraining = True)
    #clas.plotConfusionMatrix("regular")

# Task 2a) Remove sepal width feature and train with first 30 samples
    #clas.changeSetSize( 30 )
    #clas.removeFeature("Sepal width")
    #clas.plotMSEandError(3000, lst)

    # Plot the confusion matrices for the optimal alpha value from task 1a-1c)
    #clas.trainClassifier(3000, 0.009, stopTraining = True)
    #clas.plotConfusionMatrix("regular")

# Task 2b) Remove sepal length also
    #clas.removeFeature("Sepal length")
    #clas.plotMSEandError(3000, lst)

    # Plot the confusion matrices for the optimal alpha value from task 1a-1c)
    #clas.trainClassifier(3000, 0.009, stopTraining = True)
    #clas.plotConfusionMatrix("regular")

    # Then remove petal length. Train only with petal width
    #clas.removeFeature("Petal length")
    #clas.plotMSEandError(2300, lst)

    # Plot the confusion matrices for the optimal alpha value from task 1a-1c)
    #clas.trainClassifier(3000, 0.009, stopTraining = True)
    #clas.plotConfusionMatrix("regular")




