#Defining how to open other python programs on this one to create the pipeline
def run(runfile):
    with open(runfile,"r") as rnf:
        exec(rnf.read())



#Data Preprocessing, feature Engineering and dimensionality reduction of our pipeline
#Will generate numpy array files to be used by all the other methods
run("feature_creator.py")



#Learning algorithms of our pipeline
#Will generate numpy array files to be used by the final stacking method
#The files are already generated as this process could take a few hours
#run("ExtraTrees.py")
#run("catboostLearning.py")
#run("Logistic.py")
#run("NNCVLearning.py")
#run("RandomForest.py")
#run("SVC.py")

#Stacking algorithm and evaluation of our pipeline
#Will generate the final excel file to be submitted on kaggle, as well as some validation loss score to evaluate the models
run("catStackFinal.py")


