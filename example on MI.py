import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
import numpy as np
import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery,MotorImagery
from pyriemann.classification import MDM 
from tWDA import tWDA
from pyriemann.estimation import Covariances



dataset = BNCI2014001()
#dataset.subject_list = [1] ##if you want to be restriced to some subjects


n_classes = 4
#paradigm = LeftRightImagery()
paradigm = MotorImagery(n_classes = n_classes)
X, labels, meta = paradigm.get_data(dataset=dataset)
print(X.shape)

#dof of tWDA classifier
df = 10

n = X.shape[2]-1 #number of time samples
p = X.shape[1] #number of electrodes

pipelines = {}
    
classifier = "rMDM" 
pipeline = make_pipeline(Covariances(),
                         MDM(metric="riemann"))
pipelines[classifier] = pipeline


classifier = "WDA"
pipeline = make_pipeline(Covariances(),
                         tWDA(n=n,df=np.inf))
pipelines[classifier] = pipeline



classifier = "tWDA"
pipeline = make_pipeline(Covariances(),
                         tWDA(n=n,df=df))
pipelines[classifier] = pipeline
    

print(pipelines)


# Evaluate for a specific number of training samples per class
data_size = dict(policy="per_class", value=np.array([57]))
#data_size = dict(policy="ratio", value=np.array([0.2,0.4,0.5,0.6,0.8,0.9]))

# When the training data is sparse, peform more permutations than when we have a lot of data
n_perms = np.floor(np.geomspace(100, 2, len(data_size["value"]))).astype(int)

evaluation = WithinSessionEvaluation(
    paradigm=paradigm,datasets=[dataset],
    overwrite=False,hdf5_path=None,
    #data_size=data_size,
    #n_perms=n_perms,
    random_state=100
    )
results = evaluation.process(pipelines)


sns.barplot(
    x="subject", y="score", hue="pipeline", 
    data=results,ci="sd")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=7)
plt.title("Both sessions : withinsession classification for "+str(n_classes)+" classes \n df="+str(df))
plt.show()

        
