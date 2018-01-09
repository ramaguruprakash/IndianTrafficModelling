from sumoDataLoader import SumoDataLoader
dataLoader = SumoDataLoader("/Users/gramaguru/SumoNetowrk_basic/simulation_1000sec_100cars.xml",0.1,0.1,1,3)
X = dataLoader.train_X
y = dataLoader.train_y
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(X,y[:,1]).predict(X)
y_rbf_x = svr_rbf.fit(X,y[:,0]).predict(X)
