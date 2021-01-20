from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
from subprocess import call
from os.path import isfile
import sys
import struct
 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.constraints import nonneg
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import to_categorical
from keras import Input
import keras.backend as kb
import tensorflow as tf
import root_numpy
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split

from customLoss import customLoss

def remove_field_name(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b


# Generate model
def baseline_Model():
    model = Sequential()
    model.add(Dense(24, input_dim=24, kernel_initializer='normal', activation='relu'))
    #  ~model.add(Dense(25, input_dim=25, kernel_initializer='normal', activation='relu'))
    #  ~model.add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
    #  ~model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear', W_constraint=nonneg()))     #Does not work if input is transformed onto [-1,1]
    #  ~model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def baseline_Model_2():
    model = Sequential()
    #  ~model.add(Dense(24, input_dim=24, kernel_initializer='normal', activation='relu'))
    #  ~model.add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
    model.add(Dense(34, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', W_constraint=nonneg()))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def deep_Model():
    model = Sequential()
    #  ~model.add(Dense(24, input_dim=24, kernel_initializer='normal', activation='relu'))
    model.add(Dense(24, input_shape=(6,4), kernel_initializer='normal', activation='relu'))
    #  ~model.add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
    #  ~model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', W_constraint=nonneg()))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def large_Model():
    model = Sequential()
    model.add(Dense(500, input_dim=24, kernel_initializer='normal', activation= "relu"))
    #  ~model.add(Dense(500, input_dim=24, kernel_initializer='normal',kernel_regularizer=l2(0.001), activation= "relu"))
    #  ~model.add(Dropout(0.2))
    #  ~model.add(Dense(500, input_dim=32, kernel_initializer='normal', activation= "relu"))
    #  ~model.add(Dense(500, input_dim=34, kernel_initializer='normal', activation= "relu"))
    #  ~model.add(Dense(500, kernel_initializer='normal',kernel_regularizer=l2(0.001), activation= "relu"))
    model.add(Dense(500, kernel_initializer='normal', activation= "relu"))
    #  ~model.add(Dropout(0.2))
    model.add(Dense(500, kernel_initializer='normal', activation= "relu"))
    #  ~model.add(Dense(500, kernel_initializer='normal',kernel_regularizer=l2(0.001), activation= "relu"))
    #  ~model.add(Dropout(0.2))
    #  ~model.add(Dense(1, kernel_initializer='normal', W_constraint=nonneg()))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    #  ~lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,decay_steps=20*200,decay_rate=1,staircase=False)

    #  ~model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    return model

def autokeras_Model():
    model = Sequential()
    model.add(Dense(256, input_dim=24, kernel_initializer='normal', activation= "relu"))
    model.add(Dense(256, kernel_initializer='normal', activation= "relu"))
    model.add(Dense(32, kernel_initializer='normal', activation= "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def trainTMVA(tree,inputVars,cutDict):
    # Setup TMVA
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()
    
    #Get Model
    model = baseline_Model()
    model.save('model.h5')
    model.summary()
    #  ~tf.keras.utils.plot_model(model, to_file='model.eps', show_shapes=True, show_layer_names=False)
    #  ~model = baseline_Model_2()
    #  ~model.save('model_2.h5')
    #  ~model.summary()
    #  ~model = deep_Model()
    #  ~model.save('model_deep.h5')
    #  ~model.summary()
    #  ~model = large_Model()
    #  ~model.save('model_large.h5')
    #  ~model.summary()
    #  ~model = autokeras_Model()
    #  ~model.save('model_autokeras.h5')
    #  ~model.summary()

    for MetBin in cutDict:
        
        output = TFile.Open("TMVA_"+MetBin+".root", 'RECREATE')
        factory = TMVA.Factory('TMVARegression', output,
            #  ~'!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Regression')
            #  ~'!V:!Silent:Color:DrawProgressBar:Transformations=U:AnalysisType=Regression')
            '!V:!Silent:Color:DrawProgressBar:Transformations=N:AnalysisType=Regression')
        dataloader = TMVA.DataLoader('dataset')

        for var in inputVars:
          dataloader.AddVariable(var)
       
        dataloader.AddTarget("genMET")
        dataloader.AddSpectator("PuppiMET")
        dataloader.AddSpectator("genMET")
         
        dataloader.AddRegressionTree(tree, 1.0)
        
        #  ~dataloader.SetWeightExpression("abs(genMET-56.6)/10.+1.0","Regression")
        #  ~dataloader.SetWeightExpression("(genMET<100)*1./(exp(-(56.07-genMET)*(56.07-genMET)/(2*21.4*21.4)))","Regression")
        #  ~dataloader.SetWeightExpression("N","Regression")
        #  ~dataloader.SetWeightExpression("N*SF","Regression")
        
        dataloader.PrepareTrainingAndTestTree(TCut(cutDict[MetBin]+" && genDecayMode<=3"),
                #  ~'nTrain_Regression=10000:nTest_Regression=40000:SplitMode=Random:NormMode=None:!V')
                #  ~'nTrain_Regression=10000:nTest_Regression=40000:SplitMode=Random:SplitSeed=0:NormMode=None:!V')
                'nTrain_Regression=10000:nTest_Regression=40000:SplitMode=Alternate:NormMode=None:!V')
                #  ~'nTrain_Regression=240000:nTest_Regression=240000:SplitMode=Alternate:NormMode=None:!V')    #Bin2
                #  ~'nTrain_Regression=74000:nTest_Regression=74000:SplitMode=Alternate:NormMode=None:!V')    #Bin1
                #  ~'nTrain_Regression=150000:nTest_Regression=150000:SplitMode=Alternate:NormMode=None:!V')    #Bin3
                #  ~'nTrain_Regression=56000:nTest_Regression=56000:SplitMode=Alternate:NormMode=None:!V')    #Bin4
                #  ~'nTrain_Regression=22500:nTest_Regression=22500:SplitMode=Alternate:NormMode=None:!V')    #Bin5
        #  ~dataloader.PrepareTrainingAndTestTree(TCut('PuppiMET>230'),
                #  ~'nTrain_Regression=10000::SplitMode=Random:NormMode=NumEvents:!V')
        #  ~dataloader.PrepareTrainingAndTestTree(TCut('PuppiMET>80 && PuppiMET<120'),
                #  ~'nTrain_Regression=10000:nTest_Regression=100000:SplitMode=Random:NormMode=NumEvents:!V')
        #  ~dataloader.PrepareTrainingAndTestTree(TCut('PuppiMET>0'),
                #  ~'nTrain_Regression=100000:nTest_Regression=1000000:SplitMode=Random:NormMode=NumEvents:!V')
        
         
        # Book methods
        #  ~factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras_simple'+MetBin,
                #  ~'H:!V:VarTransform=D,G:FilenameModel=model_simple.h5:NumEpochs=20:BatchSize=32')
        factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras'+MetBin,
                'H:!V:FilenameModel=model.h5:NumEpochs=20:BatchSize=64')
                #  ~'H:!V:FilenameModel=model.h5:VarTransform=N:NumEpochs=20:BatchSize=64')
                #  ~'H:!V:FilenameModel=model.h5:NumEpochs=100:BatchSize=512')
        #  ~factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras_2'+MetBin,
                #  ~'H:!V:FilenameModel=model_2.h5:NumEpochs=20:BatchSize=64')
        #  ~factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras_large'+MetBin,
                #  ~'H:!V:FilenameModel=model_large.h5:NumEpochs=20:BatchSize=64')
                #  ~'H:!V:FilenameModel=model_large.h5:NumEpochs=40:BatchSize=500')
                #  ~'H:!V:FilenameModel=model_large.h5:VarTransform=N:SaveBestOnly=false:NumEpochs=100:BatchSize=512')    #Overtraining example
                #  ~'H:!V:FilenameModel=model_large.h5:VarTransform=N:SaveBestOnly=false:NumEpochs=200:BatchSize=512')    #Overtraining example
        #  ~factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras_deep'+MetBin,
                #  ~'H:!V:FilenameModel=model_deep.h5:NumEpochs=20:BatchSize=64')
        #  ~factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras_autokeras'+MetBin,
                #  ~'H:!V:FilenameModel=model_autokeras.h5:NumEpochs=200:BatchSize=512')
        #  ~factory.BookMethod(dataloader,TMVA.Types.kBDT, "BDTG_AD",
                       #  ~"!H:!V:NTrees=64.BoostType=Grad:Shrinkage=0.3:nCuts=20:MaxDepth=4:"+"RegressionLossFunctionBDTG=AbsoluteDeviation")
     
        # Run TMVA
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()

def featureImportance(tree,importVars,cutDict):
    for MetBin in cutDict:
        # Feature importance    (requires training without TMVA)
        inputVars.append("genMET")
        inputArray = root_numpy.tree2array(tree,branches=inputVars,selection=cutDict[MetBin])
        y = inputArray["genMET"]
        x = remove_field_name(inputArray,"genMET")
        
        x_new = x.view(np.float32).reshape(x.shape + (-1,))
        x_new=x_new[:,:-1]
                
        #  ~train_x, val_x, train_y, val_y = train_test_split(x_new, y, random_state=1, test_size=40000, train_size=50000)
        train_x, val_x, train_y, val_y = train_test_split(x_new, y, random_state=1, test_size=10000, train_size=15000)
        #  ~train_x, val_x, train_y, val_y = train_test_split(x_new, y, random_state=1, test_size=10000, train_size=40000)
        #  ~train_x, val_x, train_y, val_y = train_test_split(x_new, y, random_state=1)
                
        #  ~my_model = KerasRegressor(build_fn=baseline_Model, epochs=20, batch_size=64, verbose=1)
        my_model = KerasRegressor(build_fn=baseline_Model, epochs=40, batch_size=64, verbose=1)
        #  ~my_model = KerasRegressor(build_fn=large_Model, epochs=200, batch_size=512, verbose=1)
        #  ~my_model = KerasRegressor(build_fn=deep_Model, epochs=20, batch_size=64, verbose=1)
        my_model.fit(train_x,train_y,validation_split = 0.8)
        
        #  ~perm = PermutationImportance(my_model, random_state=1).fit(val_x,val_y)
        #  ~output = eli5.format_as_text(eli5.explain_weights(perm, target_names = "genMET",feature_names = inputVars[:-1]))
        #  ~print MetBin
        #  ~print output
        
        y_hat_train = my_model.predict(train_x)
        y_hat_test = my_model.predict(val_x)

        # display error values
        print ('Train RMSE: ', round(np.sqrt(((train_y - y_hat_train)**2).mean()), 2))    
        print ('Train MEAN: ', round(((train_y - y_hat_train).mean()), 2))    
        print ('Test RMSE: ', round(np.sqrt(((val_y - y_hat_test)**2).mean()), 2))
        print ('Test MEAN: ', round(((val_y - y_hat_test).mean()), 2))
    
#############################################################

if __name__ == "__main__":
    # Load data
    data = TFile.Open("/net/data_cms1b/user/dmeuser/top_analysis/2016/v21/minTrees/TTbar_diLepton_100.0.root")
    tree = data.Get("ttbar_res100.0/TTbar_diLepton")
    
    #  ~hist=TH1F("test","",100,0,200)
    #  ~tree.Draw("genMET>>test","(PuppiMET>0 && PuppiMET<40)","goff")
    #  ~fit = hist.Fit("gaus","SQ")
    #  ~for i in range(0,100):
        #  ~hist.SetBinContent(i,hist.GetBinContent(i)/(fit.Parameter(0)*TMath.Gaus(hist.GetBinCenter(i),fit.Parameter(1),fit.Parameter(2))))
        #  ~print hist.GetBinCenter(i), hist.GetBinContent(i)
        
    # Define Input Variables
    inputVars = ["PuppiMET","METunc_Puppi","MET","HT","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E"]
    #  ~inputVars = ["DeepMET","PuppiMET","METunc_Puppi","MET","HT","nJets","n_Interactions","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep1_flavor","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Lep2_flavor","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E"]
    #  ~inputVars = ["PuppiMET","PuppiMET_phi","METunc_Puppi","MET","PFMET_phi","METunc_PF","CaloMET","CaloMET_phi","HT","MHT","nJets","n_Interactions","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep1_flavor","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Lep2_flavor","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet1_unc","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E","Jet2_unc"]
    #  ~inputVars = ["PuppiMET","PuppiMET_phi","METunc_Puppi","MET","PFMET_phi","METunc_PF","CaloMET","CaloMET_phi","HT","MHT","nJets","n_Interactions","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep1_flavor","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Lep2_flavor","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet1_bTagScore>0.2217","Jet1_unc","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E","Jet2_bTagScore>0.2217","Jet2_unc"]
    #  ~inputVars = ["PuppiMET","METunc_Puppi","MET","HT","nJets","n_Interactions","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_pt*sinh(Lep1_eta)","Lep1_E","Lep1_flavor","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_pt*sinh(Lep2_eta)","Lep2_E","Lep2_flavor","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_pt*sinh(Jet1_eta)","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_pt*sinh(Jet2_eta)","Jet2_E",]


    # Define Binning Scheme
    cutDict={"Bin1":"PuppiMET>0 && PuppiMET<40","Bin2":"PuppiMET>40 && PuppiMET<80","Bin3":"PuppiMET>80 && PuppiMET<120","Bin4":"PuppiMET>120 && PuppiMET<160","Bin5":"PuppiMET>160 && PuppiMET<230","Bin6":"PuppiMET>230"}
    #  ~cutDict={"Bin6":"PuppiMET>230"}
    #  ~cutDict={"Bin5":"PuppiMET>160 && PuppiMET<230"}
    #  ~cutDict={"Bin4":"PuppiMET>120 && PuppiMET<160"}
    #  ~cutDict={"Bin3":"PuppiMET>80 && PuppiMET<120"}
    #  ~cutDict={"Bin2":"PuppiMET>40 && PuppiMET<80"}
    #  ~cutDict={"Bin1":"PuppiMET>0 && PuppiMET<40"}
    
    trainTMVA(tree,inputVars,cutDict)
    
    #  ~featureImportance(tree,inputVars,cutDict)
        
    
