from ROOT import TMVA, TFile, TTree, TCut, ObjectProxy
from subprocess import call
from os.path import isfile
import ROOT
 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
 
# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
 
output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVAMulticlass', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=multiclass')
 
# Load data
data = TFile.Open("/net/data_cms1b/user/dmeuser/top_analysis/output/ttbar_res100.0_new.root")
t_all = data.Get("ttbar_res100.0/ttbar_res_dilepton_CP5")
 
dataloader = TMVA.DataLoader('dataset')

#  ~for var in ["PuppiMET","METunc_Puppi","MET","HT","nJets","n_Interactions","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep1_flavor","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Lep2_flavor","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E"]:
for var in ["PuppiMET","METunc_Puppi","HT","nJets","n_Interactions","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep1_flavor","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Lep2_flavor","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E"]:
  dataloader.AddVariable(var)

dataloader.AddTree(t_all, 'Bin1')
dataloader.AddTree(t_all, 'Bin2')
dataloader.AddTree(t_all, 'Bin3')
dataloader.AddTree(t_all, 'Bin4')
dataloader.AddTree(t_all, 'Bin5')
dataloader.AddTree(t_all, 'Bin6')
dataloader.AddCut(TCut('genMET>0 && genMET<40'), 'Bin1')
dataloader.AddCut(TCut('genMET>40 && genMET<80'), 'Bin2')
dataloader.AddCut(TCut('genMET>80 && genMET<120'), 'Bin3')
dataloader.AddCut(TCut('genMET>120 && genMET<160'), 'Bin4')
dataloader.AddCut(TCut('genMET>160 && genMET<230'), 'Bin5')
dataloader.AddCut(TCut('genMET>230'), 'Bin6')
dataloader.PrepareTrainingAndTestTree(TCut('PuppiMET>0'),'nTrain_Bin1=10000:nTest_Bin1=70000:nTrain_Bin2=10000:nTest_Bin2=70000:nTrain_Bin3=10000:nTest_Bin3=70000:nTrain_Bin4=10000:nTest_Bin4=70000:nTrain_Bin5=10000:nTest_Bin5=70000:nTrain_Bin6=10000:nTest_Bin6=70000:SplitMode=Random:NormMode=NumEvents:!V')
 
# Generate model
 
# Define model
model = Sequential()
#  ~model.add(Dense(24, input_dim=24, activation='relu'))
model.add(Dense(23, input_dim=23, activation='relu'))
model.add(Dense(6, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Store model to file
model.save('model.h5')
model.summary()
 
# Book methods
#  ~factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTG",
        #  ~"!H:!V:NTrees=1000:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.50:nCuts=20:MaxDepth=2")
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, "PyKeras_multi",
        'H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=20:BatchSize=32')
 
# Run TMVA
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
