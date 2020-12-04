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
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Classification')
 
# Load data
data = TFile.Open("/net/data_cms1b/user/dmeuser/top_analysis/output/ttbar_res100.0_new.root")
#  ~signal = data.Get("ttbar_res100.0/ttbar_res_dilepton_CP5")
#  ~background = data.Get("ttbar_res100.0/ttbar_res_dilepton_CP5")
t_all = ROOT.TTree();
data.GetObject("ttbar_res100.0/ttbar_res_dilepton_CP5", t_all);
 
dataloader = TMVA.DataLoader('dataset')
#  ~for branch in signal.GetListOfBranches():
    #  ~dataloader.AddVariable(branch.GetName())

#  ~for var in ["n_Interactions","nJets", "HT","dPhiMETbJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiLep1bJet","Jet1_pt","Jet2_pt","ratioMET_sqrtMETunc_Puppi","dPhiLep1Lep2","dPhiLep1Jet1","ratio_pTj1_vecsum_pT_l1_l2_bjet"]:
  #  ~dataloader.AddVariable(var)
for var in ["nJets","n_Interactions","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep1_flavor","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Lep2_flavor","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet1_bTagScore","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E","Jet2_bTagScore"]:
  dataloader.AddVariable(var)
 
#  ~dataloader.AddSignalTree(signal, 1.0)
#  ~dataloader.AddBackgroundTree(background, 1.0)
met6_sig = ROOT.TCut("PuppiMET>230 && absmetres_PUPPI<40")
met6_bg = ROOT.TCut("PuppiMET>230 && absmetres_PUPPI>40")
dataloader.SetInputTrees(t_all,met6_sig,met6_bg)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'nTrain_Signal=10000:nTrain_Background=10000:SplitMode=Random:NormMode=NumEvents:!V')
 
# Generate model
 
# Define model
model = Sequential()
model.add(Dense(64, activation='relu', W_regularizer=l2(1e-5), input_dim=22))
model.add(Dense(2, activation='softmax'))
#  ~model.add(Dense(226, activation='sigmoid', W_regularizer=l2(1e-5), input_dim=13))
#  ~model.add(Dense(2, activation='softmax'))
 
# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01), metrics=['accuracy', ])
 
# Store model to file
model.save('model.h5')
model.summary()
 
# Book methods
factory.BookMethod(dataloader, TMVA.Types.kFisher, 'Fisher',
                   '!H:!V:Fisher:VarTransform=D,G')
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   'H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=20:BatchSize=32')
#  ~factory.BookMethod(dataloader,TMVA.Types.kBDT, "BDT",
                      #  ~"!V:NTrees=200:MinNodeSize=2%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=-1" );
#  ~factory.BookMethod(dataloader,TMVA.Types.kBDT, "BDT",
                      #  ~"!V:NTrees=200:MinNodeSize=2%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=-1" );
 
# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
