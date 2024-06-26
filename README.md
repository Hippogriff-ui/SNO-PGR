# SNO-PGR
Prediction of protein S-nitrosylation sites.

## UbNiRF uses the following programming languages and versions:
* python 3.10
* python 3.6
* MATLAB 2016a


## Guiding principles:

The dataset folder contains S-nitrosylation site datasets.  
The code folder is the code implementation in the article.  
The "physiochemical_factors.xlsx" file contains 10 physical and chemical properties of the nitration or nitrosylation peptide.

feature extraction:  
   BE.py is the implementation of BE.  
   CKSAAP.py is the implementation of CKSAAP.  
   EAAC.py is the implementation of EAAC.  
   PWM.py is the implementation of PWM.  
   PFR.py is the implementation of PFR.  
   "PSSM" folder is the implementation of PSSM.
   
feature selection:  
   Null_importances.py represents the Null importances.
   pcc.py represents the PCC.

data augmentation:
  cwgan.py is the implementation of CWGAN.
  smote.py is the implementation of SMOTE.
  
classifier:  
   RF_Kfold_Test.py are the implementation of RF.  

The "chi2_window.py" file is the implementation of the chi-square test with graphing.
   
The "feature_combine.py" is the implementation of combination of 6 features.

