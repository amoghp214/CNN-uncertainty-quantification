# CNN Uncertainty Quantification

This code trains a CNN and performs uncertainty quantification on it using the Dropout Ensemble Technique and the Rejection Confindence Variance.

#### Part 1 
To train the model, run:
`python train.py`

#### Part 2a
To perform uncertainty quantification using the Dropout Ensemble Technique, run:
`python perform_dropout_ensemble.py <model_path>`

#### Part 2b
To perform uncertainty quantification using  Rejection Confidence Variance, run:
`python perform_rcv.py <model_path>`

##### Note:
This repository does not include any model state files.




