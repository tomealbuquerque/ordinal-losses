# Ordinal Losses for Classification of Cervical Cancer Risk

https://peerj.com/articles/cs-457/

by Tomé Albuquerque, Ricardo Cruz, Jaime S. Cardoso

## Usage

  1. Run preprocess.py to generate the folds.
  2. Run train.py to train the models you want.
  3. Run evaluate.py to generate results table.

## Code organization

  * **data:** avaiable at: http://mde-lab.aegean.gr/index.php/downloads
  * **train.py:** train the different models with the different ordinal losses
    and outputs probabilities.
  * **evaluate.py:** generate latex tables with results using the output
    probabilities.
