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
    
## Citation
If you find this work useful for your research, please cite our paper:
```
@article{albuquerque2021ordinal,
 title = {Ordinal losses for classification of cervical cancer risk},
 author = {Albuquerque, Tomé and Cruz, Ricardo and Cardoso, Jaime S.},
 year = 2021,
 month = 4,
 keywords = {Cervical cytology, Convolutional Neural networks, Deep learning, Ordinal classification, Pap smear},
 volume = 7,
 pages = {e457},
 journal = {PeerJ Computer Science},
 issn = {2376-5992},
 url = {https://doi.org/10.7717/peerj-cs.457},
 doi = {10.7717/peerj-cs.457}
}
```

If you have any questions about our work, please do not hesitate to contact tome.albuquerque@gmail.com
