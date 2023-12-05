# Named Entity Recognition in De-identification context
- [Named Entity Recognition System](https://www.scitepress.org/AffiliationsProfile.aspx?Org=GF8fm9Pz1/t4nCksDBuZSAev55k9kp2r6cj9ox63seJ01X5Vb84wVmxj5/W4g5/p2hmNPfBolRCcLXCUAeMgHA==&t=1)


## Reference
Please cite the following paper:
```
    @conference{healthinf23,
    author={Yakini Tchouka. and Jean{-}Fran\c{c}ois Couchot. and David Laiymani.},
    title={An Easy-to-Use and Robust Approach for the Differentially Private De-Identification of Clinical Textual Documents},
    booktitle={Proceedings of the 16th International Joint Conference on Biomedical Engineering Systems and Technologies},
    year={2023},
    pages={94-104},
    publisher={SciTePress},
    organization={INSTICC},
    doi={10.5220/0011646600003414},
    isbn={978-989-758-631-6},
    issn={2184-4305},
    }
```


## Requirements
* Python >= 3.6 (recommended via anaconda)
* Install the required Python packages with `pip install -r requirements.txt`
* If the specific versions could not be found in your distribution, you could simple remove the version constraint. Our code should work with most versions.

## Dataset
Obviously for privacy reasons, we are not allowed to share the dataset used in this work. For execution you have to put your data in `data` folder

We assume that the dataset in conll format
e.g: paris LOC

## Architectures
It's an implementation of BERT-like based architecture.
This code is based on the python library NERDA :
```
    @inproceedings{nerda,
    title = {NERDA},
    author = {Kjeldgaard, Lars and Nielsen, Lukas},
    year = {2021},
    publisher = {{GitHub}},
    url = {https://github.com/ebanalyse/NERDA}
    }

```
## How to run

### Model Training
1. Provide a training and evaluation text file in conll format : `ner-train.txt` and `ner-eval.txt`
2. Run the following command to train the model 

```
    python Train.py --train_file ner-train.txt --eval_file ner-eval.txt --transformer flaubert/flaubert_base_uncased
```

### Notes
- You can change the default hyper-parameters of the models in `Train.py` 
