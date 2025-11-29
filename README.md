
## Highlights of FOCUS

- **FOCUS** (A Foundational Generative Model for Cross-platform Unified Enhancement of Spatial Transcriptomics) 


## How to apply the work
### 1. Environment

- Python >= 3.10
- Use the following command to create your own environment.
```
conda env create -f DDPM310_full.yaml
```

### 2. Prepare data

You can get the data (about 1TB) from this link.

### 3. Train

You can use the below commands to train the model:
```
    python ./train.py 
```

### 4. Test

You can download the provided [pretrained model](https://www.dropbox.com/scl/fo/gte7fbz2y14syitka3mb0/AOhsVHx4Rdlk9BU2oJRySv4?rlkey=aytpdtg1ae05jf8i139a2e15c&st=5qi9943t&dl=0) Use the below command to test the model.
```
    python ./test.py
```

### 5. Other doc
You can find other model we have used here:

[BioBERT](https://github.com/dmis-lab/biobert)

[scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial)

[Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)






