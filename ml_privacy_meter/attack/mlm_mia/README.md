

# **MLM Membership Inference Attack**

Repository for  the Quantifying Privacy Risks of Masked Language Models paper

This repository borrows the environment and the models from [https://github.com/elehman16/exposing_patient_data_release](https://github.com/elehman16/exposing_patient_data_release)

## Creating  Conda Environment

```markdown
conda env create -f conda_env.yml
```

## Acquiring Models and Datasets

Since both MIMIC-III and i2b2 datasets require an access process, we have not made them available online. 

### MIMIC-III

MIMIC-III data and model authorization process: [https://mimic.mit.edu/docs/gettingstarted/](https://mimic.mit.edu/docs/gettingstarted/)

You can download the models from this link once you have gone through the authorization process, login to physionet and then access this link: [https://physionet.org/files/clinical-bert-mimic-notes/1.0.0/](https://physionet.org/files/clinical-bert-mimic-notes/1.0.0/)

### i2b2

You can request access to i2b2 dataset here: [https://portal.dbmi.hms.harvard.edu/](https://portal.dbmi.hms.harvard.edu/)  

### Gaining access to the processed sequences used for experiments

Once you have gained access to MIMIC-III and i2b2, forward the access grant email to fmireshg@eng.ucsd.edu and you will have the processed data shared with you.

Once you have received the processed files, make a folder named ‘CSV_Files’ and place the csv files sent to you there.

## Running the MIA attack

To generate the loss values of the clinicalbert-base (target model) and pubmed (reference model) for each sequence, run the following from the root directory of the repo:

```markdown
bash scripts/get_loss_clinicalbert.sh
bash scripts/get_loss_pubmed.sh
```

If you want to get these numbers for other models, just replace the model_path in the bash scripts with the path/name of your model. Only caveat is that if your model is a huggingface model, you want to use the template from get_loss_pubmed.sh. If your model is another one of the clinicalbert models provided in the MIMIC-III files, use the get_loss_clinicalbert.sh template. The reason is  the clinicalbert model is trained using the pytorch_pretrained_bert, which is an older package compared to transformers HF, so some function calls are different. 

 If you want to use normalized energy values (not in the paper, does improve the results a bit) instead of loss, run the following:

```markdown
bash scripts/get_enorm_clinicalbert.sh
bash scripts/get_enorm_pubmed.sh
```

When you run these, the loss/energy scores will be saved in the loss_values/enorm_values folders. For reproduction purposes, we have already placed our outputs used in the paper there. To get the attack success metrics (AUC, precision, recall) for our method and the baseline  run the following: 

```markdown
bash scripts/metrics/get_sample_metrics.sh
bash scripts/metrics/get_sample_metrics_enorm.sh
```

These will return the results and metrics for sample-level attack. 

For user-level attack, run

```markdown
bash scripts/metrics/get_user_metrics.sh
```

## Re-producing the plots

The plots in the paper can be roprodueced by running the  ./ipynb/plots.ipynb notebook.
