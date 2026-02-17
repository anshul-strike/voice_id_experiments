# Voice Identification Experiments for IntelUpsell

## Overview
This repository's notebooks detail various experiments for voice identification of operators for IntelUpsell

## Notebook Overview
### Julian_George.ipynb
- This notebook leverages a test set of manually clipped transactions, each of which are manually labeled as Julian or Geroge
- It is possible to include Raw Ground Truth voice samples of just Julian and George or Raw Ground Truth voice samples of all DQ Fuquay Operators
- Before running the notebook, ensure to load Raw Ground Truth samples and Processed Transaction Clips into:
  - "Raw Ground Truths"
  - "Processed Transaction Clips"
- Ensure to empty the following folders:
  - "Processed Ground Truths"
  - "To Be Embedded Ground Truths"
  - "To Be Embedded Transactions"
  - "Transaction Embeddings"
  - "Ground Truth Embeddings"

### AssemblyAI.ipynb
- The experiment setup is similar in this notebook, transactions are still manually labeled as Julian or George but now they are clipped (into operator and customer intervals) using AssemblyAI API
- Similarly, it is possible to include Raw Ground Truth voice samples of just Julian and George or Raw Ground Truth voice samples of all DQ Fuquay Operators
- Before running the notebook, ensure to load Raw Ground Truth samples and Unclipped Processed Transactions into:
  - "Raw Ground Truths"
  - "Unclipped Processed Transactions"/"Julian Files"
  - "Unclipped Processed Transactions"/"George Files"
- Ensure to empty the following folders:
  - "Processed Ground Truths"
  - "Concat Transactions"
  - "To Be Embedded Ground Truths"
  - "To Be Embedded Transactions"
  - "Transaction Embeddings"
  - "Ground Truth Embeddings"

## Notebook Pipelines
### Julian_George.ipynb
- Load audio clips into specified folders and apply preprocessing steps (bandpass, AGC, limiter)
  - After preprocessing, unprocessed files in "Raw Ground Truths" will have respective processed files in "Processed Ground Truths"
- Prepare both processed ground truth voice samples and processed transaction clips for embedding
  - Files in "Processed Ground Truths" have respective files in "To Be Embedded Ground Truths"
  - Files in "Processed Transaction Clips" have respective files in "To Be Embedded Transactions"
- Embed ground truths and transactions into a 192-dim vector via TitaNet
  - Files in "To Be Embedded Ground Truths" have respective embeddings in "Ground Truth Embeddings"
  - Files in "To Be Embedded Transactions" have respective embeddings in "Transaction Embeddings"
- Create a data frame where each row represents a transaction embedding and each column represents a ground truth embedding. Calculate cosine similarity at each (row, column) combination
  - In each row, the cell with the highest cosine similarity value represents the identified operator

### AssembyAI.ipynb
- Load unclipped processed transactions for both Julian and George
- Collect transcripts with timestamps and operator/customer splits for each transaction for both Julian and George
- Collect intervals of operator speech for each transaction for both Julian and George
- Extract operator speech and concatatenate clips to form an audio file with only operator speech
  - This is a whole customer-operator transaction but only the operator portion. These files are saved in "Concat Transactions"
- Load audio clips into specified folders and apply preprocessing steps (bandpass, AGC, limiter)
  - After preprocessing, unprocessed files in "Raw Ground Truths" will have respective processed files in "Processed Ground Truths"
- Prepare both processed ground truth voice samples and processed transaction clips for embedding
  - Files in "Processed Ground Truths" have respective files in "To Be Embedded Ground Truths"
  - Files in "Concat Transactions" have respective files in "To Be Embedded Transactions"
- Embed ground truths and transactions into a 192-dim vector via TitaNet
  - Files in "To Be Embedded Ground Truths" have respective embeddings in "Ground Truth Embeddings"
  - Files in "To Be Embedded Transactions" have respective embeddings in "Transaction Embeddings"
- Create a data frame where each row represents a transaction embedding and each column represents a ground truth embedding. Calculate cosine similarity at each (row, column) combination
  - In each row, the cell with the highest cosine similarity value represents the identified operator


## Installation

```bash
conda create -n voice_id_env python=3.11.11
conda activate voice_id_env
pip install -r requirements.txt
```