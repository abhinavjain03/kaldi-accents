# Accent Embeddings - HMM & Baseline
This repo is a part of
1. Current
2. [Accent Classifier](https://github.com/abhinavjain03/kaldi-accentrecognizer "AccentClassifier")
3. [Accent Embeddings - Multitask](https://github.com/abhinavjain03/kaldi-accentsmultitask "AE - MTL") 

and form the complete work mentioned in the paper submitted in Interspeech 2018. [Paper](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1864.html "IS1864").

Pre-Requisites - 
1. You have worked with the Kaldi toolkit and are quite familiar with it, meaning you are familiar with training a DNN Acoustic Model and know the requirements.
2. We use Mozilla CommonVoice Dataset for all the experiments. A detailed split can be found at - 
[Accents Unearthed](https://sites.google.com/view/accentsunearthed-dhvani/ "AccentsUnearthed")

## What we are doing?
To start with, we first train a classical GMM-HMM Speech Recognition model which will be further used to get the alignments for training a DNN Acoustic Model, which will act as our baseline. Also, we will extract Accent Embeddings using [Accent Classifier](https://github.com/abhinavjain03/kaldi-accentrecognizer "Accent Classifier") repository and use them as additional input to improve the performance. 
This is the script [my_run.sh](./my_run.sh).

## Data Prep
1. We adapted the scripts from the *swbd* recipe for lang generation as well as the classical HMM models.
2. All the HMM models and the DNN models are trained on Train7, validated on Dev4, and tested on Test4, TestNZ and TestIN.

**Note :** In the scripts,
1. Train7 - cv_train_nz, cv_trainx_nz
2. Dev4 - cv_dev_nz
3. Test4 - cv_test_nz
4. TestNZ - cv_test_onlynz
5. TestIN - cv_test_onlyindian

## Steps
Rest of the steps are pretty standard. We train a Speaker Adapted HMM Model *(tri4)* and will use the alignments generated from this model for further training in this as well as the multitask setup [Accent Embeddings - Multitask](https://github.com/abhinavjain03/kaldi-accentsmultitask "AE - MTL").
1. **MFCCS** - The script creates both standard and hires MFCCS of train, dev and test data provided the correct paths.
2. **ivectors** - ivectors are used for speaker adaptation while training DNN Acoustic Models. The scripts for training an ivector extractor comes with kaldi.
3. **alignments** - We use alignments generated from the tri4 model as target for training of the DNN Acoustic models.
4. Once all these are done, xconfig and training is standard.

**Note :** 
1. This repository is a pre-requisite for further work [Accent Embeddings - Multitask](https://github.com/abhinavjain03/kaldi-accentsmultitask "AE - MTL") mentioned in the paper.
2. [get_bnf_text.sh](./get_bnf_text.sh "get_bnf_text.sh") & [make_utterance_level_bnf.py](./make_utterance_level_bnf.py "make_utterance_level_bnf.py") are the scripts used for generating utterance level bnfs.
