---
Authorship Verification
The website address of ".pth" files: https://drive.google.com/drive/folders/1G9CSCXn0FT2xiv9BBcowGl2Vgw8ACLpE?usp=sharing
---

# Model Card for a65914zc-j29115zl-AV
Bi-LSTM with three layers was trained to conduct binary classification experiments, detecting whether two pieces of sequences were written by the same author.

## Model Details

### Model Description
Bi-LSTM with three layers was employed to capture potential language representations. Besides, the word embedding of "distilbert-base-uncased" was downloaded from Hugging Face and fine-tuned. The prediction head was designed with the fully connected feed-forward network, batch normalization layer, and gelu activation layer, to calculate the cosine similarity of two input sequences. The dimension of hidden states decreased from 768 to 256 in the prediction head. Adam optimizer was used to decrease cross-entropy losses.

- **Developed by:** Zihao Chen and Zhimeng Liu
- **Language(s):** English
- **Model type:** Supervised with Fine-tuning
- **Model architecture:** Bi-LSTM
- **Finetuned from model:** Word Embedding from "distilbert-base-uncased"

### Model Resources
- **Repository:** https://huggingface.co/distilbert/distilbert-base-uncased and https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm

## Training Details

### Training Data
Bi-LSTM was trained (fine-tuned) based on 30k pairs of the text in **"train.csv"** and **only** considered pairs with more than 4 characters for each sequence.

### Training Procedure
1. After the sequence was tokenized and converted into **index** values, the "blank nosing" was introduced as the regularization technique to prevent overfitting, as **data augmentation**, following instructions from the Stanford University research [1].

2. Only considering 256 tokens in sequences after tokenizing, giving the average length of sequence was about 100 words for both "text_1" and "text_2".

3. DistilBERT tokenizer automatically converted words in the text to lowercase.

#### Training Hyperparameters
      - random seed: 68
      - train batch size: 16
      - eval batch size: 16 
      - learning rate: 1e-5
      - weight decay: 1e-5
      - epoch: 10
      - max token length: 256

#### Speeds, Sizes, Times
      - Overall Training Time: 76min 31s
      - Average Training Time per Epoch: 7min 39s
      - Model Size: 252 MB

## Evaluation

### Testing Data & Metrics

#### Testing Data
After training on **"train.csv"** for one epoch, **"dev.csv"** with 6k pairs of the text, viewed as the validation set, was used to evaluate the model performance. Following each epoch, the corresponding model was saved as a ".pth" file. Subsequently, the model exhibiting the **highest accuracy** on the validation set was retained. This retained one was then imported from Google Drive to predict results for **"test.csv"**.

#### Metrics
      - Accuracy
      - Precision
      - Recall
      - F1-Score

### Results
The model obtained 69.94% for accuracy, 78.25% for precision, 55.64% for recall, and 65.04% for f1-Score on the validation set.

## Technical Specifications

### Hardware
		- GPU: V100 16GB on Google Colab Pro
	  - System RAM: at least 2.9GB
	  - GPU RAM: at least 4.9GB
	  - Disk Storage: at least 27.3GB

### Software
      - transformers 4.40.0
      - torch 2.2.1
      - sklearn 1.2.2
      (From Colab)

## Bias, Risks, and Limitations
1. A portion of extremely short sequence pairs were discarded as invalid data for both in training and validation sets, though they might still contain semantic information, which probably influenced the bias of validation results.
2. Future works could be allowed to facilitate several ".py" scripts, employing bash commands in the terminal for the detailed hyperparameter tuning and convenient for multi-GPU environment training.

## Additional Information
Both training and demo code files leveraged "argparse" to construct the whole experiment, accepting various hyperparameters in main(), ensuring **strong** reproducibility and reusability. Additionally, these hyperparameters and some file locations in there could be flexibly adjusted. 

For example, changing default model type in parser.add_argument("--model_type", type=str, choices=['distilbert', 'lstm'], default='distilbert') to change the experiment model.



**References**:

[1] Ziang Xie, Sida I Wang, Jiwei Li, Daniel Lévy, Aiming Nie, Dan Jurafsky, and Andrew Y Ng. Data noising as smoothing in neural network language models. arXiv preprint arXiv:1703.02573, 2017.
