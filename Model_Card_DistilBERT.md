---
Authorship Verification
The website address of ".pth" files: https://drive.google.com/drive/folders/1G9CSCXn0FT2xiv9BBcowGl2Vgw8ACLpE?usp=sharing
---

# Model Card for a65914zc-j29115zl-AV
DistilBERT as Pretrained Language Model (PLM) was fine-tuned to conduct binary classification experiments, detecting whether two pieces of sequences were written by the same author.

## Model Details

### Model Description
PLM DistilBERT was downloaded from Hugging Face. The prediction head was designed with the dropout layer, fully connected feed-forward network, batch normalization layer, and gelu activation layer, to calculate the cosine similarity of two input sequences. The dimension of hidden states decreased from 768 to 256 in the prediction head. Adam optimizer was used to decrease cross-entropy losses.

- **Developed by:** Zihao Chen and Zhimeng Liu
- **Language(s):** English
- **Model type:** Supervised with Fine-tuning
- **Model architecture:** Encoder of Transformer
- **Finetuned from model:** distilbert-base-uncased

### Model Resources
- **Repository:** https://huggingface.co/distilbert/distilbert-base-uncased
- **Paper or documentation:** https://arxiv.org/abs/1910.01108 and https://arxiv.org/abs/1503.02531

## Training Details

### Training Data
DistilBERT was fine-tuned based on 30k pairs of the text in **"train.csv"** and **only** considered pairs with more than 4 characters for each sequence.

### Training Procedure
1. After the sequence was tokenized and converted into **index** values, the "blank nosing" was introduced as the regularization technique to prevent overfitting, as **data augmentation**, following instructions from the Stanford University research [1].

2. DistilBERT tokenizer automatically converted words in the text to lowercase.

3. Only considering 256 tokens in sequences after tokenizing, giving the average length of sequence was about 100 words for both "text_1" and "text_2".

4. Parameters of the word embedding and the first three encoders of DistilBERT were frozen without updates during fine-tuning, to prevent overfitting.

#### Training Hyperparameters
      - random seed: 68
      - train batch size: 16
      - eval batch size: 16 
      - learning rate: 1e-5
      - weight decay: 1e-5
      - dropout: 0.1
      - epoch: 10
      - max token length: 256

#### Speeds, Sizes, Times
      - Overall Training Time: 47min 2s
      - Average Training Time per Epoch: 4min 42s
      - Model Size: 268.7 MB

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
The model obtained 76.32% for accuracy, 83.52% for precision, 65.87% for recall, and 73.65% for f1-Score on the validation set.

## Technical Specifications

### Hardware
		- GPU: V100 16GB on Google Colab Pro
	  - System RAM: at least 2.7GB
	  - GPU RAM: at least 3.0GB
	  - Disk Storage: at least 27.1GB

### Software
      - transformers 4.40.0
      - torch 2.2.1
      - sklearn 1.2.2
      (From Colab)

## Bias, Risks, and Limitations
1. DistilBERT utilized knowledge distillation, reducing memory requirements with faster deployment speeds. Despite employing a range of techniques such as data augmentation and the cosine similarity, DistilBERT still demonstrated certain limitations in this assignment. 
2. A portion of extremely short sequence pairs were discarded as invalid data for both in training and validation sets, though they might still contain semantic information, which probably influenced the bias of validation results.
3. Future works could be allowed to facilitate several ".py" scripts, employing bash commands in the terminal for the detailed hyperparameter tuning and convenient for multi-GPU environment training.

## Additional Information
Both training and demo code files leveraged "argparse" to construct the whole experiment, accepting various hyperparameters in main(), ensuring **strong** reproducibility and reusability. Additionally, these hyperparameters and some file locations in there could be flexibly adjusted. 

For example, changing default model type in parser.add_argument("--model_type", type=str, choices=['distilbert', 'lstm'], default='distilbert') to change the experiment model.



**References**:

[1] Ziang Xie, Sida I Wang, Jiwei Li, Daniel Lévy, Aiming Nie, Dan Jurafsky, and Andrew Y Ng. Data noising as smoothing in neural network language models. arXiv preprint arXiv:1703.02573, 2017.
