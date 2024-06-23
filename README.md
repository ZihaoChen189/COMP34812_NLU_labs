# AV Group 52 

## DistilBERT
### Data Sources of the Coursework:
- Training Set: "train.csv"
- Validation Set: "dev.csv"
- Test Set: "test.csv"

### Fine-Funed Model Address
- The website address of the ".pth" file: https://drive.google.com/drive/folders/1G9CSCXn0FT2xiv9BBcowGl2Vgw8ACLpE?usp=sharing

## Bi-LSTM
### Data Sources of the Coursework:
- Training Set: "train.csv"
- Validation Set: "dev.csv"
- Test Set: "test.csv"

### Fine-Funed Model Address:
- The website address of the ".pth" file: https://drive.google.com/drive/folders/1G9CSCXn0FT2xiv9BBcowGl2Vgw8ACLpE?usp=sharing



## Code Sources of the Coursework:
- from transformers import DistilBertModel, DistilBertTokenizer
https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#distilbert

- from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

- class BaseDataset(Dataset): https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

- def worker_init_fn_seed(worker_id): https://pytorch.org/docs/stable/notes/randomness.html#dataloader

- criterion = nn.BCEWithLogitsLoss() https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

- optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam

- train_loader = DataLoader(train_set, batch_size=args.bs, num_workers=4, shuffle=True, pin_memory=True, worker_init_fn=worker_init_fn_seed) https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

- self.bn1 = nn.BatchNorm1d(hidden_size)
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d

- self.dropout = nn.Dropout(dropout)
https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout

- emb = F.gelu(emb) https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#torch.nn.GELU
