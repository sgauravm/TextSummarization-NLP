import torch
#from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Adafactor
import torch
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tqdm.auto import tqdm


PATH = 'data/news_summary.csv'
TORCH_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'google/pegasus-xsum'

print(TORCH_DEVICE)

# def data_preprcessing(path):
#     data = pd.read_csv(path, encoding='latin-1')
#     data = data.dropna()
#     X = list(data.ctext)
#     y = list(data.text)
#     return train_test_split(X, y, test_size=0.2, random_state=42)
#
#
# def tokenize(text_data):
#     tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
#     return tokenizer.prepare_seq2seq_batch(text_data, truncation=True, padding='longest', return_tensors="pt")
#
#
# class NewsSummaryDataset(torch.utils.data.Dataset):
#     def __init__(self, x_train, y_train):
#         self.input_encodings = tokenize(x_train)
#         self.target_encodings = tokenize(y_train)
#
#     def __getitem__(self, idx):
#         data = dict()
#         data["encoder_input"] = self.input_encodings["input_ids"][idx]
#         data["encoder_attention_masks"] = self.input_encodings["attention_mask"][idx]
#         data["decoder_output"] = self.target_encodings["input_ids"][idx]
#         decoder_input = self.target_encodings.input_ids[idx]
#         decoder_input = torch.roll(decoder_input, 1, -1)
#         decoder_input[0] = torch.tensor(0)
#         data["decoder_input"] = decoder_input
#         return data
#
#     def __len__(self):
#         return self.target_encodings.input_ids.shape[0]
#
#
# x_train, x_test, y_train, y_test = data_preprcessing(PATH)
# train_dataset = NewsSummaryDataset(x_train[0:6],y_train[0:6])
# INPUT_BATCH_SIZE=3
# train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=INPUT_BATCH_SIZE)
#
# model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(TORCH_DEVICE)
# model.train()
#
# optim = Adafactor(model.parameters())
#
# EPOCH = 2
# for i in range(EPOCH):
#     for batch in train_loader:
#         optim.zero_grad()
#         encoder_input = batch["encoder_input"].to(TORCH_DEVICE)
#         encoder_attention_mask = batch["encoder_attention_masks"].to(TORCH_DEVICE)
#         decoder_input = batch["decoder_input"].to(TORCH_DEVICE)
#         decoder_output = batch["decoder_output"].to(TORCH_DEVICE)
#         output = model(input_ids=encoder_input,
#                      attention_mask=encoder_attention_mask,
#                      decoder_input_ids=decoder_input,
#                     labels=decoder_output)
#         loss = output[0]
#         loss.backward()
#         optim.step()
#         print(loss)
# model.eval()
