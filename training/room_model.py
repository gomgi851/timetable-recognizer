"""Small VGG/BiLSTM/CTC recognizer compatible with EasyOCR english_g2.

Architecture adapted from JaidedAI/EasyOCR easyocr/model/{vgg_model,modules}.py.
Upstream license: Apache-2.0. No lexicon or course/room lookup is used.
"""
import torch
from torch import nn

CHARACTERS = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True),
            nn.Conv2d(128,128,3,1,1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(128,256,3,1,1,bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1,bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)), nn.Conv2d(256,256,2,1,0), nn.ReLU(True))
    def forward(self, x):
        return self.ConvNet(x)

class RoomRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.FeatureExtraction = VGG()
        self.SequenceModeling = nn.Sequential(BidirectionalLSTM(256,256,256),BidirectionalLSTM(256,256,256))
        self.Prediction = nn.Linear(256,len(CHARACTERS)+1)
    def forward(self, x):
        visual = self.FeatureExtraction(x).mean(dim=2).transpose(1,2)
        return self.Prediction(self.SequenceModeling(visual).contiguous())

def load_model(path, device):
    model = RoomRecognizer()
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state)
    return model.to(device)

def decode(logits):
    # Fixed generic room-code alphabet; identical before/after fine-tuning.
    allowed = {0} | {i+1 for i,c in enumerate(CHARACTERS) if c.isascii() and (c.isalnum() or c=='-')}
    logits = logits.clone()
    for i in range(logits.shape[-1]):
        if i not in allowed: logits[:,:,i] = -1e9
    labels = logits.argmax(-1).cpu().tolist()
    result=[]
    for sequence in labels:
        last=-1; text=''
        for c in sequence:
            if c and c!=last: text+=CHARACTERS[c-1]
            last=c
        result.append(text.upper())
    return result
