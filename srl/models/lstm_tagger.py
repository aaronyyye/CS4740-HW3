import torch
import torch.nn as nn
import torch.nn.functional as F
from srl.utils.srl_utils import *
from typing import List

class LSTMTagger(nn.Module):
    def __init__(self, src_vocab, embed_dim,
    hidden_dim, output_dim, vocab_size, num_layers=1):
        '''
        :param src_vocab: vocabulary of inputs (Class Vocab)
        :param embed_dim: dimension of word embedding
        :param hidden_dim: dimension of hidden layer
        :param output_dim: dimension of tagset_size
        :param vocab_size: vocabulary size
        :param num_layers: number of LSTM layers 
        '''
        super().__init__()
        self.src_vocab = src_vocab
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        ### TODO 1: Initialize three linear layers:
              # 1. self.embedding (A word embedding layer)
              # 2. self.lstm (A LSTM layer)
              # 3. self.linear (An output linear layer)

    def compute_Loss(self, criterion, predicted_vector, gold_label):
        '''
        computes the loss of the model based on the ground truth and predicted vector

        :param criterion: loss criterion
        :param predicted_vector: model prediction
        :param gold_label: ground truth
        '''

        loss = 0
        for n in range(len(predicted_vector)): # batch size
            loss += criterion(predicted_vector[n], gold_label[n])	
        return loss

    def forward(self, source: List[List[str]], verb_indices:List[int]):
        ### GOALS : 
        ###     Write the forward function such that it processes sentences. 
        ###     Return output of the logsoftmax across all time steps
       
        # Pad input sentences and convert to word index 
        source_padded = self.src_vocab.to_input_tensor(source, device=get_device())
        batch_size = source_padded.shape[0]        
        time_steps = source_padded.shape[1]

        #TODO 2: Convert word index to embedding

        #TODO 3: Pass inputs to the lstm layer

        #TODO 4: Get hidden state of verb in the sentence

        #TODO 5: Iterate over the time dimension: 
        #       - Concatenate verb hidden state to the hidden layer output of every token
        #       - Predict SRL tag distribution with output layer and logsoftmax


    def load_model(self, load_path):
        """ Load the model from a file.
        :param load_path (str): path to model
        """
        saved_model = torch.load(load_path)
        self.load_state_dict(saved_model.state_dict())

    def save_model(self, save_path):
        """ Saves the model to a file.
        :param save_path (str): path to model
        """
        torch.save(self, save_path)
