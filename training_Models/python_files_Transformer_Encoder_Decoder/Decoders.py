from utils import *

# Decoder Using GRU
class DECODER_GRU(nn.Module):
    def __init__(self) -> None:
        super(DECODER_GRU, self).__init__()

        # LSTM
        self.lstm1 = nn.LSTM(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True) #100 to 364
        self.lstm2 = nn.LSTM(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True) #512 to 512

        # self.gru1 = nn.GRU(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True, dropout= drop_prob)
        # self.gru2 = nn.GRU(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True, dropout= drop_prob)
        # self.gru3 = nn.GRU(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True, dropout= drop_prob)


        # attention layer
        self.attention_Q = nn.Linear(D_To_Output_output_size*2, D_To_Output_output_size*2)
        self.attention_K = nn.Linear(D_To_Output_output_size*2, D_To_Output_output_size*2)
        self.attention_V = nn.Linear(D_To_Output_output_size*2, D_To_Output_output_size*2)

        # reverse embedder
        self.Linear_seq2 = nn.Sequential(
            nn.Linear(D_To_Output_output_size*2 ,Text_embedding_size)
        )

        # # initialising weights of linear layers with he_normal distribution
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight.data)
        #         nn.init.constant_(m.bias.data, 0)
        #     if isinstance(m, nn.GRU):
        #         for name, param in m.named_parameters():
        #             if 'weight' in name:
        #                 nn.init.xavier_normal_(param.data)
        #             else:
        #                 nn.init.constant_(param.data, 0)
        #     if isinstance(m, nn.LSTM):
        #         for name, param in m.named_parameters():
        #             if 'weight' in name:
        #                 nn.init.xavier_normal_(param.data)
        #             else:
        #                 nn.init.constant_(param.data, 0)

    def initialise_hidden_states(self, batch_size):
        self.hidden1 = (torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device),
                        torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device))
        self.hidden2 = (torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device),
                        torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device))

        # self.hidden1 = torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device)
        # self.hidden2 = torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device)
        # self.hidden3 = torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device)        

    def forward(self, x, Bool):


        if Bool:
            self.initialise_hidden_states(x.shape[0])

        l1, self.hidden1 = self.lstm1(x, self.hidden1)
        l1 = F.relu(l1)
        l2, self.hidden2 = self.lstm2(x, self.hidden2)
        l2 = F.relu(l2)
        
        x = torch.cat((l1, l2), dim=-1)

        # attention
        Q = self.attention_Q(x)
        K = self.attention_K(x)
        V = self.attention_V(x)
        x = torch.matmul(Q, K.transpose(-2, -1))/np.sqrt(D_To_Output_output_size)
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, V)

        # reverse embedder
        x = self.Linear_seq2(x)
        return x


# Decoder Using TransformerEncoderLayer
class Decoder_Trans(nn.Module):
    def __init__(self):
        super(Decoder_Trans, self).__init__()

        self.input_dim = E_TO_D_Input_size #150
        self.output_dim = E_TO_D_output_size #256

        self.num_layers = 7
        self.num_heads = 8
        self.hidden_dim = D_output_size #256

        self.input_2 = D_To_Output_Input_size
        self.output_2 = D_To_Output_output_size


        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias = False)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=512, dropout=drop_prob, activation='relu', batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_2, self.input_2),
            nn.ReLU(),
            nn.Linear(self.input_2, self.output_2)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def init_Mask(self, x):
        self.mask = torch.zeros(x.shape[0], x.shape[0]).to(device)
        for i in range(x.shape[0]):
            self.mask[i, :i] = 1

    def forward(self, ENC_output):
        self.init_Mask(ENC_output)
        ENC_output = self.embedding(ENC_output).squeeze(2)
        ENC_output = ENC_output.permute(1, 0, 2)
        ENC_output = F.layer_norm(ENC_output, (ENC_output.shape[1], ENC_output.shape[2]))
        ENC_output = self.transformer_encoder(ENC_output, mask = self.mask)
        ENC_output = self.fc(ENC_output)
        return ENC_output
    


# Decoder Using TransformerEncoderLayer and using bayesian Linear layers for the final output
class Decoder_Trans_New(nn.Module):
    def __init__(self):
        super(Decoder_Trans_New, self).__init__()

        self.input_dim = E_TO_D_Input_size #150
        self.output_dim = E_TO_D_output_size #256

        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = D_output_size #256

        self.input_2 = D_To_Output_Input_size
        self.output_2 = D_To_Output_output_size


        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias = False)
        
        self.transformerEncoderLayer1 = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=512, 
                                                                    dropout=drop_prob, 
                                                                    activation='gelu', 
                                                                    batch_first=True)
        
        self.transformerEncoderLayer2 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                                  nhead=self.num_heads, 
                                                                  dim_feedforward=512, 
                                                                  dropout=drop_prob, 
                                                                  activation='gelu', 
                                                                  batch_first=True)

        self.transformerEncoderLayer3 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                                  nhead=self.num_heads, 
                                                                  dim_feedforward=512, 
                                                                  dropout=drop_prob, 
                                                                  activation='gelu', 
                                                                  batch_first=True)
        
        self.transformerEncoder1 = nn.TransformerEncoder(self.transformerEncoderLayer1, num_layers=self.num_layers)
        self.transformerEncoder2 = nn.TransformerEncoder(self.transformerEncoderLayer2, num_layers=self.num_layers)
        self.transformerEncoder3 = nn.TransformerEncoder(self.transformerEncoderLayer3, num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.input_2*3, self.input_2*2 + self.output_2),
            nn.ReLU(),

            nn.Linear(self.input_2*2 + self.output_2, self.input_2 + self.output_2),
            nn.Tanh(),

            nn.Linear(self.input_2 + self.output_2, self.output_2)
        ) 

        # self.label_to_hidden = nn.Linear(Text_embedding_size, self.hidden_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, -10)
            if isinstance(m, nn.TransformerDecoderLayer):
                for name, param in m.named_parameters():
                    if 'weight' in name and param.dim() == 2:
                        nn.init.xavier_normal_(param.data)
                    else:
                        nn.init.constant_(param.data, -10)

    def init_Mask(self, batch_size):
        self.mask = torch.zeros(int(batch_size*self.num_heads), 100, 100).to(device)
        for i in range(100):
            self.mask[:, i, :i] = 1
        
    def forward(self, ENC_output):
        self.init_Mask(ENC_output.shape[0])
        ENC_output = self.embedding(ENC_output).squeeze(1)

        ENC_output = F.layer_norm(ENC_output, (ENC_output.shape[1], ENC_output.shape[2]))

        # RSLabels = self.label_to_hidden(RSLabels)

        ENC_output1 = self.transformerEncoder1(ENC_output)
        ENC_output2 = self.transformerEncoder2(ENC_output)
        ENC_output3 = self.transformerEncoder3(ENC_output)

        ENC_output = torch.cat((ENC_output1, ENC_output2, ENC_output3), dim = -1)

        ENC_output = self.fc(ENC_output)
        return ENC_output