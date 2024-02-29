from utils import *

# Decoder Using GRU
class DECODER_GRU(nn.Module):
    def __init__(self) -> None:
        super(DECODER_GRU, self).__init__()

        # LSTM
        # self.lstm1 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #100 to 364
        # self.lstm2 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #512 to 512

        self.gru1 = nn.GRU(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True, dropout= drop_prob)
        self.gru2 = nn.GRU(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True, dropout= drop_prob)
        self.gru3 = nn.GRU(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True, dropout= drop_prob)


        # attention layer
        self.attention_Q = nn.Linear(LSTM_output_size*3, LSTM_output_size*3)
        self.attention_K = nn.Linear(LSTM_output_size*3, LSTM_output_size*3)
        self.attention_V = nn.Linear(LSTM_output_size*3, LSTM_output_size*3)

        # reverse embedder
        self.Linear_seq2 = nn.Sequential(
            nn.Linear(LSTM_output_size ,Text_embedding_size)
        )

        # initialising weights of linear layers with he_normal distribution
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)

    def initialise_hidden_states(self, batch_size):
        # self.hidden1 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
        #                 torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        # self.hidden2 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
        #                 torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))

        self.hidden1 = torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device)
        self.hidden2 = torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device)
        self.hidden3 = torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device)        

    def forward(self, x, Bool=True):


        if Bool:
            self.initialise_hidden_states(x.shape[0])

        l1, self.hidden1 = self.gru1(x, self.hidden1)
        l1 = F.relu(l1)
        l2, self.hidden2 = self.gru2(x, self.hidden2)
        l2 = F.relu(l2)
        l3, self.hidden3 = self.gru3(x, self.hidden2)
        l3 = F.relu(l3)
            
        # attention
        Q = self.attention_Q(x)
        K = self.attention_K(x)
        V = self.attention_V(x)
        x = torch.matmul(Q, K.transpose(-2, -1))/np.sqrt(LSTM_output_size)
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, V)

        # reverse embedder
        x = self.Linear_seq2(x)
        return x


# Decoder Using TransformerDecoderLayer
class Decoder_Trans(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, num_heads=2, hidden_dim=256, dropout=0.1):
        super(Decoder_Trans, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self._generate_positional_encoding(hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_memory(self, x):
        self.memory = torch.zeros_like(x).to(device)        


    def forward(self, x, bool):
        batch_size, seq_len, _ = x.size()
        
        # Apply embedding layer and positional encoding
        x = self.embedding(x) + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Permute dimensions for transformer input
        x = x.permute(1, 0, 2)
        if bool:
            self.init_memory(x)
        x = F.layer_norm(x, x.size()[1:])
        # Transformer decoder
        output = self.transformer_decoder(x, x)
        
        # Permute dimensions back
        output = output.permute(1, 0, 2)
        
        # Linear layer to get output dimension
        output = self.fc(output)

        # print("x: ", torch.max(x), " | ", torch.min(x))
        # print("output: ", torch.max(output), " | ", torch.min(output))
        # print("memory: ", torch.max(self.memory), " | ", torch.min(self.memory))
        # print(" \n\n")
        
        return output

    def _generate_positional_encoding(self, hidden_dim, max_len=5000):
        # Generate positional encodings up to max_len
        positional_encoding = torch.zeros(max_len, hidden_dim).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding
    
# Decoder = Decoder_Trans(input_dim=150, output_dim=346, num_layers=2, num_heads=2, hidden_dim=256, dropout=0.1).to(device)
# input = torch.rand(64, 100, 150).to(device)
# output = Decoder(input, True)
# print(output.shape)
