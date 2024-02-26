from utils import *

class DECODER(nn.Module):
    def __init__(self) -> None:
        super(DECODER, self).__init__()

        # LSTM
        # self.lstm1 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #100 to 364
        # self.lstm2 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #512 to 512

        # self.gru1 = nn.GRU(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True, dropout= drop_prob)
        # self.gru2 = nn.GRU(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True, dropout= drop_prob)
        # self.gru3 = nn.GRU(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True, dropout= drop_prob)


        self.lin_seq_1 = nn.Sequential(
            nn.Linear(LSTM_Input_size, LSTM_output_size),
            nn.ReLU(),
        )
        # transformer from LSTM_INPUT_SIZE to LSTM_OUTPUT_SIZE
        self.transformer1 = nn.TransformerDecoderLayer(d_model=LSTM_output_size, nhead=2, dim_feedforward=2048, dropout=drop_prob, activation='relu')
        self.transformer2 = nn.TransformerDecoderLayer(d_model=LSTM_output_size, nhead=2, dim_feedforward=2048, dropout=drop_prob, activation='relu')
        self.transformer3 = nn.TransformerDecoderLayer(d_model=LSTM_output_size, nhead=2, dim_feedforward=2048, dropout=drop_prob, activation='relu')
                                           

        # # attention layer
        # self.attention_Q = nn.Linear(LSTM_output_size*3, LSTM_output_size*3)
        # self.attention_K = nn.Linear(LSTM_output_size*3, LSTM_output_size*3)
        # self.attention_V = nn.Linear(LSTM_output_size*3, LSTM_output_size*3)

        # reverse embedder
        self.Linear_seq2 = nn.Sequential(
            nn.Linear(LSTM_output_size*3 ,Text_embedding_size)
        )

        # initialising weights of linear layers with he_normal distribution
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.Transformer):
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_normal_(p)


    # def initialise_hidden_states(self, batch_size):
    #     # self.hidden1 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
    #     #                 torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
    #     # self.hidden2 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
    #     #                 torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))

    #     self.hidden1 = torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device)
    #     self.hidden2 = torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device)
    #     self.hidden3 = torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device)
        

    def forward(self, x, Bool):

        # if Bool:
        #     self.initialise_hidden_states(x.shape[0])

        # l1, self.hidden1 = self.gru1(x, self.hidden1)
        # l1 = F.relu(l1)
        # l2, self.hidden2 = self.gru2(x, self.hidden2)
        # l2 = F.relu(l2)
        # l3, self.hidden3 = self.gru3(x, self.hidden2)
        # l3 = F.relu(l3)

        x = self.lin_seq_1(x)

        l1 = self.transformer1(x, x)
        l1 = F.tanh(l1)
        l2 = self.transformer2(x, x)
        l2 = F.tanh(l2)
        l3 = self.transformer3(x, x)
        l3 = F.tanh(l3)

        x = torch.cat((l1, l2, l3), dim=2)

        # # attention
        # Q = self.attention_Q(x)
        # K = self.attention_K(x)
        # V = self.attention_V(x)
        # x = torch.matmul(Q, K.transpose(-2, -1))/np.sqrt(LSTM_output_size)
        # x = F.softmax(x, dim=-1)
        # x = torch.matmul(x, V)

        # reverse embedder
        x = self.Linear_seq2(x)
        return x
        
# lstm = LSTM_NET().to(device)
# input = torch.randn(20, 1, 512).to(device)
# output = lstm(input, True)
# print(output.shape)
