from utils import *

class LSTM_NET(nn.Module):
    def __init__(self) -> None:
        super(LSTM_NET, self).__init__()

        # embedder
        self.Linear_seq = nn.Sequential(
            nn.Linear(Joiner_Input_size, Joiner_output_size),
            nn.ReLU(),
        )

        # LSTM
        self.lstm1 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #512 to 512
        self.lstm2 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #512 to 512
        self.lstm3 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True,batch_first=True) #512 to 512
        # self.lstm4 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True,batch_first=True) #512 to 512
        # self.lstm5 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True,batch_first=True) #512 to 512

        # attention layer
        self.attention_Q = nn.Linear(LSTM_output_size, LSTM_output_size)
        self.attention_K = nn.Linear(LSTM_output_size, LSTM_output_size)
        self.attention_V = nn.Linear(LSTM_output_size, LSTM_output_size)

        # reverse embedder
        self.Linear_seq2 = nn.Sequential(
            nn.Linear(Reverse_Input_size, Reverse_output_size),
            nn.LeakyReLU(negative_slope=2),
        )

        # initialising the weights in Linear_seq2
        for m in self.Linear_seq2.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(100, 0.02)
                m.bias.data.zero_()

    def initialise_hidden_states(self, batch_size):
        self.hidden1 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
                        torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        self.hidden2 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
                        torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        self.hidden3 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
                        torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        # self.hidden4 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
                        # torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        # self.hidden5 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
                        # torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        

    def forward(self, x, Bool):
        # embedder
        x = self.Linear_seq(x)

        if Bool:
            self.initialise_hidden_states(x.shape[0])

        x, self.hidden1 = self.lstm1(x, self.hidden1)
        x, self.hidden2 = self.lstm2(x, self.hidden2)
        x, self.hidden3 = self.lstm3(x, self.hidden3)
        # x, self.hidden4 = self.lstm4(x, self.hidden4)
        # x, self.hidden5 = self.lstm5(x, self.hidden5)

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
        
# lstm = LSTM_NET().to(device)
# input = torch.randn(20, 1, 512).to(device)
# output = lstm(input, True)
# print(output.shape)
