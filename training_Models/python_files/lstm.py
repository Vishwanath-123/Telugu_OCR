from utils import *

class LSTM_NET(nn.Module):
    def __init__(self) -> None:
        super(LSTM_NET, self).__init__()

        # LSTM
        self.lstm1 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #512 to 512
        self.lstm2 = nn.LSTM(input_size=LSTM_Input_size, hidden_size=int(LSTM_output_size/2), num_layers=LSTM_num_layers, bidirectional = True, batch_first=True) #512 to 512

        # attention layer
        self.attention_Q = nn.Linear(LSTM_output_size*2, LSTM_output_size*2)
        self.attention_K = nn.Linear(LSTM_output_size*2, LSTM_output_size*2)
        self.attention_V = nn.Linear(LSTM_output_size*2, LSTM_output_size*2)

        # reverse embedder
        self.Linear_seq2 = nn.Sequential(
            nn.Linear(LSTM_output_size*2, LSTM_output_size),
        )

        # initialising the weights in Linear_seq2
        for m in self.Linear_seq2.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(10, 0.01)
                m.bias.data.zero_()

    def initialise_hidden_states(self, batch_size):
        self.hidden1 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
                        torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        self.hidden2 = (torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device),
                        torch.zeros(2*LSTM_num_layers, batch_size, int(LSTM_hidden_size/2)).to(device))
        

    def forward(self, x, Bool):


        if Bool:
            self.initialise_hidden_states(x.shape[0])

        l1, self.hidden1 = self.lstm1(x, self.hidden1)
        l1 = nn.ReLU()(l1)
        l2, self.hidden2 = self.lstm2(x, self.hidden2)
        l2 = nn.Tanh()(l2)

        x = torch.cat((l1, l2), dim=2)

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
