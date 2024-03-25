from utils import *

# Decoder Using GRU
class DECODER_GRU(nn.Module):
    def __init__(self) -> None:
        super(DECODER_GRU, self).__init__()

        # LSTM
        # self.lstm1 = nn.LSTM(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True) #100 to 364
        # self.lstm2 = nn.LSTM(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True) #512 to 512

        self.gru1 = nn.GRU(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True, dropout= drop_prob)


        self.tranformerEncoderLayer = nn.TransformerEncoderLayer(d_model=D_To_Output_output_size,
                                                                nhead=4,
                                                                dim_feedforward=512,
                                                                dropout=0.1,
                                                                activation='relu',
                                                                batch_first=True)
        
        self.transformerEncoder = nn.TransformerEncoder(self.tranformerEncoderLayer, num_layers=2)

        # reverse embedder
        self.Linear_seq2 = nn.Sequential(
            nn.Linear(D_To_Output_output_size ,Text_embedding_size)
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
        # self.hidden1 = (torch.zeros(2*Layers, batch_size, D_To_Output_output_size).to(device),
        #                 torch.zeros(2*Layers, batch_size, D_To_Output_output_size).to(device))
        # self.hidden2 = (torch.zeros(2*Layers, batch_size, D_To_Output_output_size).to(device),
        #                 torch.zeros(2*Layers, batch_size, D_To_Output_output_size).to(device))

        self.hidden1 = torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device)

    def forward(self, x):


        self.initialise_hidden_states(x.shape[0])

        l1 = []
        for i in range(x.shape[2]):
            temp, self.hidden1 = self.gru1(x[:, :, i, :], self.hidden1)
            temp = F.relu(temp)
            l1.append(temp)
        
        l1 = torch.stack(l1, dim=1).squeeze(2)
        # transformer
        x = self.transformerEncoder(l1)

        # reverse embedder
        x = self.Linear_seq2(x).permute(1, 0, 2)

        return x
