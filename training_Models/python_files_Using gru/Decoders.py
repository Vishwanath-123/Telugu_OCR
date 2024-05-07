from utils import *
from BayesianLayers import BayesianLinear

# Decoder Using GRU
class DECODER_RNN(nn.Module):
    def __init__(self) -> None:
        super(DECODER_RNN, self).__init__()

        #----------------------------------------------LSTM------------------------------------
        self.lstm1 = nn.LSTM(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True) #100 to 364
        self.lstm2 = nn.LSTM(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True) #512 to 512
        self.lstm3 = nn.LSTM(input_size=E_TO_D_Input_size, hidden_size=int(D_To_Output_output_size/2), num_layers=Layers, bidirectional = True, batch_first=True) #512 to 512

        self.concat_Linear = nn.Sequential(
            nn.Linear(D_To_Output_output_size*3, D_To_Output_output_size),
        )


        #-----------------------------------------------GRU------------------------------------
        # self.gru1 = nn.GRU(input_size=E_TO_D_Input_size, 
        #                    hidden_size=int(D_To_Output_output_size/2), 
        #                    num_layers=Layers, 
        #                    bidirectional = True, 
        #                    batch_first=True, 
        #                    dropout= drop_prob)
        
        # self.gru2 = nn.GRU(input_size=E_TO_D_Input_size, 
        #                    hidden_size=int(D_To_Output_output_size/2), 
        #                    num_layers=Layers, 
        #                    bidirectional = True, 
        #                    batch_first=True, 
        #                    dropout= drop_prob)
        
        # self.gru3 = nn.GRU(input_size=E_TO_D_Input_size, 
        #                    hidden_size=int(D_To_Output_output_size/2), 
        #                    num_layers=Layers, 
        #                    bidirectional = True, 
        #                    batch_first=True, 
        #                    dropout= drop_prob)

        # self.concat_Linear = nn.Sequential(
        #     nn.Linear(D_To_Output_output_size*3, D_To_Output_output_size*2),
        #     nn.ReLU(),
        #     nn.Linear(D_To_Output_output_size*2, D_To_Output_output_size),
        # )


        #------------------------------------------Transformer-----------------------------------
        # self.tranformerEncoderLayer = nn.TransformerEncoderLayer(d_model=D_To_Output_output_size,
        #                                                         nhead=4,
        #                                                         dim_feedforward=512,
        #                                                         dropout=0.1,
        #                                                         activation='relu',
        #                                                         batch_first=True)
        
        # self.transformerEncoder = nn.TransformerEncoder(self.tranformerEncoderLayer, num_layers=2)

        # reverse embedder
        self.Linear_seq2 = nn.Sequential(
            nn.Linear(D_To_Output_output_size ,Text_embedding_size)
            # BayesianLinear(D_To_Output_output_size, Text_embedding_size),
            # nn.ReLU(),
            # BayesianLinear(Text_embedding_size, Text_embedding_size),
        )

        # initialising weights with xavier normal.
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
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)
            if isinstance(m, nn.TransformerEncoderLayer):
                for name, param in m.named_parameters():
                    if 'weight' in name and param.data.dim() > 1:
                        nn.init.xavier_normal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)

    def initialise_hidden_states(self, batch_size):

        # --------------------------Hidden states for LSTM-----------------------------------------
        self.hidden1 = (torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device),
                        torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device))
        self.hidden2 = (torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device),
                        torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device))
        self.hidden3 = (torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device),
                        torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device))

        # # --------------------------Hidden states for GRU------------------------------------------
        # self.hidden1 = torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device)
        # self.hidden2 = torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device)
        # self.hidden3 = torch.zeros(2*Layers, batch_size, int(D_To_Output_output_size/2)).to(device)

    def forward(self, x):
        self.initialise_hidden_states(x.shape[0])

        #--------------------------------LSTM-------------------------
        l1 = []
        l2 = []
        l3 = []

        for i in range(x.shape[2]):
            if i%50 == 0:
                self.initialise_hidden_states(x.shape[0])
            temp, self.hidden1 = self.lstm1(x[:, :, i, :], self.hidden1)
            l1.append(temp)
            temp, self.hidden2 = self.lstm2(x[:, :, i, :], self.hidden2)
            l2.append(temp)
            temp, self.hidden3 = self.lstm3(x[:, :, i, :], self.hidden3)
            l3.append(temp)
        
        l1 = torch.stack(l1, dim=1).squeeze(2)
        l2 = torch.stack(l2, dim=1).squeeze(2)
        l3 = torch.stack(l3, dim=1).squeeze(2)

        x = torch.cat((l1, l2, l3), dim=2)
        x = self.concat_Linear(x)

        # #--------------------------------GRU-------------------------------
        # l1 = []
        # l2 = []
        # l3 = []
        # for i in range(x.shape[2]):
        #     if i%50 == 0:
        #         self.initialise_hidden_states(x.shape[0])

        #     temp, self.hidden1 = self.gru1(x[:, :, i, :], self.hidden1)
        #     temp = F.relu(temp)
        #     l1.append(temp)

        #     temp, self.hidden2 = self.gru2(x[:, :, i, :], self.hidden2)
        #     temp = F.relu(temp)
        #     l2.append(temp)

        #     temp, self.hidden3 = self.gru3(x[:, :, i, :], self.hidden3)
        #     temp = F.relu(temp)
        #     l3.append(temp)
        
        # l1 = torch.stack(l1, dim=1).squeeze(2)
        # l2 = torch.stack(l2, dim=1).squeeze(2)
        # l3 = torch.stack(l3, dim=1).squeeze(2)

        # x = torch.cat((l1, l2, l3), dim=2)
        # x = self.concat_Linear(x)
        
        
        #----------------------------Transformer-----------------------
        # x = self.transformerEncoder(l1)


        # reverse embedder
        x = self.Linear_seq2(x).permute(1, 0, 2)
        return x
