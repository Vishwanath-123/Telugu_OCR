from utils import *

# Decoder Using TransformerEncoderLayer
class Decoder_Trans_Encoder(nn.Module):
    def __init__(self):
        super(Decoder_Trans_Encoder, self).__init__()

        self.input_dim = E_TO_D_Input_size #150
        self.output_dim = E_TO_D_output_size #256

        self.num_layers = 6
        self.num_heads = 4
        self.hidden_dim = D_output_size #256

        self.input_2 = D_To_Output_Input_size
        self.output_2 = D_To_Output_output_size


        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias = False)
        
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=512, 
                                                                    dropout=drop_prob, 
                                                                    activation='gelu', 
                                                                    batch_first=True)
        
        self.transformerEncoder = nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.input_2, self.output_2),
        ) 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, -10)
            if isinstance(m, nn.TransformerEncoderLayer):
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

        ENC_output = self.transformerEncoder(ENC_output)

        ENC_output = self.fc(ENC_output)
        return ENC_output

# Decoder Using TransformerEncoderLayer 
class Decoder_Trans_Decoder(nn.Module):
    def __init__(self):
        super(Decoder_Trans_Decoder, self).__init__()

        self.input_dim = E_TO_D_Input_size #150
        self.output_dim = E_TO_D_output_size #256

        self.num_layers = 6
        self.num_heads = 4
        self.hidden_dim = D_output_size #256

        self.input_2 = D_To_Output_Input_size
        self.output_2 = D_To_Output_output_size


        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias = False)
        
        self.transformerDecoderLayer = nn.TransformerDecoderLayer(d_model=self.hidden_dim,
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=512, 
                                                                    dropout=drop_prob, 
                                                                    activation='gelu', 
                                                                    batch_first=True)
    
        
        self.transformerDecoder = nn.TransformerDecoder(self.transformerDecoderLayer, num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.input_2, self.output_2),
        ) 

        self.label_to_hidden = nn.Linear(Text_embedding_size, self.hidden_dim)

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
        
    def forward(self, ENC_output, RSLabels):
        self.init_Mask(ENC_output.shape[0])
        ENC_output = self.embedding(ENC_output).squeeze(1)

        ENC_output = F.layer_norm(ENC_output, (ENC_output.shape[1], ENC_output.shape[2]))

        RSLabels = self.label_to_hidden(RSLabels)

        ENC_output = self.transformerDecoder(ENC_output, RSLabels, tgt_mask = self.mask)

        ENC_output = self.fc(ENC_output)
        return ENC_output