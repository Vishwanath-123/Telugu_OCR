from utils import *

# CNN model
class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()

        self.conv_seq11 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.conv_seq12 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.conv_seq21 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        self.conv_seq22 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        self.conv_seq31 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        
        self.conv_seq32 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        # initializing weights of conv layers with he_normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

        self.Linear_seq = nn.Sequential(
            nn.Linear(128*5, 320),
            nn.ReLU(),
            nn.Linear(320, Image_embedding_size),
        )

        #initializing weights of linear layers with Xavier normal
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.conv_seq11(x) + self.conv_seq12(x)
        x = self.conv_seq21(x) + self.conv_seq22(x)
        x = self.conv_seq31(x) + self.conv_seq32(x)

        # x = self.conv_seq_one_layer(x)

        x = x.reshape(x.shape[0],x.shape[3], -1)
        x = self.Linear_seq(x)
        return x
    
# cnn = EncoderCNN()
# input = torch.randn(20, 1, 40, 800)
# output = cnn(input)
# print(output.shape)