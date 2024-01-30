from utils import *

# CNN model
class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()

        self.conv_seq = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout2d(drop_prob),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout2d(drop_prob),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout2d(drop_prob),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),


            # # 64 x 15 x 198
            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 194
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),

            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 190
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),
            
            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 186
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),

            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 182
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),

            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 178
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),

            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 174
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),

            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 170
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),

            # nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 64 x 15 x 166
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),

            # nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 162
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 158
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),
            
            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 154
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 150
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 146
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 142\
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 138
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 134
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 130
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 126
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 122
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0)), # 128 x 15 x 118
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(0, 0)), # 128 x 13 x 114
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(0, 0)), # 128 x 11 x 110
            # nn.Dropout2d(drop_prob),
            # nn.Tanh(),

            # nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(0, 0)), # 128 x 9 x 106
            # nn.Dropout2d(drop_prob),
            # nn.ReLU(),

            # nn.Conv2d(128, 128, kernel_size=(3, 7), stride=(1, 1), padding=(0, 0)), #128 x 7 x 100
            # nn.Dropout2d(drop_prob),
            # nn.Sigmoid(),
        )

        # initializing weights of conv layers with he_normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

        self.Linear_seq = nn.Sequential(
            nn.Linear(32*5, 100),
            nn.SELU(),
            nn.Linear(100, Image_embedding_size),
        )

    def forward(self, x):
        x = self.conv_seq(x)
        x = x.reshape(x.shape[0],x.shape[3], -1)
        x = self.Linear_seq(x)
        return x
    
# cnn = EncoderCNN()
# input = torch.randn(20, 1, 40, 800)
# output = cnn(input)
# print(output.shape)