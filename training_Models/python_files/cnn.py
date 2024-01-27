from utils import *

# CNN model
class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()

        self.conv_seq = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.AvgPool2d(kernel_size=(2, 4), stride=(2, 4), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.Linear_seq = nn.Sequential(
            nn.Linear(128*7, Image_embedding_size)
        )

    def forward(self, x):
        x = self.conv_seq(x)
        x = x.reshape(x.shape[0],x.shape[3], -1)
        x = self.Linear_seq(x)
        return x
    
# cnn = EncoderCNN()
# input = torch.randn(20, 1, 30, 800)
# output = cnn(input)
# print(output.shape)

