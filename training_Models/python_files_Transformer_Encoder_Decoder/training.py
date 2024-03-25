from Decoders import Decoder_Trans_New
from cnn import CRNN_ENCODER
from dataset import TeluguOCRDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Epoch_run import *

Encoder = CRNN_ENCODER().to(device)
Decoder = Decoder_Trans_New().to(device)

# loss function and optimizer
torch.autograd.set_detect_anomaly(True)
criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean') if torch.cuda.is_available() else nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean')

params = list(Encoder.parameters()) + list(Decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-9)
# SGD optimizer
# optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-7, nesterov=True)

clip = 5
torch.nn.utils.clip_grad_norm_(params, clip)

num_of_epochs = 200

Losses = []
val_losses = []

save_num = 1

dataset = TeluguOCRDataset("/home/ocr/teluguOCR/Dataset/Cropped_Dataset/Images", "/home/ocr/teluguOCR/Dataset/Cropped_Dataset/Labels")

# splitting the dataset into training and validation
torch.manual_seed(100)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)


for epoch in range(1, num_of_epochs + 1):
    start_time = time.time()
    # Training the model
    epoch_loss = EPOCH_RUN(Encoder, Decoder, train_dataloader, optimizer, criterion, training = True)
    # Validation
    val_loss = EPOCH_RUN(Encoder, Decoder, val_dataloader, optimizer, criterion, training = False)
    print("Epoch : ", epoch, " | Loss : ", (epoch_loss*64)/len(train_dataset), " | Validation Loss : ", (val_loss*64)/len(val_dataset), " | Time : ", time.time() - start_time)
    Losses.append((epoch_loss*64)/len(train_dataset))
    val_losses.append((val_loss*64)/len(val_dataset))
    if epoch %10 == 0:
        torch.save(Encoder.state_dict(), "/home/ocr/teluguOCR/Models/CNN/ModelTrans_E_TO_E" + str(save_num) + ".pth")
        torch.save(Decoder.state_dict(), "/home/ocr/teluguOCR/Models/RNN/ModelTrans_E_TO_E" + str(save_num) + ".pth")
        save_num += 1

# saving the losses into a pt file
torch.save(torch.tensor(Losses), "/home/ocr/teluguOCR/Losses/Training_Trans_Losses_Final_1.pt")
torch.save(torch.tensor(val_losses), "/home/ocr/teluguOCR/Losses/Validation_Trans_Losses_Final_1.pt")

# Plotting the losses
plt.figure(figsize=(12, 8))
plt.plot(Losses, label = "Training Loss", color = 'blue')
plt.plot(val_losses, label = "Validation Loss", color = 'red')
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    shadow=True,
    ncol=2
)
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("/home/ocr/teluguOCR/Losses/Losses_Trans_Plot_Final_1.png")      