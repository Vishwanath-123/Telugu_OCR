from utils import *
from lstm import DECODER
from cnn import EncoderCNN
from dataset import TeluguOCRDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

cnn = EncoderCNN().to(device)
decoder = DECODER().to(device)

# # loading the model
# cnn.load_state_dict(torch.load("/home/ocr/teluguOCR/Models/CNN/ModelGRU_10.pth"))
# decoder.load_state_dict(torch.load("/home/ocr/teluguOCR/Models/RNN/ModelGRU_10.pth"))

cnn.train()
decoder.train()

# loss function and optimizer
criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean') if torch.cuda.is_available() else nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean')

params = list(cnn.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-6)

num_of_epochs = 100

Losses = []
val_losses = []

save_num = 11

dataset = TeluguOCRDataset("/home/ocr/teluguOCR/Dataset/Cropped_Dataset/Images", "/home/ocr/teluguOCR/Dataset/Cropped_Dataset/Labels")

# splitting the dataset into training and validation
torch.manual_seed(0)
train_dataset, val_dataset= torch.utils.data.random_split(dataset, [0.6, 0.4])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

for epoch in range(1, num_of_epochs + 1):
    cnn.train()
    decoder.train()

    start_time = time.time()
    epoch_loss = 0
    idx = 1
    for images, labels, target_lengths in train_dataloader:
            print(idx, end = "\r")
            idx+=1
            images = images.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)

            # cnn forward pass
            cnn_output = cnn(images).unsqueeze(1)

            # # decoder forward pass
            # f_output = torch.zeros(Image_length, images.shape[0], Text_embedding_size).to(device)

            # for k in range(Image_length):
            #     f_output[k, : , :] = decoder(cnn_output[:, :, k, :], k == 0).squeeze(1)
            f_output = decoder(cnn_output.squeeze(1), True).permute(1, 0, 2)


            # applying log_softmax
            f_output[:, :, 0:110] = F.log_softmax(f_output[:, :, 0:110], dim=2)
            f_output[:, :, 110:131] = F.log_softmax(f_output[:, :, 110:131], dim=2)
            f_output[:, :, 131:152] = F.log_softmax(f_output[:, :, 131:152], dim=2)
            f_output[:, :, 152:173] = F.log_softmax(f_output[:, :, 152:173], dim=2)
            f_output[:, :, 173:194] = F.log_softmax(f_output[:, :, 173:194], dim=2)
            f_output[:, :, 194:232] = F.log_softmax(f_output[:, :, 194:232], dim=2)
            f_output[:, :, 232:270] = F.log_softmax(f_output[:, :, 232:270], dim=2)
            f_output[:, :, 270:308] = F.log_softmax(f_output[:, :, 270:308], dim=2)
            f_output[:, :, 308:346] = F.log_softmax(f_output[:, :, 308:346], dim=2)

            # Loss calculation
            input_lengths = torch.full(size=(images.shape[0],), fill_value=Image_length, dtype=torch.long).to(device)

            Loss = 0
            # for base
            Loss += criterion(f_output[:, :, 0:110], labels[:, :,0], input_lengths, target_lengths)
            # for vm1
            Loss += criterion(f_output[:, :,110:131], labels[:, :,1], input_lengths, target_lengths)
            # for vm2
            Loss += criterion(f_output[:, :, 131:152], labels[:, :,2], input_lengths, target_lengths)
            # for vm3
            Loss += criterion(f_output[:, :, 152:173], labels[:, :,3], input_lengths, target_lengths)
            # for vm4
            Loss += criterion(f_output[:, :, 173:194], labels[:, :,4], input_lengths, target_lengths)
            # for cm1
            Loss += criterion(f_output[:, :, 194:232], labels[:, :,5], input_lengths, target_lengths)
            # for cm2
            Loss += criterion(f_output[:, :, 232:270], labels[:, :,6], input_lengths, target_lengths)
            # for cm3
            Loss += criterion(f_output[:, :, 270:308], labels[:, :,7], input_lengths, target_lengths)
            # for cm4
            Loss += criterion(f_output[:, :, 308:346], labels[:, :,8], input_lengths, target_lengths)
        
            # Backpropagation
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            epoch_loss += Loss.item()


    # Calculating Validation loss 
    cnn.eval()
    decoder.eval()
    val_loss = 0

    for images, labels, target_lengths in val_dataloader:

            images = images.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)

            # cnn forward pass
            cnn_output = cnn(images).unsqueeze(1)

            # decoder forward pass
            f_output = torch.zeros(Image_length, images.shape[0], Text_embedding_size).to(device)
            for k in range(Image_length):
                f_output[k, : , :] = decoder(cnn_output[:, :, k, :], k == 0).squeeze(1)

            # applying log_softmax
            f_output[:, :, 0:110] = F.log_softmax(f_output[:, :, 0:110], dim=2)
            f_output[:, :, 110:131] = F.log_softmax(f_output[:, :, 110:131], dim=2)
            f_output[:, :, 131:152] = F.log_softmax(f_output[:, :, 131:152], dim=2)
            f_output[:, :, 152:173] = F.log_softmax(f_output[:, :, 152:173], dim=2)
            f_output[:, :, 173:194] = F.log_softmax(f_output[:, :, 173:194], dim=2)
            f_output[:, :, 194:232] = F.log_softmax(f_output[:, :, 194:232], dim=2)
            f_output[:, :, 232:270] = F.log_softmax(f_output[:, :, 232:270], dim=2)
            f_output[:, :, 270:308] = F.log_softmax(f_output[:, :, 270:308], dim=2)
            f_output[:, :, 308:346] = F.log_softmax(f_output[:, :, 308:346], dim=2)

            # Loss calculation
            input_lengths = torch.full(size=(images.shape[0],), fill_value=Image_length, dtype=torch.long).to(device)

            Loss = 0
            # for base
            Loss += criterion(f_output[:, :, 0:110], labels[:, :,0], input_lengths, target_lengths)
            # for vm1
            Loss += criterion(f_output[:, :,110:131], labels[:, :,1], input_lengths, target_lengths)
            # for vm2
            Loss += criterion(f_output[:, :, 131:152], labels[:, :,2], input_lengths, target_lengths)
            # for vm3
            Loss += criterion(f_output[:, :, 152:173], labels[:, :,3], input_lengths, target_lengths)
            # for vm4
            Loss += criterion(f_output[:, :, 173:194], labels[:, :,4], input_lengths, target_lengths)
            # for cm1
            Loss += criterion(f_output[:, :, 194:232], labels[:, :,5], input_lengths, target_lengths)
            # for cm2
            Loss += criterion(f_output[:, :, 232:270], labels[:, :,6], input_lengths, target_lengths)
            # for cm3
            Loss += criterion(f_output[:, :, 270:308], labels[:, :,7], input_lengths, target_lengths)
            # for cm4
            Loss += criterion(f_output[:, :, 308:346], labels[:, :,8], input_lengths, target_lengths)

            val_loss += Loss.item()
    
    print("Epoch : ", epoch, " | Loss : ", (epoch_loss*64)/len(train_dataset), " | Validation Loss : ", (val_loss*64)/len(val_dataset), " | Time : ", time.time() - start_time)
    Losses.append((epoch_loss*64)/len(train_dataset))
    val_losses.append((val_loss*64)/len(val_dataset))
    if epoch %10 == 0:
        torch.save(cnn.state_dict(), "/home/ocr/teluguOCR/Models/CNN/ModelTrans_" + str(save_num) + ".pth")
        torch.save(decoder.state_dict(), "/home/ocr/teluguOCR/Models/RNN/ModelTrans_" + str(save_num) + ".pth")
        save_num += 1

# saving the losses into a pt file
torch.save(torch.tensor(Losses), "/home/ocr/teluguOCR/Losses/Training_Trans_Losses_Final_2.pt")
torch.save(torch.tensor(val_losses), "/home/ocr/teluguOCR/Losses/Validation_Trans_Losses_Final_2.pt")

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
plt.savefig("/home/ocr/teluguOCR/Losses/Losses_Trans_Plot_Final_2.png")      