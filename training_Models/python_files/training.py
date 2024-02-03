from utils import *
from lstm import *
from cnn import *

cnn = EncoderCNN().to(device)
lstm = LSTM_NET().to(device)

# loading the model
cnn.load_state_dict(torch.load("/home/ocr/teluguOCR/Models/CNN/Model2.pth"))
lstm.load_state_dict(torch.load("/home/ocr/teluguOCR/Models/RNN/Model2.pth"))

cnn.train()
lstm.train()

# loss function and optimizer
criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean') if torch.cuda.is_available() else nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean')

params = list(cnn.parameters()) + list(lstm.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

clip = 10
for p in params:
    p.register_hook(lambda grad: torch.clamp(grad, -clip, clip) if grad is not None else None)

num_of_epochs = 2000

Losses = []

save_num = 1

for epoch in range(201, num_of_epochs + 1):

    start_time = time.time()
    num_of_files = 24
    Number_of_images = 500
    epoch_loss = 0
    for file in range(1, num_of_files+1):
        images = torch.load("/home/ocr/teluguOCR/Dataset/Full_Image_Tensors/Full_Image_Tensors" + str(file) + ".pt")
        labels = torch.load("/home/ocr/teluguOCR/Dataset/Full_Label_Tensors/Full_Label_Tensors" + str(file) + ".pt")
        target_lengths = torch.load("/home/ocr/teluguOCR/Dataset/Full_label_length_tensors/Full_Label_Lengths" + str(file) + ".pt")

        images = images.to(device)
        labels = labels.to(device)
        target_lengths = target_lengths.to(device)

        # cnn forward pass
        cnn_output = cnn(images).unsqueeze(1)

        # lstm forward pass
        f_output = torch.zeros(Image_length, Number_of_images, Text_embedding_size).to(device)
        for k in range(Image_length):
            f_output[k, : , :] = lstm(cnn_output[:, :, k, :], k == 0).squeeze(1)

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
        input_lengths = torch.full(size=(Number_of_images,), fill_value=Image_length, dtype=torch.long).to(device)

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

        print("file Number", file, end = '\r')
    
    print("Epoch : ", epoch, " | Loss : ", epoch_loss, " | Time taken : ", time.time() - start_time)
    Losses.append(epoch_loss)

    if epoch %100 == 0:
        torch.save(cnn.state_dict(), "/home/ocr/teluguOCR/Models/CNN/Model" + str(save_num) + ".pth")
        torch.save(lstm.state_dict(), "/home/ocr/teluguOCR/Models/RNN/Model" + str(save_num) + ".pth")
        save_num += 1


# Plotting the losses
import matplotlib.pyplot as plt
plt.plot(Losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("/home/ocr/teluguOCR/Losses.png")      