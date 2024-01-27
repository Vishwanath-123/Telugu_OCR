from utils import *
from lstm import LSTM_NET
from cnn import EncoderCNN

cnn = EncoderCNN().to(device)
lstm = LSTM_NET().to(device)

cnn.train()
lstm.train()

# loss function and optimizer
criterion = nn.CTCLoss(blank=0, zero_infinity=True).cuda() if torch.cuda.is_available() else nn.CTCLoss(blank=0, zero_infinity=True)

params = list(cnn.parameters()) + list(lstm.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)

num_of_epochs = 150

Losses = []

save_num = 1

for epoch in range(1, num_of_epochs + 1):
    start_time = time.time()
    num_of_files = 5
    for file in range(1, num_of_files + 1):
        images = torch.load("/home/ocr/teluguOCR/Dataset/Full_Image_Tensors/Full_Image_Tensors" + str(file) + ".pt")
        labels = torch.load("/home/ocr/teluguOCR/Dataset/Full_Label_Tensors/Full_Label_Tensors" + str(file) + ".pt")
        target_lengths = torch.load("/home/ocr/teluguOCR/Dataset/Full_label_length_tensors/Full_Label_Lengths" + str(file) + ".pt")

        images = images.to(device)
        labels = labels.to(device)
        target_lengths = target_lengths.to(device)

        # cnn forward pass
        cnn_output = cnn(images).unsqueeze(1)

        # lstm forward pass
        f_output = torch.zeros(100, Image_length, Text_embedding_size).to(device)
        for k in range(Image_length):
            f_output[:, k, :] = lstm(cnn_output[:, :, k, :], k == 0).squeeze(1) 

        # applying log_softmax
        f_output[:, :, :114] = F.log_softmax(f_output[:, :, :114], dim=2)
        f_output[:, :, 114:135] = F.log_softmax(f_output[:, :, 114:135], dim=2)
        f_output[:, :, 135:156] = F.log_softmax(f_output[:, :, 135:156], dim=2)
        f_output[:, :, 156:177] = F.log_softmax(f_output[:, :, 156:177], dim=2)
        f_output[:, :, 177:198] = F.log_softmax(f_output[:, :, 177:198], dim=2)
        f_output[:, :, 198:240] = F.log_softmax(f_output[:, :, 198:240], dim=2)
        f_output[:, :, 240:282] = F.log_softmax(f_output[:, :, 240:282], dim=2)
        f_output[:, :, 282:324] = F.log_softmax(f_output[:, :, 282:324], dim=2)
        f_output[:, :, 324:366] = F.log_softmax(f_output[:, :, 324:366], dim=2)

        # Loss calculation
        input_lengths = torch.full(size=(100,), fill_value=Image_length, dtype=torch.long).to(device)

        Loss = 0
        # for base
        Loss += criterion(f_output[:, :, :114], labels[:, :,0], input_lengths, target_lengths)
        # for vm1
        Loss += criterion(f_output[:, :, 114:135], labels[:, :,1], input_lengths, target_lengths)
        # for vm2
        Loss += criterion(f_output[:, :, 135:156], labels[:, :,2], input_lengths, target_lengths)
        # for vm3
        Loss += criterion(f_output[:, :, 156:177], labels[:, :,3], input_lengths, target_lengths)
        # for vm4
        Loss += criterion(f_output[:, :, 177:198], labels[:, :,4], input_lengths, target_lengths)
        # for cm1
        Loss += criterion(f_output[:, :, 198:240], labels[:, :,5], input_lengths, target_lengths)
        # for cm2
        Loss += criterion(f_output[:, :, 240:282], labels[:, :,6], input_lengths, target_lengths)
        # for cm3
        Loss += criterion(f_output[:, :, 282:324], labels[:, :,7], input_lengths, target_lengths)
        # for cm4
        Loss += criterion(f_output[:, :, 324:366], labels[:, :,8], input_lengths, target_lengths)
    
        # Backpropagation
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        print("file Number", file, end = '\r')
    
    print("Epoch : ", epoch, " | Loss : ", Loss.item(), " | Time taken : ", time.time() - start_time)
    Losses.append(Loss.item())

    if epoch %100 == 0:
        torch.save(cnn.state_dict(), "/home/ocr/teluguOCR/Models/Model" + str(save_num) + ".pt")
        torch.save(lstm.state_dict(), "/home/ocr/teluguOCR/Models/Model" + str(save_num) + ".pt")
        save_num += 1
        