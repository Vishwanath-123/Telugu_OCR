from utils import *


def Epoch_Run(encoder, decoder, dataloader, optimizer, criterion, training = True):
    if training:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()
    
    epoch_loss = 0

    idx = 1

    for images, labels, target_lengths, input_lengths in dataloader:
        if training:
            print("T: ", idx, end = "\r")
        else:
            print("V: ", idx, end = "\r")

        idx+=1
        images = images.to(device)
        labels = labels.to(device)
        target_lengths = target_lengths.to(device)
        input_lengths = input_lengths.to(device)

        # Encoder
        ENC_output = encoder(images).unsqueeze(1)

        # Decoder
        # Dec_Out = torch.zeros(Image_length, images.shape[0], Text_embedding_size).to(device)
        # for k in range(Image_length):
        #     Dec_Out[k, : , :] = decoder(ENC_output[:, :, k, :], k%10 == 0).squeeze(1)

        Dec_Out = decoder(ENC_output)

        # applying log_softmax
        Dec_Out[:, :, 0:110] = F.log_softmax(Dec_Out[:, :, 0:110], dim=2)

        Dec_Out[:, :, 110:131] = F.log_softmax(Dec_Out[:, :, 110:131], dim=2)
        Dec_Out[:, :, 131:152] = F.log_softmax(Dec_Out[:, :, 131:152], dim=2)
        
        Dec_Out[:, :, 152:190] = F.log_softmax(Dec_Out[:, :, 152:190], dim=2)
        Dec_Out[:, :, 190:228] = F.log_softmax(Dec_Out[:, :, 190:228], dim=2)

        input_lengths = torch.full(size=(images.shape[0],), fill_value=100, dtype=torch.long).to(device)

        # Loss Calculation
        Loss = 0

        # for base
        Loss += criterion(Dec_Out[:, :, 0:110], labels[:, :,0], input_lengths, target_lengths)

        # for vm1
        Loss += criterion(Dec_Out[:, :,110:131], labels[:, :,1], input_lengths, target_lengths)
        # for vm2
        Loss += criterion(Dec_Out[:, :, 131:152], labels[:, :,2], input_lengths, target_lengths)

        # for cm1
        Loss += criterion(Dec_Out[:, :, 152:190], labels[:, :,5], input_lengths, target_lengths)
        # for cm2
        Loss += criterion(Dec_Out[:, :, 190:228], labels[:, :,6], input_lengths, target_lengths)



        if training:
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        epoch_loss += Loss.item()

    return epoch_loss