from utils import *

def EPOCH_RUN(cnn, Decoder, dataloader, optimizer, criterion, training = True):
    if training:
        cnn.train()
        Decoder.train()
    else:
        cnn.eval()
        Decoder.eval()
    
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
        
        #CRNN + TRANSFORMER
        Encoder_output = cnn(images)

        # # adding the starting token to the labels and making it RSLabels
        # RSLabels = torch.zeros(labels.shape[0], labels.shape[1]+1, Text_embedding_size).to(device)
        # RSLabels[:, 0, 1] = 1
        # for i in range(labels.shape[0]):
        #     for j in range(labels.shape[1]):
        #         RSLabels[i, j+1, int(labels[i, j, 0])+2] = 1
        
        # # Adding the ending token to the labels for loss calculation
        # LSLabels = torch.zeros(labels.shape[0], labels.shape[1]+1, 9).to(device)
        # LSLabels[:, -1, 0] = 2
        # for i in range(labels.shape[0]):
        #     for j in range(labels.shape[1]):
        #         LSLabels[i, j, 0] = labels[i, j, 0]
        


        # f_output = Decoder(Encoder_output, RSLabels).permute(1, 0, 2)

        f_output = Decoder(Encoder_output).permute(1, 0, 2)

        input_lengths = torch.full(size=(images.shape[0],), fill_value=45, dtype=torch.long).to(device)

        # applying log_softmax
        f_output[:, :, 0:112] = F.log_softmax(f_output[:, :, 0:112], dim=2)
        f_output[:, :, 110:131] = F.log_softmax(f_output[:, :, 110:131], dim=2)
        f_output[:, :, 131:152] = F.log_softmax(f_output[:, :, 131:152], dim=2)
        f_output[:, :, 152:190] = F.log_softmax(f_output[:, :, 152:190], dim=2)
        f_output[:, :, 190:228] = F.log_softmax(f_output[:, :, 190:228], dim=2)

        Loss = 0
        # for base
        Loss += criterion(f_output[:, :, 0:110], labels[:, :,0], input_lengths, target_lengths)
        # for vm1
        Loss += criterion(f_output[:, :,110:131], labels[:, :,1], input_lengths, target_lengths)
        # for vm2
        Loss += criterion(f_output[:, :, 131:152], labels[:, :,2], input_lengths, target_lengths)
        # for cm1
        Loss += criterion(f_output[:, :, 152:190], labels[:, :,5], input_lengths, target_lengths)
        # for cm2
        Loss += criterion(f_output[:, :, 190:228], labels[:, :,6], input_lengths, target_lengths)

        if training:
            # Backpropagation
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        epoch_loss += Loss.item()

    return epoch_loss