# -*- coding: utf-8 -*-
"""
problem tanimi: CIFAR10 veriseti siniflandirma problemi
CNN

"""
# %% import libraries
import torch # pytorch
import torch.nn as nn # sinir agi katmanlari icin
import torch.optim as optim # optimizasyon algortimasi icin
import torchvision # goruntu isleme icin
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load dataset 
def get_data_loaders(batch_size = 64): # batch size; her iterasyonda islenecek veri sayisi

    transform = transforms.Compose([
        transforms.ToTensor(), # goruntuyu tensore cevir
        transforms.Normalize(((0.5, 0.5 ,0.5)),(0.5, 0.5, 0.5)) # rgb kanallarini normalize et
        ])
    
    # CIFAR10 verisetini indir ve egitim test verisetini olustur
    train_set = torchvision.datasets.CIFAR10(root = "./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root = "./data", train=False, download=True, transform=transform)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# %% visualize dataset (veri gorsellestirme)
def imshow(img):
    # verileri normalize etmeden once geri donustur
    img = img / 2 + 0.5 # normalize isleminin tersi
    np_image = img.numpy() # tensordan numpy array e dondur
    plt.imshow(np.transpose(np_image, (1,2,0))) # 3 kanal icin renkleri dogru sekilde gosterme
    #plt.show()
    
def get_samples_images(train_loader): # veri kumesinden ornek gıorselleri almak icin fonksiyon

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(n):
    train_loader, test_loader = get_data_loaders()
    
    # n tane veri gorsellestirme
    images, labels = get_samples_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(images[i]) # gorsellestir
        plt.title(f"label: {labels[i].item()}")
        plt.axis("off")
    plt.show()
    
#visualize(3)
    

# %% build cnn model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1) # in_channels = rgb-3, out_channels-filtre sayisi, kernel_size-3*3
        self.relu = nn.ReLU() # aktivasyon fonksiyonu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2*2 boyutunda pooling katmani
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding=1) # 64 filtreli 2. convolutional layer
        self.dropout = nn.Dropout(0.2) # dropout %20 oraninda calisir. degerleri yok sayarak fazla ogrenmenin onune gecer
        self.fc1 = nn.Linear(64*8*8, 128) # fully connected layer giris = 4096, cikis = 128
        self.fc2 = nn.Linear(128, 10) # output layer
        
        
        # img 3*32*32 -> conv (32) -> relu (32) -> pool (16)
        # conv (16) -> relu (16) -> pool (8) -> img = 8*8
        
    def forward(self,x):
        """
        img 3*32*32 -> conv (32) -> relu (32) -> pool (16)
        conv (16) -> relu (16) -> pool (8) -> img = 8*8
        flatten
        fc1 -> relu -> dropout
        fc2 -> output
        
        """
        x = self.pool(self.relu(self.conv1(x))) # ilk convolutional blok
        x = self.pool(self.relu(self.conv2(x))) # ikinci convolutional blok
        x = x.view(-1, 64*8*8) # flatten
        x = self.dropout(self.relu(self.fc1(x))) # fully connected layer
        x = self.fc2(x) # output
        return x
        
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = CNN().to(device)
        
        
# define loss function and optimizer

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # multiclass classification problem
    optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) # momentum: stochastic gradient descent (SGD) optimizasyon algortimasina hiz kazandirir
    # ve yerel minimumlara takilmayi onleyen  bir teknik.
    )



# %% training 
def train_model(model, train_loader, criterion, optimizer, epochs = 5):
    
    model.train() # modeli egitim moduna alalim
    train_losses = [] # loss degerlerini saklamak icin bir liste olustur
    
    for epoch in range(epochs): # belirtilen epochs sayisi kadar for dongusu olustur
        total_loss = 0 # toplam loss degerini saklamak icin
        for images, labels in train_loader: # tüm egitim verisetini taramak icin for dongusu
            images, labels = images.to(device), labels.to(device)
            
            
            optimizer.zero_grad() # gradyanlari sifirlamak
            outputs = model(images) # forward pro. (prediction) output=etiket, label, class
            loss = criterion(outputs, labels) # loss degeri hesapla
            loss.backward() # geri yayilim (gradyan hesaplama)
            optimizer.step() # ogrenme = parametre yani agirlik guncelleme
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader) # ortalama kayip hesaplama
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epoch}, Loss: {avg_loss:.5f}")
        
        # kayip loss grafigi
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
        
# train_loader, test_loader = get_data_loaders()
# model = CNN().to(device)
# criterion, optimizer = define_loss_and_optimizer(model)
# train_model(model, train_loader, criterion, optimizer, epochs = 5)
    


# %% test
def test_model(model, test_loader, dataset_type):
    model.eval() # degerlendirme modu
    correct = 0 # dogru tahmin sayaci
    total = 0 # toplam veri sayaci
    
    with torch.no_grad(): # gradyan hesaplamasini kapat
        for images, labels in test_loader: # test verisetini kullanarak degerlendirme
            images, labels = images.to(device), labels.to(device) # verileri cihaza tasi
            
            outputs = model(images) # prediction
            _, predicted = torch.max(outputs, 1) # en yuksek olasilikli sinifi sec
            total += labels.size(0) # toplam veri sayisi
            correct += (predicted == labels).sum().item() # dogru tahminleri say
            
    print(f"{dataset_type} accuracy: {100 * correct / total}%") # dogruluk oranini ekrana yazdir
#test_model(model, test_loader, dataset_type="test") # test accuracy: 55.42%
#test_model(model, train_loader, dataset_type="training") # training accuracy: 55.82%


# %% main program
if __name__ == "__main__":
    
    # veriseti yukleme
    train_loader, test_loader = get_data_loaders()
    
    # gorsellestirme
    visualize(5)
    
    # training
    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs = 10)
    
    # test
    test_model(model, test_loader, dataset_type="test") # test accuracy: 55.42%
    test_model(model, train_loader, dataset_type="training") # training accuracy: 55.82%








