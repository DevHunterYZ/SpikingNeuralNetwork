import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import norse.torch as norse

# Veri yükleme ve ön işleme (önceki gibi)
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Veriyi PyTorch tensörlerine dönüştürme
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)



# Güncellenmiş Spiking Neural Network modeli
class SNNClassifier(nn.Module):
    def __init__(self, input_features, hidden_neurons, output_classes):
        super(SNNClassifier, self).__init__()
        self.input_layer = norse.LIFCell(
            norse.LIFParameters(method="super", alpha=100)
        )
        self.hidden_layer = nn.Linear(input_features, hidden_neurons)
        self.output_layer = norse.LIFCell(
            norse.LIFParameters(method="super", alpha=100)
        )
        self.readout = nn.Linear(hidden_neurons, output_classes)
        
    def forward(self, x, dt=0.001):
        seq_length = 100
        batch_size = x.shape[0]
        hidden_state = None
        out_state = None
        
        # Girişi tekrarlayarak spike dizisi oluşturma
        spikes = x.unsqueeze(0).repeat(seq_length, 1, 1)
        
        outputs = []
        for step in range(seq_length):
            z, hidden_state = self.input_layer(spikes[step], hidden_state)
            z = self.hidden_layer(z)
            z, out_state = self.output_layer(z, out_state)
            outputs.append(z)
        
        outputs = torch.stack(outputs)
        return self.readout(outputs.mean(0))  # Zaman boyunca ortalama alıp sınıflandırma

# Model, loss fonksiyonu ve optimizer
model = SNNClassifier(4, 10, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Eğitim döngüsü
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Her 10 epoch'ta bir test yapma
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).float().mean()
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy.item():.4f}')

# Final test
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Final Test Accuracy: {accuracy.item():.4f}')
