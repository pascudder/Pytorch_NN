import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

    
def get_data_loader(training=True):
    
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    
    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if training:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    else:
        dataset = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    return data_loader


def build_model():
    """
    INPUT: 
        None
    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        correct = 0
        total_loss = 0
        total = 0

        for images, labels in train_loader:
            opt.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / total
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({accuracy:.2%}) Loss: {avg_loss:.3f}")
    return None


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += criterion(output, labels).item() * data.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total

    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    return None

def predict_label(model, test_images, index):
    """
    INPUT:
        model - the trained model
        test_images - a tensor. test image set of shape Nx1x28x28
        index - specific index i of the image to be tested: 0 <= i <= N - 1
    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    model.eval()
    
    with torch.no_grad():
        logits = model(test_images[index].unsqueeze(0))
        prob = F.softmax(logits, dim=1)
    
    top3_prob, top3_labels = torch.topk(prob, 3)
    
    for i in range(3):
        print(f"{class_names[top3_labels[0][i]]}: {top3_prob[0][i]:.2%}")
    
    return None

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()

    #train_loader = get_data_loader()
    #print(type(train_loader))
    #print(train_loader.dataset)
    #test_loader = get_data_loader(False)
    #print(type(test_loader))
    #print(test_loader.dataset)
    #model = build_model()
    #print(model)
    #train_model(model, train_loader, criterion, 1)
    #evaluate_model(model,train_loader,criterion)
    #test_images = next(iter(test_loader))[0]
    #predict_label(model, test_images, 1)
