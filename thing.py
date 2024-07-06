import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fcl = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fcl(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# Load MNIST dataset
train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')

for epoch in range(1, 11):
    train(epoch)
    test()

# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn.pth')

# Initialize the Tkinter window
root = tk.Tk()
root.title("Draw a digit")

# Set up the canvas
canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack()

# Create an empty image for drawing
image = Image.new("L", (200, 200), color=255)
draw = ImageDraw.Draw(image)

# Function to draw on the canvas
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill=0, width=5)

# Bind the paint function to the left mouse button
canvas.bind("<B1-Motion>", paint)

# Function to preprocess the image and make a prediction
def predict():
    # Resize and invert the image (black background)
    resized_image = image.resize((28, 28))
    inverted_image = ImageOps.invert(resized_image)
    
    # Convert the image to a numpy array and normalize it
    image_array = np.array(inverted_image) / 255.0
    image_array = image_array.astype(np.float32)
    image_array = image_array[np.newaxis, np.newaxis, :, :]  # Add batch and channel dimensions

    # Convert the numpy array to a PyTorch tensor
    input_tensor = torch.from_numpy(image_array).to(device)
    
    # Make a prediction using the neural network
    with torch.no_grad():
        output = model(input_tensor)
        prediction = F.softmax(output, dim=1).argmax(dim=1).item()

    # Display the prediction
    result_label.config(text=f"Prediction: {prediction}")

# Load your trained model
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# Button to make a prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Label to display the result
result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

# Start the Tkinter event loop
root.mainloop()
