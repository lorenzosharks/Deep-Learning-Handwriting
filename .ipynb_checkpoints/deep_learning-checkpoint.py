from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils


train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor,
    download = True
)

train_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor,
    download = True
)

