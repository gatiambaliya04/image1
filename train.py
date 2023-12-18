import argparse
from torchvision import models
from train_utils import train_model

parser = argparse.ArgumentParser(description='Train a flower image classification model')
parser.add_argument('--data_dir', type=str, default='path/to/your/flower/dataset', help='Path to the flower dataset')
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help='Architecture of the model')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in the classifier')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

# Load a pre-trained model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
    input_size = 512

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Define a new classifier
classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

# Replace the model classifier
model.classifier = classifier

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Move model to device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Load and preprocess the data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(root=f"{args.data_dir}/{x}", transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train
