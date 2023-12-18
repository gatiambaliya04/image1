import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

parser = argparse.ArgumentParser(description='Predict the flower class of an image')
parser.add_argument('--image_path', type=str, default='path/to/your/image.jpg', help='Path to the input image')
parser.add_argument('--checkpoint', type=str, default='flower_classification_checkpoint.pth', help='Path to the model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Top K most probable classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names mapping file')
parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction')

args = parser.parse_args()

# Load the model checkpoint
checkpoint = torch.load(args.checkpoint)
arch = checkpoint['arch']

# Load the pre-trained model
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif arch == 'resnet18':
    model = models.resnet18(pretrained=True)
    input_size = 512

# Replace the model classifier
model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

# Move model to device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Load and preprocess the input image
image = Image.open(args.image_path)
image = data_transforms['val'](image)
image = image.unsqueeze(0).to(device)

# Make prediction
model.eval()
with torch.no_grad():
    output = model(image)

# Get top K probabilities and class indices
probs, indices = torch.topk(torch.exp(output), args.top_k)
probs = probs.cpu().numpy()[0]
indices = indices.cpu().numpy()[0]

# Load category names mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Map indices to class names
class_names = [cat_to_name[str(model.class_to_idx[str(idx)])] for idx in indices]

# Print the results
print("\nTop K Classes and Probabilities:")
for i in range(args.top_k):
    print(f"{class_names[i]}: {probs[i]*100:.2f}%")
