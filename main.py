import torch
from torchvision import models, transforms
from PIL import Image


def load_file_path_to_tensor(path_to_image):
    # Define the transformation to apply to the image
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    image = Image.open(path_to_image)
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image



def classify(input_tensor):
    # simple method written by ChatGPT to start the classifier off
    # Load a pre-trained model, for example, ResNet18
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    image=load_file_path_to_tensor(image_path)

    # Perform the classification
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(image)
        _, predicted = outputs.max(1)

    # Decode the prediction
    # Load the labels (ensure you have the ImageNet labels file if using an ImageNet model)
    imagenet_labels = []
    with open("imagenet_classes.txt") as f:  # You need to have this file with the class names
        imagenet_labels = [line.split(", ")[1].strip() for line in f.readlines()]

    # Get the predicted class label
    predicted_class = imagenet_labels[predicted.item()]

    print(f"Predicted class: {predicted_class}")

for image_name in ["bird", "cat", "shark", "snake"]:
    image_path=f"inputs/{image_name}.jpg"
    classify(image_path)    