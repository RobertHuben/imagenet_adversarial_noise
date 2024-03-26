import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

def optimize_towards_class(original_image, target_class, num_iter=10, alpha=1e-3):
    noised_image=torch.clone(original_image)
    noised_image.requires_grad_() #we'll be optimizing this to add noise

    optimizer=torch.optim.Adam(params=[noised_image], lr=0.0001)
    for _ in range(num_iter):
        optimizer.zero_grad()
        loss=loss_from_given_class(noised_image, target_class)+alpha*loss_l2(noised_image, original_image)
        loss.backward()
        optimizer.step()
    


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


def classify(image_as_tensor):
    # Perform the classification
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(image)
        _, predicted = outputs.max(1)

    # Get the predicted class label
    imagenet_labels = load_labels()
    predicted_class = imagenet_labels[predicted.item()]
    print(f"Predicted class: {predicted_class}")

def loss_from_given_class(image_as_tensor, target_class):
    model_output = model(image_as_tensor)
    loss=-1*torch.log(model_output[0,target_class])
    return loss

def loss_l2(image_as_tensor, original_image_as_tensor):
    return torch.norm(image_as_tensor-original_image_as_tensor, p=2)

def load_labels():
    with open("imagenet_classes.txt") as f:  # You need to have this file with the class names
        imagenet_labels = [line.split(", ")[1].strip() for line in f.readlines()]
    return imagenet_labels

if __name__=="__main__":
    for image_name in ["bird", "cat", "shark", "snake"]:
        image_path=f"inputs/{image_name}.jpg"
        image=load_file_path_to_tensor(image_path)
        optimize_towards_class(image, 10)
        classify(image)    