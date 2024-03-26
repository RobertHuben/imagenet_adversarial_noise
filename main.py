import torch
from torchvision import models, transforms, utils
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

def model_probabilities(input):
    return torch.softmax(model(input), dim=1).flatten()

def optimize_towards_class(original_image, target_class, num_iter=10, alpha=1e-3):
    noised_image=torch.clone(original_image)
    noised_image.requires_grad_() #we'll be optimizing this to add noise

    optimizer=torch.optim.Adam(params=[noised_image], lr=0.01)
    print("Before noise is added, my original prediction is:")
    classify(original_image)
    for step_num in range(num_iter):
        optimizer.zero_grad()
        loss=loss_from_given_class(noised_image, target_class)+alpha*loss_l2(noised_image, original_image)
        loss.backward()
        optimizer.step()
        class_probability=model_probabilities(noised_image)[target_class]
        print(f"After {step_num} steps, the chance of this class being {target_class} is {class_probability}")
    print("After all the steps I predict the object to be:")
    classify(noised_image)
    utils.save_image(noised_image[0],"foo.jpg")


def load_file_path_to_tensor(path_to_image):
    # Define the transformation to apply to the image
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])
    image = Image.open(path_to_image)
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image


def classify(image_as_tensor):
    # Perform the classification
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model_probabilities(image_as_tensor)
        prob, class_num = outputs.max(0)

    # Get the predicted class label
    imagenet_labels = load_labels()
    predicted_class = imagenet_labels[class_num.item()]
    print(f"Predicted class: {predicted_class}, probability: {prob.item():.2%}")

def loss_from_given_class(image_as_tensor, target_class):
    model_output = model_probabilities(image_as_tensor)
    loss=-1*torch.log(model_output[target_class])
    return loss

def loss_l2(image_as_tensor, original_image_as_tensor):
    return torch.norm(image_as_tensor-original_image_as_tensor, p=2)

def load_labels():
    with open("imagenet_classes.txt") as f:
        imagenet_labels = [line.split(", ")[1].strip() for line in f.readlines()]
    return imagenet_labels

if __name__=="__main__":
    for image_name in ["bird", "cat", "shark", "snake"]:
        image_path=f"inputs/{image_name}.jpg"
        image=load_file_path_to_tensor(image_path)
        optimize_towards_class(image, 10)
        classify(image)    