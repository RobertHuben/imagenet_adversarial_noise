import torch
from torchvision import models, transforms, utils
from PIL import Image
from matplotlib import pyplot as plt

model = models.resnet18(weights='DEFAULT')
model.eval()  # Set the model to evaluation mode

def model_probabilities(input):
    return torch.softmax(model(input), dim=1).flatten()

def optimize_towards_class(original_image, target_class, num_iter=10, alpha=1e0):
    noised_image=torch.clone(original_image)
    noised_image.requires_grad_() #we'll be optimizing this to add noise

    optimizer=torch.optim.Adam(params=[noised_image], lr=0.01)
    for step_num in range(num_iter):
        optimizer.zero_grad()
        loss=loss_from_given_class(noised_image, target_class)+alpha*loss_l2(noised_image, original_image)
        loss.backward()
        optimizer.step()
        class_probability=model_probabilities(noised_image)[target_class]
        print(f"After {step_num+1} steps, the chance of this class being {target_class} is {class_probability}")
    make_comparison_image(original_image, noised_image)
    utils.save_image(noised_image[0],"foo.jpg")

def make_comparison_image(original_image, noised_image, save_name="comparison_images/output.png"):
    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0].transpose(0,2).transpose(0,1))
    original_prob, original_class_name, _=classify(original_image)
    plt.title(f"Original:\npredicts {original_class_name}\n(p={original_prob:.1%})")

    plt.subplot(1, 2, 2)
    plt.imshow(noised_image[0].detach().transpose(0,2).transpose(0,1))
    noised_prob, noised_class_name, _=classify(noised_image)
    plt.title(f"Adversarial Noise:\npredicts {noised_class_name}\n(p={noised_prob:.1%})")
    plt.savefig(save_name)


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
    class_name = imagenet_labels[class_num.item()]
    return prob, class_name, class_num

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
    target_class=9
    for image_name in ["bird", "cat", "shark", "snake"]:
        image_path=f"inputs/{image_name}.jpg"
        image=load_file_path_to_tensor(image_path)
        optimize_towards_class(image, target_class)