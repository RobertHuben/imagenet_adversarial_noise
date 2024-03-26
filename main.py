import torch
from torchvision import models, transforms
from PIL import Image
from matplotlib import pyplot as plt
import argparse

model = models.resnet18(weights='DEFAULT')
model.eval()  # Set the model to evaluation mode

def model_probabilities(input_image):
    '''
    runs the model on the given input_image, returning probabilities for each class after a softmax
    input_image must be a tensor of shape (1,3,256,256)
    output is a tensor of shape (1000)
    '''
    return torch.softmax(model(input_image), dim=1).flatten()

def top_one_classification(input_image):
    '''
    runs the model on given input_image, and returns the top-1 probability, class name, and class number
    input_image must be a tensor of shape (1,3,256,256)
    output is a float, a string, and an int
    '''
    # Perform the classification
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model_probabilities(input_image)
        prob, class_num = outputs.max(0)
    prob=prob.item()
    class_num=class_num.item()
    # Get the predicted class label
    imagenet_labels = load_labels()
    class_name = imagenet_labels[class_num]
    return prob, class_name, class_num

def load_image_from_file_path(path_to_image):
    '''
    loads the image at path_to_image, and processes it to be ready for imagenet
    '''
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

def load_labels():
    '''
    returns an array of the imagenet classes
    '''
    with open("imagenet_classes.txt") as f:
        imagenet_labels = [line.split(", ")[1].strip() for line in f.readlines()]
    return imagenet_labels

def optimize_towards_class(original_image, target_class, num_iter=10, l2_loss_coefficient=1e-2, print_updates=True):
    '''
    runs a simple optimization loop to increase the probability of the model saying the target class, without greatly increasing l2 distance from the original
    loss functions are loss_from_given_class and loss_l2, defined below
    prints updates of the chance of being the target image as we go
    returns the noised image
    '''
    noised_image=torch.clone(original_image)
    noised_image.requires_grad_() #we'll be optimizing this to add noise

    optimizer=torch.optim.Adam(params=[noised_image], lr=0.01)
    class_probability=model_probabilities(noised_image)[target_class]
    if print_updates:
        print(f"Before any steps, imagenet says the chance of being class {target_class} is {class_probability:.1%}")

    for step_num in range(num_iter):
        optimizer.zero_grad()
        loss=loss_from_given_class(noised_image, target_class)+l2_loss_coefficient*loss_l2(noised_image, original_image)
        loss.backward()
        optimizer.step()
        class_probability=model_probabilities(noised_image)[target_class]
        if print_updates:
            print(f"After {step_num+1} steps, imagenet says the chance of being class {target_class} is {class_probability:.1%}")
    return noised_image

def loss_from_given_class(image_as_tensor, target_class):
    '''
    first of two loss functions for our adversarial optimization. 
    this adds noise to the image to push it to being the target class
    '''
    model_output = model_probabilities(image_as_tensor)
    loss=-1*torch.log(model_output[target_class])
    return loss

def loss_l2(image_as_tensor, original_image_as_tensor):
    '''
    second of two loss functions for our adversarial optimization. 
    this keeps the noise "minimal" so that the original image is still recognizable
    edit: actually it looks like this isn't needed, so I've set the coefficient to 0!
    '''
    return torch.norm(image_as_tensor-original_image_as_tensor, p=2)

def make_comparison_image(original_image, noised_image, save_file_name="comparison_images/output.png"):
    '''
    plots the two images and their classes against each other
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0].transpose(0,2).transpose(0,1).clamp(0,1))
    original_prob, original_class_name, _=top_one_classification(original_image)
    plt.title(f"Original:\nPredicts {original_class_name}\n(p={original_prob:.1%})")

    plt.subplot(1, 2, 2)
    plt.imshow(noised_image[0].detach().transpose(0,2).transpose(0,1).clamp(0,1))
    noised_prob, noised_class_name, _=top_one_classification(noised_image)
    plt.title(f"Adversarial Noise:\nPredicts {noised_class_name}\n(p={noised_prob:.1%})")
    plt.savefig(save_file_name)

def main():
    # Arg Parser written by ChatGPT
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # Integer argument for class number
    parser.add_argument('class_number', type=int, nargs='?', choices=range(1000), default=42,
                        help='An integer for the class number between 0 and 999.')
    
    # File name argument
    parser.add_argument('file_name', type=str, nargs='?', default="cat",
                        help='A file name or a special keyword (bird, cat, shark, snake) pointing to specific files.')
    
    args = parser.parse_args()

    # Mapping special file names to their paths
    special_files = {
        "bird": "inputs/bird.jpg",
        "cat": "inputs/cat.jpg",
        "shark": "inputs/shark.jpg",
        "snake": "inputs/snake.jpg",
    }

    # Check if the file name is one of the special keywords
    if args.file_name in special_files:
        file_path = special_files[args.file_name]
    else:
        file_path = args.file_name

    try:
        original_image=load_image_from_file_path(file_path)
        noised_image=optimize_towards_class(original_image, args.class_number)
    except FileNotFoundError:
        print(f"Could not load the image at {file_path}. Please check file path and try again")

    file_stem=file_path.split('/')[-1].split('.')[0]
    save_file_name=f"comparison_images/{file_stem}_to_class_{args.class_number}"
    make_comparison_image(original_image, noised_image, save_file_name=save_file_name)
    print(f"Check out the comparison image in {save_file_name}!")

if __name__ == "__main__":
    main()


