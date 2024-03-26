Method for adding adversarial noise to imagenet images.

To use: 
1. Download the repo
2. Make sure all required packages are installed. You can do this by running `pip install -r requirements.txt`
3. Run `main.py` from the command line, with two positional arguments, the desired class number (int 0-999) and a file name to an input image. The file name can also be "bird", "cat", "shark", or "snake" to use the basic images I was testing on. 
4. `main.py` will add adversarial noise to the image to make imagenet think it is a different class. The before/after comparison will be saved to the comparison_images/ directory.


Example usage: 
`python main.py 50 cat`
`python main.py 0 snake`
`python main.py 998 inputs/my_custom_image.jpg`

To avoid errors, make sure the image is in the inputs/ directory.