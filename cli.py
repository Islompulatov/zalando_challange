import torch
from model_theo import neuralnet
import time
import argparse

# Loading model
state_dict = torch.load('model.pth')

neuralnet.load_state_dict(state_dict)


# Setting up the CLI
parser = argparse.ArgumentParser(
    description='Predict the images in Fashion MNST')

parser.add_argument('file_path', type=str,
                    help='Path to samples images')

args = parser.parse_args()

path_to_images = args.file_path

# Loading records
images = torch.load(path_to_images)

# from int to article names
article_name = {0: 'T-shirt/top',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle Boot'}

# Predictions
for img in images:

    pred = torch.argmax(neuralnet(img.view(1, -1)), dim=1)
    print(article_name[pred.item()])

    # Break (to simulate readings every 3 secondes)
    time.sleep(3)
