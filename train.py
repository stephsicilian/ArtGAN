import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

from Discriminator import Discriminiator
from Generator import Generator

device = torch.device('mps')

def get_random(count):
    return torch.randn(count, device=device)

D = Discriminiator()
D.to(device)
G = Generator()
G.to(device)

directory = 'images'
files = os.listdir(directory)

epochs = 3

for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    for i in range(5700): # number of files being trained with
        if i % 100 == 0:
            print(i)

        image = Image.open(directory + '/' + files[i])
        image = image.resize((128, 128))

        a = np.array(image) / 255.0
        a = a.reshape(128 * 128 * 3)

        image = torch.FloatTensor(a).to(device)

        target = torch.FloatTensor([1.0]).to(device)

        D.train(image, target)
        g_output = G.forward(get_random(300)).detach()

        target = torch.FloatTensor([0.0]).to(device)
        D.train(g_output, target)

        target = torch.FloatTensor([1.0]).to(device)
        G.train(D, get_random(300), target)

torch.save(G.state_dict(), 'gan.pth')