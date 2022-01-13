import matplotlib.pyplot as plt

from skimage.draw import random_shapes
import numpy as np


def get_3D_gradient_h(width, height):
    result = np.zeros((height, width, 3), dtype=np.float)
    for i in range(3):
        start, stop = np.random.choice(255,2, replace=False)
        result[:,:,i] = np.tile(np.linspace(start, stop, width), (height, 1))
    return result.astype('uint8')

def get_3D_gradient_v(width, height):
    result = np.zeros((height, width, 3), dtype=np.float)
    for i in range(3):
        start, stop = np.random.choice(255,2, replace=False)
        result[:,:,i] = np.tile(np.linspace(start, stop, width), (height, 1)).T
    return result.astype('uint8')

def random_image(size=(256,256)):
    width, height = size
    
    image1, _ = random_shapes((width, height), min_shapes=10, max_shapes=10, min_size=10, allow_overlap=True)
    image2, _ = random_shapes((width, height), min_shapes=10, max_shapes=10, min_size=10, allow_overlap=True)
    
    gradient1 = get_3D_gradient_h(width, height)
    gradient2 = get_3D_gradient_v(width, height)
    
    
    noise = np.random.choice([0,1,2],(width,height,3),p=(0.8,0.1,0.1))
    out = image1 - image2 - gradient1 - gradient2 - noise
    out = np.clip(out,0,255)
    
    return out.astype('uint8')

if __name__ == "__main__":   
    image = random_image()
    
    plt.imshow(image)
    plt.show()
    plt.close()
