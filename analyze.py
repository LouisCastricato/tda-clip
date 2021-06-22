import numpy as np
import skdim 
import scipy

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def generate_antipodal(point, set_size=100000, samples=1000):

    def distance(x):
        return np.dot(x, point)**2

    set = skdim.datasets.hyperSphere(set_size, point.shape[0])
    filtered = list(filter(lambda x: x[0]**2 > 0.3, set))
    prob = scipy.special.softmax(list(map(lambda x: distance(x)**2, filtered)))
    print(prob[0:100])
    idx = np.random.choice(np.arange(len(filtered)), samples, replace=False, p=prob)
    #print(idx)
    return set[idx]


pole = [0]*15
pole[0] = 1

pole = np.array(pole)
sample = generate_antipodal(pole)
sample = sample.reshape(1, *sample.shape)



literal = """
size = 2000

#Text embeddings
arr1 = np.load('coco_train2017_text_embeds.npy')
sample_idx = np.random.choice(np.arange(arr1.shape[0]), size = size, replace=False)
sample1 = np.expand_dims(normalized(arr1[sample_idx]), axis=0)


#Image embeddings
arr2 = np.load('coco_train2017_image_embeds.npy')
#sample_idx = np.random.choice(np.arange(arr2.shape[0]), size = size, replace=False)
sample2 = np.expand_dims(normalized(arr2[sample_idx]), axis=0)

sample = np.concatenate([sample1, sample2], axis=1)
print(sample.shape)
"""
from gtda.homology import VietorisRipsPersistence

print("Computing VR")

print(sample.shape)
#import sys
#sys.exit()

VR = VietorisRipsPersistence(metric="cosine", homology_dimensions=[0,1,2])
diagram = VR.fit_transform(sample)

print("Plotting")

from gtda.plotting import plot_diagram
plot_diagram(diagram[0]).write_image("output.png")