from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

#PARAMETERS
small_path = "./peppers-small.tiff"
large_path = "./peppers-large.tiff"
number_of_colors = 16
max_iter=30

def init_centroids(points, count):
  if count>len(points): count=len(points)
  rng=np.random.default_rng()
  return np.array(rng.choice(np.array(points), count))

def assign_centroids(points, centroids):
  assigned_centroids=[] 
  for cr_point in points:
    best_centroid = 0
    for i in range(1, len(centroids)):
      if np.linalg.norm(centroids[i]-cr_point) < np.linalg.norm(centroids[best_centroid]-cr_point):
        best_centroid=i
    assigned_centroids.append(best_centroid)
  return np.array(assigned_centroids)

def update_centroids(points, centroids):
  updated_centroids=np.zeros(centroids.shape)
  colors=assign_centroids(points, centroids)
  for i in range(len(centroids)):
    updated_centroids[i]=np.mean(points[(colors.reshape(-1)==i)], axis=0).astype("int")
  delta=np.linalg.norm(updated_centroids-centroids)
  return (updated_centroids, delta)

if __name__ == "__main__":
  small_image = imread(small_path)  
  points = []
  for i in range(small_image.shape[0]):
    for j in range(small_image.shape[1]):
      points.append(np.array(small_image[i, j]))
  points=np.array(points)
  centroids=init_centroids(points, number_of_colors)
  for elem in centroids: print(elem)
  it=0
  while it<max_iter:
    centroids, delta=update_centroids(points, centroids)
    #print(25*"=")
    #for elem in centroids: print(elem)
    print(delta)
    it+=1

  for elem in centroids: print(elem)
  large_image=imread(large_path)
  compressed_image=np.copy(large_image)
  for i in range(large_image.shape[0]):
    for j in range(large_image.shape[1]):
      color=large_image[i, j]
      compressed_image[i, j] = centroids[np.argmin(np.linalg.norm(centroids - color, axis=1))]

  figure_idx=0
  plt.figure(figure_idx)
  figure_idx += 1
  plt.imshow(large_image)
  plt.title('Original large image')
  plt.axis('off')
  savepath = os.path.join('.', 'orig_large.png')
  plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')
  plt.figure(figure_idx)
  figure_idx += 1
  plt.imshow(compressed_image.astype(int))
  plt.title('compressed large image')
  plt.axis('off')
  savepath = os.path.join('.', 'compressed.png')
  plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')