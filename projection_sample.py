import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.spatial import distance
from tqdm import tqdm

G = nx.karate_club_graph()

data = nx.adjacency_matrix(G).todense()

N = data.shape[0]
proj_plane = np.random.uniform(size=(N,2))

def compute_energy(data, proj_plane):
    E = 0
    for i in range(N):
        for j in range(N):
            hd = distance.hamming(data[i,:], data[j,:])
            ed = distance.euclidean(proj_plane[i,:], proj_plane[j,:])
            E += (ed-hd)**2
    return E
  
if __name__ == '__main__':  
  step = 0.01
  
  temp = 1000
  base = 0.95
  T = lambda t: temp * base**(t)

  curr_pos = proj_plane.copy()
  curr_E = compute_energy(data, curr_pos)


  accepted_energy = np.zeros(epochs)
  rejected_energy = np.zeros(epochs)
  
  epochs = 10000
  for t in tqdm(range(epochs)):
      rand_angle = np.random.uniform(
          low=0,high=2*np.pi,size=N
      )
      delta = step*np.array([
          np.cos(rand_angle),
          np.sin(rand_angle),
      ]).reshape(-1,2)

      new_pos = curr_pos + delta
      new_E = compute_energy(data, new_pos)
      delta_E = (new_E - curr_E)
      if delta_E > 0:
          p = np.exp(-delta_E/T(t))
          #print(delta_E, p)
          draw = np.random.binomial(1, p)
          if draw == 1:
            #print('accept')
            curr_pos = new_pos
            curr_E = new_E
            accepted_energy[t] = new_E
        else:
            rejected_energy[t] = new_E
            #print('decline')
    else:
        #print('accept 2')
        curr_E = new_E
        accepted_energy[t] = new_E
        curr_pos = new_pos
