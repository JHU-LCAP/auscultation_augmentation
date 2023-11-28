import os
import pickle
import librosa
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
#import torchaudio
import scipy.io as sio
import torch
import glob
from sklearn.cluster import KMeans
from random import randint
import random
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


class PERCHDataset_SMOTE(Dataset):

    def __init__(self,SMOTE_num,k_NN,k_cluster,indices,annotations_file):
        self.SMOTE_num = SMOTE_num
        self.k = k_NN
        self.indices = indices
        self.annotations = pd.read_csv(annotations_file)
        self.new_index = 0
        self.X = self._get_X()
        #self.k_cluster = k_cluster
        #self.k_means = self._get_k_means()
        #self.mean = self._get_mean()
        self.eigen_vec = self._get_eigen_vectors()
        self.k_nearest_indices = self._find_k()
        self.synthetic_samples = self._generate()


    def __len__(self):
        return len(self.indices)*self.SMOTE_num

    def __getitem__(self, index):
        index_actual = index/self.SMOTE_num
        output_mel_spec = self.synthetic_samples[index]
        output_mel_spec = (np.reshape(output_mel_spec.cpu().data.numpy().astype('float32'), (1,32,64)))
        output_mel_spec = torch.FloatTensor(output_mel_spec)
        assert torch.sum(~torch.isnan(output_mel_spec))
        assert torch.sum(~torch.isinf(output_mel_spec))
        output_mel_spec = self._normalise_z_score(output_mel_spec)
        assert torch.sum(~torch.isnan(output_mel_spec))
        assert torch.sum(~torch.isinf(output_mel_spec))
        label = 1
        label_actual = 2
        return output_mel_spec, label, label_actual,self.annotations.iloc[index_actual,9]

    def _get_X(self):
        dtype = torch.FloatTensor
        X = torch.zeros((1,2048)).type(dtype)
        for i in range(len(self.indices)):
            data = np.load(self._get_out_path(self.indices[i]))
            mel_spec = torch.FloatTensor(data['mel_spec'])
            mel_spec = self._normalise_z_score(mel_spec)
            temp = torch.flatten(mel_spec)
            temp = temp[None,:]
            X = torch.cat((X, temp),0)
        X = X[1:,:]
        return X

    def _get_mean(self):
        output = torch.mean(self.X,0)/torch.max(torch.mean(self.X,0))
        assert torch.sum(~torch.isnan(output))
        #output = abs(output)
        #output = self._normalise_min_max(output)
        assert torch.sum(~torch.isnan(output))
        return output

    def _get_k_means(self):
        k_centers = KMeans(n_clusters=self.k_cluster, tol=1e-5, max_iter=300, n_init=10, init='k-means++').fit(self.X)
        output = k_centers.cluster_centers_
        #k_centers = GaussianMixture(n_components=self.k_cluster, tol=1e-5, max_iter=1000,random_state=0).fit(self.X)
        #output = k_centers.means_
        output = torch.FloatTensor(output)
        assert torch.sum(~torch.isnan(output))
        print(output)
        return output


    def _k_neighbors(self, euclid_distance, k):
        nearest_idx = torch.zeros((euclid_distance.shape[0],euclid_distance.shape[0]), dtype = torch.int64)
        idxs = torch.argsort(euclid_distance, dim=1)
        nearest_idx[:,:] = idxs
        return nearest_idx[:,1:k]

    def _find_k(self):
        X = self.X
        k = self.k
        euclid_distance = torch.zeros((len(X),len(X)), dtype = torch.float32)
        for i in range(len(X)):
            dif = (torch.sub(X, X[i]))**2
            dist = torch.sqrt(dif.sum(axis=1))
            euclid_distance[i] = dist
        return self._k_neighbors(euclid_distance,k)

    def _get_nearest_mask(self, array):
        euclid_distance = torch.sub(self.k_means,array)**2
        dist = torch.sqrt(euclid_distance.sum(axis=1))
        val = torch.argmin(dist)
        output = self.k_means[val]
        output = output/torch.max(output)
        #output = abs(output)
        #output = self._normalise_min_max(output)
        return output

    def _get_eigen_vectors(self):
        X = self.X.cpu().data.numpy().astype('float32')
        cov_matrix = np.cov(X,rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        return eig_vecs

    def populate(self, N,i,nnarray,k):
        dtype = torch.FloatTensor
        min_samples = self.X
        while N:
            nn = randint(0, k-2)
            diff = min_samples[nnarray[nn]] - min_samples[i]
            gap = np.random.uniform(size=(diff.shape),low=0,high=1)
            gap = torch.tensor(gap).type(dtype)
            input = min_samples[i] + gap*diff
            #nearest_mask = self._get_nearest_mask(input)
            #nearest_mask = self.mean
            #temp3 = input
            temp = (min_samples[i] + gap * diff).cpu().data.numpy().astype('float32')
            temp1 = np.dot(temp,self.eigen_vec)
            temp2 = np.dot(temp1,np.transpose(self.eigen_vec))
            temp3 = torch.FloatTensor(temp2)
            self.synthetic_samples[self.new_index,:] = temp3
            self.new_index += 1
            N -= 1

    def _generate(self):
        """
            Returns (N/100) * n_minority_samples synthetic minority samples.
                Parameters
                ----------
                min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
                    Holds the minority samples
                N : percetange of new synthetic samples:
                    n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
                k : int. Number of nearest neighbours.
                Returns
                -------
                S : Synthetic samples. array,
                    shape = [(N/100) * n_minority_samples, n_features].
        """
	T = self.X.shape[0]
        N = self.SMOTE_num
        k = self.k
        self.synthetic_samples = torch.zeros(N*T,self.X.shape[1])
        for i in range(self.k_nearest_indices.shape[0]):
            self.populate(N, i, self.k_nearest_indices[i], k)
        return self.synthetic_samples

    def _get_audio_sample_label(self, index):
        label_letter = self.annotations.iloc[index, 5]
        label_array = ['0','W','C','WC']
        label = label_array.index(label_letter)
        if label:
            label = 1
        return label

    def _get_audio_sample_actual_label(self, index):
        label_letter = self.annotations.iloc[index, 5]
        label_array = ['0','W','C','WC']
        label = label_array.index(str(label_letter))
        return label

    def _get_out_path(self, index):
        return self.annotations.iloc[index, 6]


    def _site(self, index):
        return self.annotations.iloc[index, 3]

    def _surety(self, index):
        return self.annotations.iloc[index, 4]

    def _normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def _get_patient_index(self):
        patient_index = self.annotations.iloc[:,7]
        patient_index = np.unique(patient_index)
        return patient_index

    def _normalise_min_max(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        #norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array


    def _normalise_z_score(self, array):
        norm_array = (array - array.mean()) / (array.std())
        #norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array