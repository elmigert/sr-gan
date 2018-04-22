"""
Code for accessing the data in the database easily.
"""
import os
import torch
from urllib.request import urlretrieve
import imageio
import tarfile
import numpy as np
from skimage import transform, color
from torch.utils.data import Dataset
from scipy.io import loadmat
from datetime import datetime


class AgeDataset(Dataset):
    def __init__(self, start=None, end=None, gender_filter=None):
        mat_path = '../imdb_wiki_data/imdb_crop/imdb.mat'
        self.dataset_path = '../imdb_wiki_data/imdb_crop/'
        image_paths, dobs, genders, time_stamps, face_scores, second_face_scores, ages = get_database_meta(mat_path)
        indexes = np.where(face_scores > 1.0)
        if gender_filter is not None:
            gender_indexes = np.where(genders == gender_filter)
            indexes = np.intersect1d(indexes, gender_indexes)
        image_paths = image_paths[indexes]
        ages = ages[indexes]
        indexes = []
        for index, image_path in enumerate(image_paths):
            try:
                image = imageio.imread(os.path.join(self.dataset_path, image_path))
            except FileNotFoundError:
                continue
            if image.shape[0] > 256 and image.shape[1] > 256 and abs(image.shape[0] - image.shape[1]) < 5:
                indexes.append(index)
                if end is not None and len(indexes) > end:
                    break
        indexes = indexes[start:end]
        self.image_paths = np.copy(image_paths[indexes])
        self.ages = ages[indexes].astype(np.float32)
        self.length = self.ages.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = imageio.imread(os.path.join(self.dataset_path, image_path))
        image = transform.resize(image, (128, 128))
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        age = self.ages[idx]
        age = torch.from_numpy(np.array([age]))
        return image, age


def calculate_age(taken, date_of_birth):
    birth_datetime = datetime.fromordinal(max(int(date_of_birth) - 366, 1))
    # Assume the photo was taken in the middle of the year
    if birth_datetime.month < 7:
        return taken - birth_datetime.year
    else:
        return taken - birth_datetime.year - 1


def get_database_meta(mat_path, database_name='imdb', shuffle=True):
    meta = loadmat(mat_path)
    image_paths = np.array(meta[database_name][0, 0]["full_path"][0].tolist())[:, 0]
    dobs = meta[database_name][0, 0]["dob"][0]
    genders = meta[database_name][0, 0]["gender"][0]
    time_stamps = meta[database_name][0, 0]["photo_taken"][0]
    face_scores = meta[database_name][0, 0]["face_score"][0]
    second_face_scores = meta[database_name][0, 0]["second_face_score"][0]
    ages = np.array([calculate_age(time_stamps[i], dobs[i]) for i in range(len(dobs))])
    if shuffle:
        p = np.random.permutation(len(ages))
        image_paths, dobs, genders, time_stamps, face_scores, second_face_scores, ages = (image_paths[p], dobs[p],
            genders[p], time_stamps[p], face_scores[p], second_face_scores[p], ages[p])
    return image_paths, dobs, genders, time_stamps, face_scores, second_face_scores, ages


def download_database():
    database_directory = '../imdb_wiki_data'
    if os.path.exists(database_directory):
        print('imdb-wiki database already seems to exist. Delete it for fresh download.')
        return
    os.makedirs(database_directory)
    print('Downloading...')
    urlretrieve('https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar',
                os.path.join(database_directory, 'imdb_crop.tar'))
    with tarfile.open(os.path.join(database_directory, 'imdb_crop.tar')) as tar_file:
        tar_file.extractall(path=database_directory)
        os.remove(os.path.join(database_directory, 'imdb_crop.tar'))
    print('Done.')


if __name__ == '__main__':
    download_database()
