from __future__ import print_function, division
from keras import preprocessing
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.initializers import RandomNormal
from keras.models import Sequential, Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from keras.backend import mean


import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import cv2
import os

import glob

import numpy as np
import random as rn

from pathlib import Path

"""From the SRGAN paper"""
from settings import Settings
import numpy as np
from scipy.stats import norm
from torch.nn import Module
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor


from settings import Settings
from utility import SummaryWriter, gpu, make_directory_name_unique, MixtureModel, seed_all, norm_squared, square_mean

#import datetime
import re
import select
import sys
from abc import ABC, abstractmethod
import srgan
from data import AgeDataset
from models import Generator, Discriminator
from settings import Settings

# For timestamp of generated library
import time
from pytz import timezone
import pytz
from datetime import datetime
#print('All libraries and classes imported')

class GAN():
  
  use_all_datasets = True
  IMAGE_SIZE = 448 #448 #504
  batch_norm = True
  def __init__(self, settings: Settings):


      def wasserstein_loss(y_true, y_pred):
         return mean(y_true * y_pred)
      IMAGE_SIZE =  448 #504 448
      # Input shape
      self.img_rows = IMAGE_SIZE
      self.img_cols = IMAGE_SIZE
      self.channels = 1
      self.color_dim = 1
      self.img_shape = (self.img_rows, self.img_cols, self.channels)
      self.latent_dim = 100
      self.number_of_outputs = 10

      #SRGAN Parameters
      self.settings = settings
      self.trial_directory: str = None
      self.dnn_summary_writer: SummaryWriter = None
      self.gan_summary_writer: SummaryWriter = None
      self.dataset_class = None
      self.train_dataset: Dataset = None
      self.train_dataset_loader: DataLoader = None
      self.unlabeled_dataset: Dataset = None
      self.unlabeled_dataset_loader: DataLoader = None
      self.validation_dataset: Dataset = None
      self.DNN: Module = None
      self.dnn_optimizer: Optimizer = None
      self.D: Module = None
      self.d_optimizer: Optimizer = None
      self.G: Module = None
      self.g_optimizer: Optimizer = None
      self.signal_quit = False
      self.starting_step = 0

      self.labeled_features = None
      self.unlabeled_features = None
      self.fake_features = None
      self.interpolates_features = None
      self.gradient_norm = None
      self.store_directory = "/content/SRGAN-Test";
      format = "%Y-%m-%d %H:%M:%S %Z%z"
      now_utc = datetime.now(timezone('UTC'))
      swiss_time = now_utc.astimezone(timezone('Europe/Zurich'))
      self.trial_name = "Trial-" + str(swiss_time.strftime(format))
      self.srgan_torch_mode = True
      print('Current time:', str(swiss_time.strftime(format) ))
      print("Created new experiment folder " + str( self.store_directory) + '/'+ str(self.trial_name))


      #The code below was done for Keras/tf, but it took to long -> it is unfinished and implemented in PyTorch
      if(not self.srgan_torch_mode):
        # --- Optimizer ---
        # paper uses 0.0002 since suggested 0.001  by Kingma & Ba was to high
        # optimizer = Adam(0.0002, 0.5) 
        # optimizer = SGD(learning_rate = 0.0005, momentum = 0.9,nesterov = True)
        """Prepares the optimizers of the network."""
        d_lr = self.settings.learning_rate
        g_lr = d_lr
        weight_decay = self.settings.weight_decay #Standard of keras is 0.9, in SRGAN they set it to 0?
        self.d_optimizer = Adam(learning_rate=d_lr, beta_1=weight_decay)
        self.g_optimizer = Adam(learning_rate=g_lr)
        self.dnn_optimizer = Adam(learning_rate=d_lr, beta_1=weight_decay)
        #optimizer = Adam(learning_rate = self.learning_rate, momentum = 0.9,nesterov = True)

        # --- Discriminator ---
        # Build and compile the discriminator
        
        self.discriminator = self.build_discriminator_SRGAN()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=d_optimizer, 
            metrics=['accuracy'])
      

      
        # Build second discriminator
        self.discriminatorNN = self.build_discriminator_SRGAN()
        self.discriminatorNN.compile(loss='binary_crossentropy',
            optimizer=d_optimizer,
            metrics=['accuracy'])


        # --- Generator ----
        # Build the generator
        #self.generator = self.build_generator_DCGAN()
        self.generator = self.build_generator_DCGAN()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)
  def transpose_convolution(self, model, c_out, k_size, stride=2, pad="same", bn=batch_norm, init = RandomNormal(stddev=0.02)):
      """A transposed convolution layer."""
      model.add(Conv2DTranspose(c_out, k_size, stride, padding=pad, kernel_initializer=init))
      
      #layers = [nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad)]
      if bn:
        model.add(BatchNormalization())
        #layers.append(nn.BatchNorm2d(c_out))
      model.add(LeakyReLU(alpha=0.05))
      return model
  def seed_all(self,seed=None):
      """Seed every type of random used by the SRGAN."""
      rn.seed(seed)
      np.random.seed(seed)
      if seed is None:
          seed = int(time.time())
      tf.random.set_seed(seed)
      torch.manual_seed(seed)


  def build_generator_SRGAN_TensorFlow(self,z_dim = 256, image_size = 128, conv_dim=8,pad = "same",init = RandomNormal(stddev=0.02),mom=0.99):
      # Generator from the SRGAN
      #General options
      bn = self.batch_norm
      self.seed_all(13)
      model = Sequential(name="Generator_SRGAN")

      model.add(Dense(z_dim * 7 * 7, activation="relu", input_dim=self.latent_dim))
      model.add(Reshape((7, 7, z_dim)))

      #0. Layer
      model.add(Conv2DTranspose(input_shape=self.img_shape,filters= conv_dim*32,kernel_size= int(self.IMAGE_SIZE/32),strides= 1, padding=pad, kernel_initializer=init))
      #if bn:
      #  model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.05))

      # 1. Layer
      model.add(Conv2DTranspose(filters=conv_dim*16,kernel_size= 4,strides=2, padding=pad, kernel_initializer=init))
      if bn:
        model.add(BatchNormalization(momentum=mom))
      model.add(LeakyReLU(alpha=0.05))

      # 2. Layer
      model.add(Conv2DTranspose(filters=conv_dim*8,kernel_size= 4,strides=2, padding=pad, kernel_initializer=init))
      if bn:
        model.add(BatchNormalization(momentum=mom))
      model.add(LeakyReLU(alpha=0.05))

      # 3. Layer
      model.add(Conv2DTranspose(filters=conv_dim*4,kernel_size= 4,strides=2, padding=pad, kernel_initializer=init))
      if bn:
        model.add(BatchNormalization(momentum=mom))
      model.add(LeakyReLU(alpha=0.05))

      # 4. Layer
      model.add(Conv2DTranspose(filters=conv_dim*2,kernel_size= 4,strides=2, padding=pad, kernel_initializer=init))
      if bn:
        model.add(BatchNormalization(momentum=mom))
      model.add(LeakyReLU(alpha=0.05))

      # 5. Layer
      model.add(Conv2DTranspose(filters=conv_dim,kernel_size= 4,strides=2, padding=pad, kernel_initializer=init))
      if bn:
        model.add(BatchNormalization(momentum=mom))
      model.add(LeakyReLU(alpha=0.05))

      # 6. Layer
      model.add(Conv2DTranspose(filters=self.color_dim,kernel_size= 4,strides=2, padding=pad, kernel_initializer=init))
      #if bn:
      #  model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.05))
      


      self.input_size = z_dim
      model.add(Activation("tanh"))
      model.summary()
      noise = Input(shape=(self.latent_dim,))
      img = model(noise)
      return Model(noise, img)

  def build_generator_DCGAN(self):
  
      model = Sequential()
      model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
      model.add(Reshape((7, 7, 128)))
      model.add(UpSampling2D())
      model.add(Conv2D(128, kernel_size=3, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(UpSampling2D())
      model.add(Conv2D(64, kernel_size=3, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(UpSampling2D())
      model.add(Conv2D(64, kernel_size=3, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(UpSampling2D())
      model.add(Conv2D(48, kernel_size=3, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(UpSampling2D())
      model.add(Conv2D(48, kernel_size=3, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(UpSampling2D())
      model.add(Conv2D(32, kernel_size=3, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
      model.add(Activation("tanh"))
      model.summary()
      # model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy']) 

      noise = Input(shape=(self.latent_dim,))
      img = model(noise)

      return Model(noise, img)

  # ----- DISCRIMINATOR -------
  def build_discriminator_SRGAN_TensorFlow(self,conv_dim=64,mom=0.8,_stride =2,number_of_outputs=1,_alpha=0.05):
    #SRGAN discriminator - very similar to DC
    #Param initialization:
    self.number_of_outputs = number_of_outputs
    bn = self.batch_norm
    self.seed_all(13)

    model = Sequential(name="Discriminator_SRGAN")

    #1. Layer
    model.add(Conv2D(conv_dim,kernel_size=4 , strides=_stride, input_shape=self.img_shape, padding="same"))
    #if bn:
    #    model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=_alpha))

    #2. Layer
    model.add(Conv2D(conv_dim*2 , kernel_size=4, strides=_stride, padding="same"))
    if bn:
        model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=_alpha))

    #3. Layer
    model.add(Conv2D(conv_dim*4 , kernel_size=4, strides=_stride, padding="same"))
    if bn:
        model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=_alpha))

    #4. Layer
    model.add(Conv2D(conv_dim*8 , kernel_size=4, strides=_stride, padding="same"))
    if bn:
        model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=_alpha))

    #5. Layer
    model.add(Conv2D(self.number_of_outputs ,kernel_size= int(self.IMAGE_SIZE / 16), strides=1, padding="same"))
    #if bn:
    #    model.add(BatchNormalization(momentum=mom))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))


    model.summary()

    img = Input(shape=self.img_shape)
    validity = model(img)

    return Model(img, validity)

  def build_discriminator_DCGAN(self):
    #From paper
    model = Sequential(name="Discriminator_DCGAN")

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25)) #Changed from 0.25 to 0.5
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25)) #Changed to 0.5
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))


      

    model.summary()

    img = Input(shape=self.img_shape)
    validity = model(img)

    return Model(img, validity)


    #--- SRGAN TRAINING ---
  def trainSRGAN(self, epochs, batch_size=30, save_interval=50,validate_size_labeled=100,validate_size_scored=300):
      """
      Run the SRGAN training for the experiment.
      """
      #Prepair parameters:
      self.settings.steps_to_run = epochs
      self.start_time = time.time()

      self.trial_directory = os.path.join(self.store_directory, self.trial_name)
      if (self.settings.skip_completed_experiment and os.path.exists(self.trial_directory) and
              '/check' not in self.trial_directory and not self.settings.continue_existing_experiments):
          print('`{}` experiment already exists. Skipping...'.format(self.trial_directory))
          return
      if not self.settings.continue_existing_experiments:
          self.trial_directory = make_directory_name_unique(self.trial_directory)
      else:
          if os.path.exists(self.trial_directory) :
              raise ValueError('Cannot load from path and continue existing at the same time.')
          #elif self.settings.load_model_path is None:
          #    self.settings.load_model_path = self.trial_directory
          elif not os.path.exists(self.trial_directory):
              self.settings.continue_existing_experiments = False
      print(self.trial_directory)
      os.makedirs(os.path.join(self.trial_directory, self.settings.temporary_directory), exist_ok=True)
      self.prepare_summary_writers()
      seed_all(0)

      self.dataset_setup()
      self.model_setup()
      self.prepare_optimizers()
      self.load_models()
      self.gpu_mode()
      self.train_mode()

      self.training_loop( epochs, batch_size, save_interval,validate_size_labeled,validate_size_scored)

      print('Completed {}'.format(self.trial_directory))
      if self.settings.should_save_models:
          self.save_models(step=self.settings.steps_to_run)
  def training_loop(self, epochs, batch_size=30, save_interval=50,validate_size_labeled=100,validate_size_scored=300):
      """Runs the main training loop."""
      labeled_dataset_generator = self.infinite_iter(self.labeled_dataset_loader)
      scored_dataset_generator = self.infinite_iter(self.scored_dataset_loader)
      query_dataset_generator = self.infinite_iter(self.query_dataset_loader)
      step_time_start = time.time()
      print("Starting training")
      for step in range(self.starting_step, epochs):
          print("Current epoch: ", step)
          self.adjust_learning_rate(step)
          # DNN.
          samples = next(labeled_dataset_generator)
          if len(samples) == 2:
              labeled_examples, labels = samples
              labeled_examples, labels = labeled_examples.to(gpu), labels.to(gpu)
          else:
              labeled_examples, primary_labels, secondary_labels = samples
              labeled_examples, labels = labeled_examples.to(gpu), (primary_labels.to(gpu), secondary_labels.to(gpu))
          self.dnn_training_step(labeled_examples, labels, step)
          # GAN.
          unlabeled_examples = next(query_dataset_generator)[0]
          unlabeled_examples = unlabeled_examples.to(gpu)
          self.gan_training_step(labeled_examples, labels, unlabeled_examples, step)

          if self.gan_summary_writer.is_summary_step() or step == self.settings.steps_to_run - 1:
              print('\rStep {}, {}...'.format(step, time.time() - step_time_start), end='')
              step_time_start = time.time()
              self.eval_mode()
              with torch.no_grad():
                  self.validation_summaries(step)
              self.train_mode()
          self.handle_user_input(step)
          if self.settings.save_step_period and step % self.settings.save_step_period == 0 and step != 0:
              self.save_models(step=step)
      print("Time used for training: ", (self.start_time -time.time()))

  def train(self, epochs, batch_size=30, save_interval=50,validate_size_labeled=100,validate_size_scored=300):
      
      batches = int( dataset_labeled.Length/batch_size)
      params={
          "batch_size":batch_size,
          "epochs":epochs}

      experiment.log_parameters(params)
      #Experiment.get_keras_callback(self)
      # Load the dataset
      
      if self.use_all_datasets:

        #labeled
        dataset_labeled =self.loadDataset('labeled')
        batches = int( dataset_labeled_train.Length/batch_size)
        labeled_dataset_generator = self.infinite_iter(self.dataset_labeled)
        dataset_labeled = None
        #dataset_labeled_validate = dataset_labeled.count(validate_size_labeled)
        #dataset_labeled_train = dataset_labeled.skip(validate_size_labeled)         
        #dataset_labeled_train = dataset_labeled_train.batch(batches)
        
        #scored
        dataset_scored = self.loadDataset('scored')
        scored_dataset_generator = self.infinite_iter(self.dataset_scored)
        dataset_scored = None
        #dataset_scored_validate = dataset_scored.count(validate_size_scored)
        #dataset_scored_train = dataset_scored.skip(validate_size_scored)
        #dataset_scored_train = dataset_scored_train(batches)
        
        #Query
        x_train_query_train = self.loadDataset('query')
        query_dataset_generator = self.infinite_iter(self.x_train_query_train)
        #x_train_query_train = dataset_query_train(batches)
        
        
        #print('Label shape:' , label.shape , "First entries", label[1:3])
      else:
        X_train =self.loadDataset('labeledTrue')

      # Rescale -1 to 1
      X_train = X_train / 127.5 - 1.

      # Adversarial ground truths
      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1))
      step_time_start = datetime.datetime.now()
      for epoch in range(epochs):
        if(not self.use_all_datasets):
          # ---------------------
          #  Train Discriminator
          # ---------------------

          # Select a random half of images
          idx = np.random.randint(0, X_train.shape[0], batch_size)
          imgs = X_train[idx]

          # Sample noise and generate a batch of new images
          noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
          gen_imgs = self.generator.predict(noise)

          # Train the discriminator (real classified as ones and generated as zeros)
          d_loss_real = self.discriminator.train_on_batch(imgs, valid)
          d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

          # ---------------------
          #  Train Generator
          # ---------------------

          # Train the generator (wants discriminator to mistake images as real)
          g_loss = self.combined.train_on_batch(noise, valid)

          # Plot the progress
          print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
          # If at save interval => save generated image samples
          if epoch % save_interval == 0:
              self.save_imgs(epoch)
          # ---------------------
          #   COMET: Experiment Logging
          # ---------------------
          experiment.log_metric("Loss discriminator", d_loss[0])
          experiment.log_metric("Accuracy discriminator", 100*d_loss[1])
          experiment.log_metric("Loss generator", g_loss)
          #experiment.log_epoch_end(self, epoch)
        else:
            """Runs the main training loop."""
            print('SRGAN not implemented here - stopping training - and use trainSRGAN')

      print("Time elapsed for the experiment: ",  datetime.datetime.now()-step_time_start)





  #FUnctions from SRGAN Paper
  def adjust_learning_rate(self, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = self.settings.learning_rate * (0.1 ** (step // 100000))
    for param_group in self.dnn_optimizer.param_groups:
        param_group['lr'] = lr

  def infinite_iter(self,dataset):
      """Create an infinite generator from a dataset"""
      while True:
          for examples in dataset:
              yield examples

  def prepare_summary_writers(self):
      """Prepares the summary writers for TensorBoard."""
      self.dnn_summary_writer = SummaryWriter(os.path.join(self.trial_directory, 'DNN'))
      self.gan_summary_writer = SummaryWriter(os.path.join(self.trial_directory, 'GAN'))
      self.dnn_summary_writer.summary_period = self.settings.summary_step_period
      self.gan_summary_writer.summary_period = self.settings.summary_step_period
      self.dnn_summary_writer.steps_to_run = self.settings.steps_to_run
      self.gan_summary_writer.steps_to_run = self.settings.steps_to_run

  
  def dataset_setup(self):
      settings = self.settings
      """Prepares all the datasets and loaders required for the application."""
      IMAGE_SIZE = self.IMAGE_SIZE
      print("Loading dataset...")
      print("Loading dataset: labeled 1 ")
      self.labeled_dataset = AgeDataset(None,category='labeled',IMAGE_SIZE=IMAGE_SIZE, start=0, end=settings.labeled_end,
                                      seed=settings.labeled_dataset_seed, batch_size=settings.batch_size)

      self.labeled_dataset_loader = DataLoader(self.labeled_dataset, batch_size=settings.batch_size, shuffle=True,
                                              pin_memory=self.settings.pin_memory,
                                              num_workers=settings.number_of_data_workers, drop_last=True)
      print("Loading dataset: scored 1 ")
      self.scored_dataset = AgeDataset(None,category='scored',IMAGE_SIZE=IMAGE_SIZE, start=0, end=settings.scored_end,
                                      seed=settings.labeled_dataset_seed, batch_size=settings.batch_size)
      
      self.scored_dataset_loader = DataLoader(self.scored_dataset, batch_size=settings.batch_size, shuffle=True,
                                              pin_memory=self.settings.pin_memory,
                                              num_workers=settings.number_of_data_workers, drop_last=True)
      print("Loading dataset: query 1 ")
      self.query_dataset = AgeDataset(None,category='query',IMAGE_SIZE=IMAGE_SIZE, start=0,
                                          end=settings.query_end,
                                          seed=settings.query_dataset_seed, batch_size=settings.batch_size)
      
      self.query_dataset_loader = DataLoader(self.query_dataset , batch_size=settings.batch_size, shuffle=True,
                                                  pin_memory=self.settings.pin_memory,
                                                  num_workers=settings.number_of_data_workers, drop_last=True)

      print("Loading dataset: validation labeled ")
      self.labeled_validation_dataset = AgeDataset(None,category='labeled',IMAGE_SIZE=IMAGE_SIZE, start=-settings.validation_dataset_size,
                                            end=None,
                                            seed=settings.labeled_dataset_seed, batch_size=settings.batch_size)
      print("Loading dataset: validation scored ")
      self.scored_validation_dataset = AgeDataset(None,category='scored',IMAGE_SIZE=IMAGE_SIZE, start=-settings.validation_dataset_size,
                                            end=None,
                                            seed=settings.labeled_dataset_seed, batch_size=settings.batch_size)
      print("Finished loading")
  def clearDatasets(self):
    self.labeled_dataset = []
    self.labeled_dataset_loader = []
    self.scored_dataset = []
    self.scored_dataset_loader = []
    self.query_dataset = []
    self.query_dataset_loader = []

  def model_setup(self):
      self.G = Generator()
      self.D = Discriminator()
      self.DNN = Discriminator()
  @abstractmethod
  def validation_summaries(self, step: int):
        """Prepares the summaries that should be run for the given application."""
        pass

  @staticmethod
  def labeled_loss_function(predicted_labels, labels, order=2):
        """Calculate the loss from the label difference prediction."""
        return (predicted_labels - labels).abs().pow(order).mean()
  def prepare_optimizers(self):
        """Prepares the optimizers of the network."""
        d_lr = self.settings.learning_rate
        g_lr = d_lr
        weight_decay = self.settings.weight_decay
        self.d_optimizer = Adam(self.D.parameters(), lr=d_lr, weight_decay=weight_decay)
        self.g_optimizer = Adam(self.G.parameters(), lr=g_lr)
        self.dnn_optimizer = Adam(self.DNN.parameters(), lr=d_lr, weight_decay=weight_decay)
  def evaluate(self):
        """Evaluates the model on the test dataset (needs to be overridden by subclass)."""
        self.model_setup()
        self.load_models()
        self.eval_mode()

  def feature_distance_loss(self, base_features, other_features, distance_function=None):
        """Calculate the loss based on the distance between feature vectors."""
        if distance_function is None:
            distance_function = self.settings.matching_distance_function
        base_mean_features = base_features.mean(0)
        other_mean_features = other_features.mean(0)
        if self.settings.normalize_feature_norm:
            epsilon = 1e-5
            base_mean_features = base_mean_features / (base_mean_features.norm() + epsilon)
            other_mean_features = other_features / (other_mean_features.norm() + epsilon)
        distance_vector = distance_function(base_mean_features - other_mean_features)
        return distance_vector

  @property
  def inference_network(self):
      """The network to be used for inference."""
      return self.D

  def inference_setup(self):
      """
      Sets up the network for inference.
      """
      self.model_setup()
      self.load_models(with_optimizers=False)
      self.gpu_mode()
      self.eval_mode()

  def inference(self, input_):
      """
      Run the inference for the experiment.
      """
      raise NotImplementedError


  def load_models(self, with_optimizers=True):
        """Loads existing models if they exist at `self.settings.load_model_path`."""
        if self.settings.load_model_path:
            latest_model = None
            model_path_file_names = os.listdir(self.settings.load_model_path)
            for file_name in model_path_file_names:
                match = re.search(r'model_?(\d+)?\.pth', file_name)
                if match:
                    latest_model = self.compare_model_path_for_latest(latest_model, match)
            latest_model = None if latest_model is None else latest_model.group(0)
            if not torch.cuda.is_available():
                map_location = 'cpu'
            else:
                map_location = None
            if latest_model:
                model_path = os.path.join(self.settings.load_model_path, latest_model)
                loaded_model = torch.load(model_path, map_location)
                self.DNN.load_state_dict(loaded_model['DNN'])
                self.D.load_state_dict(loaded_model['D'])
                self.G.load_state_dict(loaded_model['G'])
                if with_optimizers:
                    self.dnn_optimizer.load_state_dict(loaded_model['dnn_optimizer'])
                    self.optimizer_to_gpu(self.dnn_optimizer)
                    self.d_optimizer.load_state_dict(loaded_model['d_optimizer'])
                    self.optimizer_to_gpu(self.d_optimizer)
                    self.g_optimizer.load_state_dict(loaded_model['g_optimizer'])
                    self.optimizer_to_gpu(self.g_optimizer)
                print('Model loaded from `{}`.'.format(model_path))
                if self.settings.continue_existing_experiments:
                    self.starting_step = loaded_model['step'] + 1
                    print(f'Continuing from step {self.starting_step}')
  def gpu_mode(self):
        """
        Moves the networks to the GPU (if available).
        """
        self.D.to(gpu)
        self.DNN.to(gpu)
        self.G.to(gpu)

  def train_mode(self):
        """
        Converts the networks to train mode.
        """
        self.D.train()
        self.DNN.train()
        self.G.train()
  def dnn_training_step(self, examples, labels, step):
        """Runs an individual round of DNN training."""
        self.DNN.apply(disable_batch_norm_updates)  # No batch norm
        self.dnn_summary_writer.step = step
        self.dnn_optimizer.zero_grad()
        dnn_loss = self.dnn_loss_calculation(examples, labels)
        dnn_loss.backward()
        self.dnn_optimizer.step()
        # Summaries.
        if self.dnn_summary_writer.is_summary_step():
            self.dnn_summary_writer.add_scalar('Discriminator/Labeled Loss', dnn_loss.item())
            if hasattr(self.DNN, 'features') and self.DNN.features is not None:
                self.dnn_summary_writer.add_scalar('Feature Norm/Labeled', self.DNN.features.norm(dim=1).mean().item())
  def evaluation_epoch(self, settings, network, dataset, summary_writer, summary_name, comparison_value=None):
        """Runs the evaluation and summaries for the data in the dataset."""
        dataset_loader = DataLoader(dataset, batch_size=settings.batch_size)
        predicted_ages, ages = np.array([]), np.array([])
        for images, labels in dataset_loader:
            batch_predicted_ages = self.images_to_predicted_ages(network, images.to(gpu))
            batch_predicted_ages = batch_predicted_ages.detach().to('cpu').view(-1).numpy()
            ages = np.concatenate([ages, labels])
            predicted_ages = np.concatenate([predicted_ages, batch_predicted_ages])
        mae = np.abs(predicted_ages - ages).mean()
        summary_writer.add_scalar('{}/MAE'.format(summary_name), mae, )
        mse = (np.abs(predicted_ages - ages) ** 2).mean()
        summary_writer.add_scalar('{}/MSE'.format(summary_name), mse, )
        if comparison_value is not None:
            summary_writer.add_scalar('{}/Ratio MAE GAN DNN'.format(summary_name), mae / comparison_value, )
        return mae
  def gan_training_step(self, labeled_examples, labels, unlabeled_examples, step):
        """Runs an individual round of GAN training."""
        # Labeled.
        self.D.apply(disable_batch_norm_updates)  # No batch norm
        self.gan_summary_writer.step = step
        self.d_optimizer.zero_grad()
        labeled_loss = self.labeled_loss_calculation(labeled_examples, labels)
        labeled_loss.backward()
        # Unlabeled.
        # self.D.apply(disable_batch_norm_updates)  # Make sure only labeled data is used for batch norm statistics
        unlabeled_loss = self.unlabeled_loss_calculation(labeled_examples, unlabeled_examples)
        unlabeled_loss.backward()
        # Fake.
        z = torch.tensor(MixtureModel([norm(-self.settings.mean_offset, 1),
                                       norm(self.settings.mean_offset, 1)]
                                      ).rvs(size=[unlabeled_examples.size(0),
                                                  self.G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = self.G(z)
        fake_loss = self.fake_loss_calculation(unlabeled_examples, fake_examples)
        fake_loss.backward()
        # Gradient penalty.
        gradient_penalty = self.gradient_penalty_calculation(fake_examples, unlabeled_examples)
        gradient_penalty.backward()
        # Discriminator update.
        self.d_optimizer.step()
        # Generator.
        if step % self.settings.generator_training_step_period == 0:
            self.g_optimizer.zero_grad()
            z = torch.randn(unlabeled_examples.size(0), self.G.input_size).to(gpu)
            fake_examples = self.G(z)
            generator_loss = self.generator_loss_calculation(fake_examples, unlabeled_examples)
            generator_loss.backward()
            self.g_optimizer.step()
            if self.gan_summary_writer.is_summary_step():
                self.gan_summary_writer.add_scalar('Generator/Loss', generator_loss.item())
        # Summaries.
        if self.gan_summary_writer.is_summary_step():
            self.gan_summary_writer.add_scalar('Discriminator/Labeled Loss', labeled_loss.item())
            self.gan_summary_writer.add_scalar('Discriminator/Unlabeled Loss', unlabeled_loss.item())
            self.gan_summary_writer.add_scalar('Discriminator/Fake Loss', fake_loss.item())
            self.gan_summary_writer.add_scalar('Discriminator/Gradient Penalty', gradient_penalty.item())
            self.gan_summary_writer.add_scalar('Discriminator/Gradient Norm', self.gradient_norm.mean().item())
            if self.labeled_features is not None and self.unlabeled_features is not None:
                self.gan_summary_writer.add_scalar('Feature Norm/Labeled',
                                                   self.labeled_features.mean(0).norm().item())
                self.gan_summary_writer.add_scalar('Feature Norm/Unlabeled',
                                                   self.unlabeled_features.mean(0).norm().item())
        # self.D.apply(enable_batch_norm_updates)  # Only labeled data used for batch norm running statistics
  def dnn_loss_calculation(self, labeled_examples, labels):
      """Calculates the DNN loss."""
      predicted_labels = self.DNN(labeled_examples)
      labeled_loss = self.labeled_loss_function(predicted_labels, labels, order=self.settings.labeled_loss_order)
      labeled_loss *= self.settings.labeled_loss_multiplier
      return labeled_loss

  def labeled_loss_calculation(self, labeled_examples, labels):
      """Calculates the labeled loss."""
      predicted_labels = self.D(labeled_examples)
      self.labeled_features = self.D.features
      labeled_loss = self.labeled_loss_function(predicted_labels, labels, order=self.settings.labeled_loss_order)
      labeled_loss *= self.settings.labeled_loss_multiplier
      return labeled_loss

  def unlabeled_loss_calculation(self, labeled_examples: Tensor, unlabeled_examples: Tensor):
      """Calculates the unlabeled loss."""
      _ = self.D(labeled_examples)
      self.labeled_features = self.D.features
      _ = self.D(unlabeled_examples)
      self.unlabeled_features = self.D.features
      unlabeled_loss = self.feature_distance_loss(self.unlabeled_features, self.labeled_features)
      unlabeled_loss *= self.settings.matching_loss_multiplier
      unlabeled_loss *= self.settings.srgan_loss_multiplier
      return unlabeled_loss

  def fake_loss_calculation(self, unlabeled_examples: Tensor, fake_examples: Tensor):
      """Calculates the fake loss."""
      _ = self.D(unlabeled_examples)
      self.unlabeled_features = self.D.features
      _ = self.D(fake_examples.detach())
      self.fake_features = self.D.features
      fake_loss = self.feature_distance_loss(self.unlabeled_features, self.fake_features,
                                              distance_function=self.settings.contrasting_distance_function)
      fake_loss *= self.settings.contrasting_loss_multiplier
      fake_loss *= self.settings.srgan_loss_multiplier
      return fake_loss

  def gradient_penalty_calculation(self, fake_examples: Tensor, unlabeled_examples: Tensor) -> Tensor:
      """Calculates the gradient penalty from the given fake and real examples."""
      alpha_shape = [1] * len(unlabeled_examples.size())
      alpha_shape[0] = self.settings.batch_size
      alpha = torch.rand(alpha_shape, device=gpu)
      interpolates = (alpha * unlabeled_examples.detach().requires_grad_() +
                      (1 - alpha) * fake_examples.detach().requires_grad_())
      interpolates_loss = self.interpolate_loss_calculation(interpolates)
      gradients = torch.autograd.grad(outputs=interpolates_loss, inputs=interpolates,
                                      grad_outputs=torch.ones_like(interpolates_loss, device=gpu),
                                      create_graph=True)[0]
      gradient_norm = gradients.view(unlabeled_examples.size(0), -1).norm(dim=1)
      self.gradient_norm = gradient_norm
      norm_excesses = torch.max(gradient_norm - 1, torch.zeros_like(gradient_norm))
      gradient_penalty = (norm_excesses ** 2).mean() * self.settings.gradient_penalty_multiplier
      return gradient_penalty

  def interpolate_loss_calculation(self, interpolates):
      """Calculates the interpolate loss for use in the gradient penalty."""
      _ = self.D(interpolates)
      self.interpolates_features = self.D.features
      return self.interpolates_features.norm(dim=1)

  def generator_loss_calculation(self, fake_examples, unlabeled_examples):
      """Calculates the generator's loss."""
      _ = self.D(fake_examples)
      self.fake_features = self.D.features
      _ = self.D(unlabeled_examples)
      detached_unlabeled_features = self.D.features.detach()
      generator_loss = self.feature_distance_loss(detached_unlabeled_features, self.fake_features)
      generator_loss *= self.settings.matching_loss_multiplier
      return generator_loss


  def save_imgs(self, epoch):
      path ='./GeneratedImages'
      Path(path).mkdir(parents=True, exist_ok=True)
      r, c = 5, 5
      noise = np.random.normal(0, 1, (r * c, self.latent_dim))
      gen_imgs = self.generator.predict(noise)

      # Rescale images 0 - 1
      gen_imgs = 0.5 * gen_imgs + 0.5

      fig, axs = plt.subplots(r, c)
      cnt = 0
      for i in range(r):
          for j in range(c):
              axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
              axs[i,j].axis('off')
              cnt += 1
      nameFig = "/cosmology_%d.png" % epoch
      fig.savefig(path + nameFig, dpi=300)
      
      #Comet logging:
      experiment.log_image(self, path + nameFig)

      plt.close()


  def loadDataset(self,category='labeled'):
    IMAGE_SIZE = self.IMAGE_SIZE
    """
    In this project, you are given a mix of realistic cosmology images, corrupted cosmology images, and images which show other concepts like landscapes.

    Most of the images have been scored according to their similarity to the concept of a prototypical 'cosmology image' according to our data-set. A similarity score like 2.61 means that the image almost co-incides with the prototypical cosmology image, a low similarity score like 0.00244 means that the image is a poor representative of a cosmology image -- probably because it has a different subject like a landscape or is corrupted. You can assume that similarity scores are valued in the interval [0.0, 8.0].

    Beyond the scored images you are a given a smaller set of labeled images for which you can assume that they are drawn from the same distribution as the scored images. For these images, you are not given the similarity score, but you get labels: 1.0 means that the image is a real cosmology image, whereas 0.0 means it has been corrupted or shows another subject.
    """    
    # Notes: You can assume that the distribution of real cosmology images, corrupted cosmology images and other images is roughly equal in the 3 folders.
    # The file <query_example.csv> shows you the correct submission format, simply replace the second column with your predictions and submit.
    # Image sizes
    # All of the images are grey-scale and have size 1000x1000 pixels. For model building purposes feel free to use either the full size images or crop them to derive smaller images from them,
    # if it is more convenient.


    # --- Load the dataset ---

    data_path = './cosmology_aux_data_170429/cosmology_aux_data_170429/'
    # Local data path
    #data_path = '/Users/tobiaselmiger/GitRepository/CIL/03_Data/cosmology_aux_data_170429/cosmology_aux_data_170429/'
    x_images = []


    # Quere Images:
    # Lastly, the folder query contains 1200 images which are unscored. You are required to predict the similarity scores of these images and submit these predictions as a CSV file.
    # 600 random images are scored on the public leaderboard, and the remaining 600 images are held-out. You cannot track your scores on these images, but they will be used to determine your final score.
    if (category == 'query'):
        img_dir = data_path + category + '/' # Directory of the images
        print('Search path: '+ str(img_dir))
        #!ls(img_dir)
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path)
        print('Length:',len(files))
        for f1 in files:
            #print('f1:', f1)
            img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
            #print('img:',img)
            resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            x_images.append(np.array(resized))
            
            

    # Labeled images:
    # The folder labeled contains 1200 images. The corresponding labels of each image can be found in <labeled.csv> at the top-level.
    # Make proper use of the scores and labels to build your generative model and learn the similarity function. 
    # Note: We use only the correct labeled ones
    elif ((category == 'labeledTrue')):
        category = "labeled"
        df_labeled = pd.read_csv(data_path + category + '.csv') 
        for index, row in tqdm(df_labeled.iterrows(), total=df_labeled.shape[0]):
          # Only loads the images with score 1 (which are galaxy images)
          if row['Actual'] == 1.0:
              img = cv2.imread(data_path + category + '/' + str(int(row['Id'])) + '.png', cv2.IMREAD_GRAYSCALE)
              resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
              x_images.append(np.array(resized).reshape((IMAGE_SIZE, IMAGE_SIZE, 1)))
        category = 'Labeled only true'
        print('Cosmology size of dataset ', category, ': ', len(x_images), 'shape: ', x_images[1].shape)  
        dataset = np.array(x_images)
        return dataset
    elif ((category == 'labeled')):
        df_labeled = pd.read_csv(data_path + category + '.csv') 
        for index, row in tqdm(df_labeled.iterrows(), total=df_labeled.shape[0]):
          # Loads all images
          img = cv2.imread(data_path + category + '/' + str(int(row['Id'])) + '.png', cv2.IMREAD_GRAYSCALE)
          resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
          x_images.append(np.array(resized).reshape((IMAGE_SIZE, IMAGE_SIZE, 1)))
        scores = df_labeled['Actual']
        print('Cosmology Size of dataset ', category, ': ', len(x_images), 'shape: ', x_images[1].shape) 
        dataset_img = np.array(x_images)
        #Normalize:
        dataset_img = dataset_img/ 127.5 - 1.
        dataset = tf.data.Dataset.from_tensor_slices((dataset_img, scores))
        return dataset

    # Scored images:
    # The folder scored contains 9600 images. The majority of these images are realistic cosmology images, whereas some are images of other subjects or corrupted cosmology images.
    # The corresponding similarity scores of each image can be found in <scored.csv> at the top-level.
    elif ((category == 'scored')):
        df_labeled = pd.read_csv(data_path + category + '.csv') 
        for index, row in tqdm(df_labeled.iterrows(), total=df_labeled.shape[0]):
          # Loads all images
          img = cv2.imread(data_path + category + '/' + str(int(row['Id'])) + '.png', cv2.IMREAD_GRAYSCALE)
          resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
          x_images.append(np.array(resized).reshape((IMAGE_SIZE, IMAGE_SIZE, 1)))
        scores = df_labeled['Actual']
        print('Cosmology Size of dataset ', category, ': ', len(x_images), 'shape: ', x_images[1].shape) 
        
        dataset_img = np.array(x_images)
        #Normalize:
        dataset_img = dataset_img/ 127.5 - 1.
        dataset = tf.data.Dataset.from_tensor_slices((dataset_img, scores))
        return dataset

    else:
        warnings.warn('Unkown category')

    print('Cosmology size of dataset ', category, ': ', len(x_images), 'shape: ', x_images[1].shape)  
    dataset = np.array(x_images)
    dataset = dataset/ 127.5 - 1.
    return dataset

      

#testLabeled = loadDataset('labeled')

