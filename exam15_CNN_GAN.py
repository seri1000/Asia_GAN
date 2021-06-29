import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
img_shape = (28,28,1)
epoch = 5000
batch_size = 128
noise = 100
sample_interval = 100

# build generator
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise))
generator_model.add(Reshape((7,7,256)))

generator_model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))#
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
generator_model.add(Activation('tanh'))
generator_model.summary()

#build discriminator
discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size= 3, strides= 2, padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(64, kernel_size= 3, strides= 2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(128, kernel_size= 3, strides= 2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))
discriminator_model.summary()

discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
discriminator_model.trainable = False

(X_train, _), (_,_) = mnist.load_data()
print(X_train.shape)

X_train = X_train / 127.5 - 1 #scaling
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape) #60000, 28, 28

#build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# labeling
real = np.ones((batch_size,1))
print(real)
fake = np.zeros((batch_size,1))
print(fake)

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size) # 0-59999 랜덤값 추출
    real_imgs = X_train[idx] #128장

    z = np.random.normal(0,1,(batch_size,noise)) #페이크 이미지 만들기
    fake_imgs = generator_model.predict(z) #128장

    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    for i in range(5):
        z = np.random.normal(0,1, (batch_size, noise))
        gan_hist = gan_model.train_on_batch(z, real)

    if itr % sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %(itr, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z = np.random.normal(0,1,(row*col, noise))
        fake_imgs = generator_model.predict((z))
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey = True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr))
        plt.savefig(path)
        plt.close()