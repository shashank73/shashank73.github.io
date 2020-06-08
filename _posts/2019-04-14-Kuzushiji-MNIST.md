# Kuzushiji (くずし字) Dataset Classification Problem

The Kuzushiji dataset is created by the National Institute of Japanese Literature (NIJL), and is curated by the Center for Open Data in the Humanities (CODH).

The Kuzushiji-MNIST dataset which focuses on Kuzushiji(cursive Japanese) is a classification problem similar to the MNIST dataset. It contains images of the first ten entries from the main Japanese hiragana character groups.

Kuzushiji-MNIST is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images), provided in the original MNIST format as well as a NumPy format.

The images are storred in numpy arrays of 60,000 x 28 x 28 and 10,000 x 28 x 28, respectively. The labels are also stored in two numpy arrays, one for train and another for the test set.

<p align="center">
  <img src="/assets/kmnist/kmnist_examples.png">
</p>
The 10 classes of Kuzushiji-MNIST, with the first column showing each character's modern hiragana counterpart.

## Image Specifications

- Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
- The images are storred in numpy arrays of 60,000 x 28 x 28 and 10,000 x 28 x 28, respectively.
- The labels are also stored in two numpy arrays, one for train and another for the test set.

## Loading the dataset

We will read the two data files containing the 10-class data, KMNIST, similar to MNIST.There are 10 different classes of images.

Aditionally, we will read the character class map for KMNIST, so that we can display the actual characters corresponding to the labels.

Note that .npz files are stored as a dictionary, therefore we need to access its keys to retrieve the numpy array. In this case, it is 'arr_0'.

{% highlight python %}

train_images = np.load('../input/kmnist-train-imgs.npz')['arr_0']
test_images = np.load('../input/kmnist-test-imgs.npz')['arr_0']
train_labels = np.load('../input/kmnist-train-labels.npz')['arr_0']
test_labels = np.load('../input/kmnist-test-labels.npz')['arr_0']

{% endhighlight %}

We can check the dimension of the train and test dataframes by 

{% highlight python %}

print("KMNIST train shape:", train_images.shape)
print("KMNIST test shape:", test_images.shape)

{% endhighlight %}

The output of the command is 
> KMNIST train shape: (60000, 28, 28)

> KMNIST test shape: (10000, 28, 28)

The train set contains 60000 samples of 28x28 greyscale image and the test set contains 10000 images of the same dimension

## Exploratory data analysis (EDA) 

To see the 10 characters present there is a character classmap file present in the dataset we can read it by 
```
character_df = pd.read_csv('kmnist_classmap.csv', encoding = 'utf-8')
character_df.T
```
<p align="center">
  <img src="/assets/kmnist/character.png">
</p>

Now we can check the distribution of the classes or the number of samples of each class present in the dataset

[![Class_Distribution](/assets/kmnist/class_distribution.png)](/assets/kmnist/class_distribution.png)

Since the all the classes contains same number of examples thus the dataset is balanced.

Let's look at some random examples to see the character variation across classes

[![Characters](/assets/kmnist/characters.png)](/assets/kmnist/characters.png)

The character map is not as familiar to us like the numbers (0-9) in MNIST but we can see the character variation of the same id's of image as follows

The image consists of 10 rows each containing a diffrent character and 5 colums representing samples of the same character.

[![Character_Variation](/assets/kmnist/character_variation.png)](/assets/kmnist/character_variation.png)

The variations in characters are quite evident from the above examples.

## Dimensionality reduction and visualization using t-SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) will help us visualize the high dimensional 28x28 vector in two dimension for the classification

It can be invoked on dataframe by 
```
from sklearn.manifold import TSNE
embeddings = TSNE().fit_transform(dataframe)
```

Let's randomly sample 10000 examples from our train dataset for embedding in t-SNE

[![tsne](/assets/kmnist/tsne.png)](/assets/kmnist/tsne.png)

The classes are not easily seprable as in the case of MNIST where the boundary is quite clear that is why this dataset is harder than MNIST.

## Data Preprocessing

Scaling the images in range $$[0, 1]$$
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
flattening the images for train & test from : $$28 * 28$$ to $$ 784$$
```
X_train_flat = train_images.reshape(60000, -1)
X_test_flat = test_images.reshape(10000,-1)
```

## Classification using  various Models

### K-NearestNeighbour(k-NN)
This is going to be our baseline model. We will set k to 4 as used in the KMNIST paper.

```
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
clf.fit(X_train_flat, train_labels)

clf.score(X_test_flat, test_labels)
```

The model performance is very close to the accuracy of ~ 91 % which is present in the dataset benchmark and kmnist paper.

### Support vector classifier (SVC) 
```
from sklearn.svm import SVC

 clf = SVC(C=4.1527, cache_size=200, class_weight=None, coef0=0.0,degree=3,
           gamma=0.0067,kernel='rbf', max_iter=-1, probability=False,
           random_state=SEED, shrinking=True, tol=0.001, verbose=False)

clf.fit(X_train_flat, train_labels)
```
The results are similar to the k-NN with accuracy ~ 92 %

### CNN in Keras

### Model 1 

This is the baseline model used in the benchmark which performs with ~ 94-95 % accuracy depending upon the number of epochs.
```
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```
Let's see the model summary by the inbuilt method 
```
Model.summary()
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               1179776   
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________
```

With 10 epochs the model performance is:
> Test accuracy: 0.9423

Accuracy is close to the value present in the paper, and we have successfully reproduced the results.

### Model 2

The parameters of this model are tuned one at a time by using various models with different hyperparameters. It performs much better than the previous simple model.

```
model = Sequential()
model.add(Conv2D(32,kernel_size=5,activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64,kernel_size=5,activation='relu'))
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```
```
Model.summary()
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)            (None, 24, 24, 32)        832       
_________________________________________________________________
conv2d_1 (Conv2D)          (None, 12, 12, 32)        25632     
_________________________________________________________________
dropout (Dropout)          (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)          (None, 8, 8, 64)          51264     
_________________________________________________________________
conv2d_3 (Conv2D)          (None, 4, 4, 64)          102464    
_________________________________________________________________
dropout_1 (Dropout)        (None, 4, 4, 64)          0         
_________________________________________________________________
flatten_2 (Flatten)        (None, 1024)              0         
_________________________________________________________________
dense (Dense)              (None, 2048)              2099200   
_________________________________________________________________
dropout_2 (Dropout)        (None, 2048)              0         
_________________________________________________________________
dense_1 (Dense)            (None, 10)                20490     
=================================================================
Total params: 2,299,882
Trainable params: 2,299,882
Non-trainable params: 0
_________________________________________________________________

```
At 10 epochs this model is already performing much better than the simple model

> loss: 0.0476 - acc: 0.9851 - val_loss: 0.1527 - val_acc: 0.9606

Training for 35 epochs results in the following score

> loss: 0.0149 - acc: 0.9954 - val_loss: 0.1412 - val_acc: 0.9727

And the final accuracy obtained is 

>Test accuracy: 0.9727

The validation accuracy and loss history of the model per epoch is plotted here

[![Model_performance](/assets/kmnist/model_perf.png)](/assets/kmnist/model_perf.png)

### Model inspection

Let's see the samples where the model incorrectly classified and the true classes

[![Incorrect_Classified](/assets/kmnist/false_classified.png)](/assets/kmnist/false_classified.png)

The incorrectly classified images are hard to classify due to the fact that they look similar to the incorrect class in some cases and in some rare cases the samples look different from both the classes.


### Further Improvements
- Data Augmentation by rotating and zooming
- Further Experiments with neural network architecture
- Ensemble different CNN models

## Refrences

Kuzushiji-MNIST, Dataset Github repo [https://github.com/rois-codh/kmnist](https://github.com/rois-codh/kmnist)

Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, David Ha, Deep Learning for Classical Japanese Literature, [https://arxiv.org/abs/1812.01718](https://arxiv.org/abs/1812.01718)
