In this work we described the task of exoplanet detection and the methods applied to this task, both classical and
recent. The transit method was the most popular method used by ML literature, and the most common in terms of
confirmed exoplanet discoveries. Popular works would use both traditional and deep-learning machine learning methods
in order to accomplish detection, but convolutional neural networks appeared to be the most popular.
Our work applied a transformer network to a pre-processed dataset consisting of time-series star flux values from the Kepler Space Telescope experiment; while it correctly classified
most negative cases, it was unable to classify any of the few positive cases.
There is room for future work to improve out model For example, pre-training the transformer with self-supervised learning on the huge volume of Kepler data before supervised
training could help the model learn the features of the data
necessary to make good classifications.
