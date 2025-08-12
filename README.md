## digit classification from audio using a cnn config

### Task
The task of this challenge was to build a lightweight prototype that listens to spoken digits (0–9) and predicts the correct number using the Free Spoken Digit Dataset as the source of .wav files. The assistance of Gemini was used in debugging this project, and ChatGPT was also used for a couple of personal Python questions, but the main ideas of CNN architecture and wav -> spectrogram conversion were inspired by the sources linked below.

### Dataset

The dataset used was the [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset#), an audio version of MNIST and published for cloning on GitHub by user Jakobovski. The folder "recordings" was specifically pulled from this repository for my training purposes.

### .wav to spectrogram conversion

In the fsdd_digit_classifier.py file, our function make_mel_spectrogram_transform performs a Fast-Fourier Transform, which decomposes a wave signal into discrete frequencies.
In my function make_mel_spectrogram_transform, I used tools from the torchaudio library, such as n_fft, hop_length, and n_mels to set these values.

### CNN (convolutional neural network) 
CNNs are a specialised type of artificial neural network specifically designed to process data with a grid-like structure, such as images or audio spectrograms, in our case (Aneta Drhova, 2025). They can automatically learn complex patterns and features from raw input data, making this kind of architecture perfect for ASR and TTS tasks.
For this project, I wrote a "mini CNN" (smallCNN) class from tools in the torch.nn library and used building blocks from their "convolution layers" section in the documentation. This mini model was used to avoid using larger libraries such as keras, which were both incompatible with my current setup and kind of a drain on the CPU.
I also wrote a training loop which included a function used to train one epoch as well as a function that will report to us the total and training losses of the model.

### Training
My model was trained for 20 epochs on 80% of the .wav files in our FSDD dataset, and these were the statistics reported at the last epoch:
Epoch 20  train_loss=0.6862 acc=0.870  val_loss=0.7032 acc=0.832
Done. Best val acc: 0.8316666666666667
I also ran a separate test file on a few individual wav files from the test set, and they gave a correct result every time.

### Bonus: Microphone inference


## Sources
- https://github.com/braydenoneal/neural-audio-classification/blob/master/README.md
- https://ieeexplore-ieee-org.elib.tcd.ie/document/9914596
- https://www.youtube.com/watch?v=dVfmTB8twkg&t=5s
- https://github.com/Armita84/spoken-digits-classification
- https://docs.pytorch.org/docs/stable/nn.html
