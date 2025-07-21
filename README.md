FashionMNIST Classifier from Scratch (NumPy)

This project is a complete implementation of a neural network classifier for the FashionMNIST dataset, built entirely from scratch using NumPy. It is intended as a learning exercise to understand the fundamentals of machine learning and backpropagation without using high-level frameworks and libraries.

Features
	• No external ML libraries (pure NumPy implementation)
	• Manual forward and backward propagation
	• Softmax output layer for multi-class classification
	• Cross-entropy loss function
	• One-hot encoding for labels
	• Evaluation on test data with classification accuracy

Files
	• ClsfierWithNumpyFashMINST.ipynb: Jupyter notebook containing all code for data loading, preprocessing, model implementation, training, and evaluation.

Dataset
	• Dataset: FashionMNIST
	• Source: Zalando Research
	• Classes:
	    1.T-shirt/top
	    2.Trouser
	    3.Pullover
	    4.Dress
	    5.Coat
	    6.Sandal
	    7.Shirt
	    8.Sneaker
	    9.Bag
	    10.Ankle boot

Requirements
Install the minimal requirements with:
```python
pip install numpy matplotlib tensorflow
```
Note: tensorflow is used only for loading the FashionMNIST dataset via tensorflow.keras.datasets. The actual model is built without it.

How to Run
1. Clone this repository:
```
git clone https://github.com/yourusername/fashionmnist-numpy.git
cd fashionmnist-numpy
```
2. Open the notebook:
```
jupyter notebook ClsfierWithNumpyFashMINST.ipynb
```
3. Run all cells to train and evaluate the model.


Model Overview
	• Input: 784-dimensional vector (flattened 28x28 image)
	• Hidden Layers: 2 hidden layers (128 & 64 units respectively)
	• Activation: ReLU
	• Output Layer: 10 neurons, softmax activation
	• Loss: Cross-Entropy
	• Optimizer: Adam

Results
	• Training for 60 epochs
	• Final test accuracy (approximate): ~85–87%
(Results may vary slightly depending on initialization and random seeds)

Learning Outcomes

This project was designed to:
	• Explore the inner mechanics of neural networks
	• Implement forward and backward propagation from scratch
	• Understand matrix calculus, gradient descent, and loss optimization
	• Reinforce Python and NumPy proficiency

