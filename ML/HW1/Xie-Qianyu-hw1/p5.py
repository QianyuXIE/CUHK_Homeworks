import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import random

def load_mat(path, d=16):
    data = scipy.io.loadmat(path)['zip']
    size = data.shape[0]
    y = data[:, 0].astype('int')
    X = data[:, 1:].reshape(size, d, d)
    return X, y

def cal_intensity(X):
    """
    X: (n, d), input data
    return intensity: (n, 1)
    """
    n = X.shape[0]
    return np.mean(X.reshape(n, -1), 1, keepdims=True)

def cal_symmetry(X):
    """
    X: (n, d), input data
    return symmetry: (n, 1)
    """
    n, d = X.shape[:2]
    Xl = X[:, :, :int(d/2)]
    Xr = np.flip(X[:, :, int(d/2):], -1)
    abs_diff = np.abs(Xl-Xr)
    return np.mean(abs_diff.reshape(n, -1), 1, keepdims=True)

def cal_feature(data):
    intensity = cal_intensity(data)
    symmetry = cal_symmetry(data)
    feat = np.hstack([intensity, symmetry])

    return feat

def cal_feature_cls(data, label, cls_A=1, cls_B=6):
    """ calculate the intensity and symmetry feature of given classes
    Input:
        data: (n, d1, d2), the image data matrix
        label: (n, ), corresponding label
        cls_A: int, the first digit class
        cls_B: int, the second digit class
    Output:
        X: (n', 2), the intensity and symmetry feature corresponding to 
            class A and class B, where n'= cls_A# + cls_B#.
        y: (n', ), the corresponding label {-1, 1}. 1 stands for class A, 
            -1 stands for class B.
    """
    feat = cal_feature(data)
    indices = (label==cls_A) + (label==cls_B)
    X, y = feat[indices], label[indices]
    ind_A, ind_B = y==cls_A, y==cls_B
    y[ind_A] = 1
    y[ind_B] = -1

    return X, y

def plot_feature(feature, y, plot_num, ax=None, classes=np.arange(10)):
    """plot the feature of different classes
    Input:
        feature: (n, 2), the feature matrix.
        y: (n, ) corresponding label.
        plot_num: int, number of samples for each class to be plotted.
        ax: matplotlib.axes.Axes, the axes to be plotted on.
        classes: array(0-9), classes to be plotted.
    Output:
        ax: matplotlib.axes.Axes, plotted axes.
    """
    cls_features = [feature[y==i] for i in classes]

    marks = ['s', 'o', 'D', 'v', 'p', 'h', '+', 'x', '<', '>']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'orange', 'purple']
    if ax is None:
        _, ax = plt.subplots()
    for i, feat in zip(classes, cls_features):
        ax.scatter(*feat[:plot_num].T, marker=marks[i], color=colors[i], label=str(i))
    plt.legend(loc='upper right')
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    return ax

def cal_error(theta, X, y, thres=1e-4):
    """calculate the binary error of the model w given data (X, y)
    theta: (d+1, 1), the weight vector
    X: (n, d), the data matrix [X, y]
    y: (n, ), the corresponding label
    """
    out = X @ theta - thres
    pred = np.sign(out)
    err = np.mean(pred.squeeze()!=y)
    return err

# prepare data
train_data, train_label = load_mat('train_data.mat') # train_data: (7291, 16, 16), train_label: (7291, )
test_data, test_label = load_mat('test_data.mat') # test_data: (2007, 16, 16), train_label: (2007, )

cls_A, cls_B = 1, 6
X, y, = cal_feature_cls(train_data, train_label, cls_A=cls_A, cls_B=cls_B)
X_test, y_test = cal_feature_cls(test_data, test_label, cls_A=cls_A, cls_B=cls_B)

# train
iters = 2000
d = 2
num_sample = X.shape[0]
threshold = 1e-4
theta = np.zeros((d+1, 1))
theta_pocket = theta.copy()
learningRate = 1.0

X = np.hstack([np.ones((X.shape[0], 1)), X])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

initial_error = cal_error(theta, X, y, threshold)
best_in_error = initial_error

er_in_perceptron = []
er_in_pocket = []
er_out_perceptron = []
er_out_pocket = []

for iterate in range(iters):
    # TODO: add training code for perceptron and pocket
    for j in range(len(X)):
        
        i = random.randint(0, len(X)-1)
        current_X = X[i:i+1]
        prediction = np.dot(current_X, theta)
        f = np.sign(prediction - threshold)

        # Update weights if there is an error
        if f != y[i]:
            error = y[i] - f
            theta += learningRate * error * current_X.T
            break
    
    current_in_error = cal_error(theta, X, y, threshold)

    # save the best weights for pocket algorithm
    if current_in_error < best_in_error:
        theta_pocket = theta.copy()
        best_in_error = current_in_error
    
    # record errors
    er_in_perceptron.append(cal_error(theta, X, y, threshold))
    er_in_pocket.append(cal_error(theta_pocket, X, y, threshold))
    er_out_perceptron.append(cal_error(theta, X_test, y_test, threshold))
    er_out_pocket.append(cal_error(theta_pocket, X_test, y_test, threshold))
    

# plot Er_in and Er_out
# TODO
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(er_in_perceptron, label='Perceptron (Er_in)', color='blue', alpha=0.7)
plt.plot(er_in_pocket, label='Pocket Algorithm (Er_in)', color='red', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('In-Sample Error (Er_in)')
plt.title('In-Sample Error vs. Number of Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(er_out_perceptron, label='Perceptron (Er_out)', color='blue', alpha=0.7)
plt.plot(er_out_pocket, label='Pocket Algorithm (Er_out)', color='red', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Out-of-Sample Error (Er_out)')
plt.title('Out-of-Sample Error vs. Number of Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# plot decision boundary
# TODO you may utilize the plot_feature() function.
X_plot = X[:, 1:]
y_plot = np.where(y == 1, cls_A, cls_B)
fig, ax = plt.subplots(figsize=(10, 8))
plot_feature(X_plot, y_plot, plot_num=500, ax=ax, classes=[cls_A, cls_B])
ax.set_title('Feature Plot with Final Classification Boundaries (1 vs 6)')

x1_min = X_plot[:, 0].min()
x1_max = X_plot[:, 0].max()
x1_line = np.linspace(x1_min, x1_max, 100)

# plot perceptron's decision boundary
w_perceptron = theta.flatten()
if abs(w_perceptron[2]) > 1e-10:
    x2_line_perceptron = -(w_perceptron[0] - threshold + w_perceptron[1] * x1_line) / w_perceptron[2]
    ax.plot(x1_line, x2_line_perceptron, 'b--', linewidth=2, label='Perceptron Boundary')

# plot pocket algorithm's decision boundary
w_pocket = theta_pocket.flatten()
if abs(w_pocket[2]) > 1e-10:
    x2_line_pocket = -(w_pocket[0] - threshold + w_pocket[1] * x1_line) / w_pocket[2]
    ax.plot(x1_line, x2_line_pocket, 'r', linewidth=2, label='Pocket Algorithm Boundary')

ax.legend()
ax.set_xlabel('Intensity')
ax.set_ylabel('Asymmetry')
ax.grid(True, alpha=0.3)
plt.show()

print(f"Final Perceptron Er_in:  {er_in_perceptron[-1]:.6f}")
print(f"Final Perceptron Er_out: {er_out_perceptron[-1]:.6f}")
print(f"Final Pocket     Er_in:  {er_in_pocket[-1]:.6f}")
print(f"Final Pocket     Er_out: {er_out_pocket[-1]:.6f}")