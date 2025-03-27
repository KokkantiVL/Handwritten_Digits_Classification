import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt

def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0: n_hidden * (n_input + 1)].reshape((n_hidden, n_input + 1))
    w2 = params[n_hidden * (n_input + 1):].reshape((n_class, n_hidden + 1))
    
    m = training_data.shape[0]
    bias = np.ones((m, 1))
    training_data_bias = np.hstack((training_data, bias))
    
    z2 = training_data_bias.dot(w1.T)
    a2 = sigmoid(z2)
    a2_bias = np.hstack((a2, np.ones((a2.shape[0], 1))))
    z3 = a2_bias.dot(w2.T)
    a3 = sigmoid(z3)
    
    y = np.zeros((m, n_class))
    for i in range(m):
        y[i, int(training_label[i])] = 1
        
    log_loss = -y * np.log(a3) - (1 - y) * np.log(1 - a3)
    error = np.sum(log_loss) / m
    reg_term = (lambdaval / (2 * m)) * (np.sum(np.square(w1[:, :-1])) + np.sum(np.square(w2[:, :-1])))
    obj_val = error + reg_term

    delta3 = a3 - y
    grad_w2 = (delta3.T).dot(a2_bias) / m
    delta2 = delta3.dot(w2[:, :-1]) * a2 * (1 - a2)
    grad_w1 = (delta2.T).dot(training_data_bias) / m

    grad_w1[:, :-1] += (lambdaval / m) * w1[:, :-1]
    grad_w2[:, :-1] += (lambdaval / m) * w2[:, :-1]

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()))
    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    m = data.shape[0]
    bias = np.ones((m, 1))
    data_bias = np.hstack((data, bias))
    z2 = data_bias.dot(w1.T)
    a2 = sigmoid(z2)
    a2_bias = np.hstack((a2, np.ones((a2.shape[0], 1))))
    z3 = a2_bias.dot(w2.T)
    a3 = sigmoid(z3)
    labels = np.argmax(a3, axis=1)
    return labels

def preprocess():
    pickle_obj = pickle.load(open('basecode/face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

import matplotlib.pyplot as plt

"""************** Neural Network Script Starts here ********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
n_input = train_data.shape[1]
n_class = 2

selected_features = list(range(n_input))

hidden_layer_sizes = [128, 256, 512]          
lambdaval_list = [0, 0.001, 0.01, 0.1, 10]       

grid_train = {}
grid_val = {}
grid_test = {}

best_val_acc = 0
best_params = None

for n_hidden in hidden_layer_sizes:
    for lambdaval in lambdaval_list:
        print(f"\nTraining with {n_hidden} hidden units and λ = {lambdaval}")
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        opts = {'maxiter': 50}

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        params_opt = nn_params.get('x')
        w1 = params_opt[0: n_hidden * (n_input + 1)].reshape((n_hidden, n_input + 1))
        w2 = params_opt[n_hidden * (n_input + 1):].reshape((n_class, n_hidden + 1))
        
        pred_train = nnPredict(w1, w2, train_data)
        pred_val = nnPredict(w1, w2, validation_data)
        pred_test = nnPredict(w1, w2, test_data)

        train_acc = 100 * np.mean((pred_train == train_label).astype(float))
        val_acc = 100 * np.mean((pred_val == validation_label).astype(float))
        test_acc = 100 * np.mean((pred_test == test_label).astype(float))
        print(f"Train Accuracy: {train_acc:.2f}%  |  Validation Accuracy: {val_acc:.2f}%  |  Test Accuracy: {test_acc:.2f}%")
        
        grid_train[(n_hidden, lambdaval)] = train_acc
        grid_val[(n_hidden, lambdaval)] = val_acc
        grid_test[(n_hidden, lambdaval)] = test_acc
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (n_hidden, lambdaval, w1, w2)

print("\n***** Grid Search Results for Face Classification *****")
print(f"Best combination: Hidden Units = {best_params[0]}, λ = {best_params[1]}, Validation Accuracy = {best_val_acc:.2f}%")

best_n_hidden, best_lambda, best_w1, best_w2 = best_params
final_pred_test = nnPredict(best_w1, best_w2, test_data)
final_test_acc = 100 * np.mean((final_pred_test == test_label).astype(float))
print(f"Test Set Accuracy for best combination: {final_test_acc:.2f}%")

params_to_save = {
    'selected_features': selected_features,
    'optimal_n_hidden': best_n_hidden,
    'w1': best_w1,
    'w2': best_w2,
    'optimal_lambda': best_lambda
}
with open('face_params.pickle', 'wb') as f:
    pickle.dump(params_to_save, f)
print("Parameters saved to face_params.pickle")

num_h = len(hidden_layer_sizes)
num_l = len(lambdaval_list)
heat_train = np.zeros((num_h, num_l))
heat_val = np.zeros((num_h, num_l))
heat_test = np.zeros((num_h, num_l))

for i, h in enumerate(hidden_layer_sizes):
    for j, lam in enumerate(lambdaval_list):
        heat_train[i, j] = grid_train[(h, lam)]
        heat_val[i, j] = grid_val[(h, lam)]
        heat_test[i, j] = grid_test[(h, lam)]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

im0 = axs[0].imshow(heat_train, interpolation='nearest', cmap='viridis')
axs[0].set_title("Training Accuracy")
axs[0].set_xticks(np.arange(num_l))
axs[0].set_xticklabels(lambdaval_list)
axs[0].set_yticks(np.arange(num_h))
axs[0].set_yticklabels(hidden_layer_sizes)
axs[0].set_xlabel("λ")
axs[0].set_ylabel("Hidden Units")
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(heat_val, interpolation='nearest', cmap='viridis')
axs[1].set_title("Validation Accuracy")
axs[1].set_xticks(np.arange(num_l))
axs[1].set_xticklabels(lambdaval_list)
axs[1].set_yticks(np.arange(num_h))
axs[1].set_yticklabels(hidden_layer_sizes)
axs[1].set_xlabel("λ")
axs[1].set_ylabel("Hidden Units")
fig.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(heat_test, interpolation='nearest', cmap='viridis')
axs[2].set_title("Test Accuracy")
axs[2].set_xticks(np.arange(num_l))
axs[2].set_xticklabels(lambdaval_list)
axs[2].set_yticks(np.arange(num_h))
axs[2].set_yticklabels(hidden_layer_sizes)
axs[2].set_xlabel("λ")
axs[2].set_ylabel("Hidden Units")
fig.colorbar(im2, ax=axs[2])

plt.suptitle("Grid Search Results (facennScript): Accuracy (%)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
