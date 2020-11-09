import socket
import sys
import numpy as np
from dating.utils import floats_to_msg4



PORT = int(sys.argv[1])



COST_PERCENT = 98
LR_SEARCH_EPOCHS = 200
GDSC_EPOCHS = 200
CANDID_SAMPLE_SIZE = 1_000



def gradient_descent(X, y, w_guess, lr=0.01, reg_coef = 0.0 , max_epochs=30):
    m = np.shape(X)[0]  # total number of samples
    n = np.shape(X)[1]  # total number of features
    W = w_guess
    # stores the updates on the cost function (loss function)
    cost_history_list = []
    for current_iteration in np.arange(max_epochs):  # begin the process
        # compute the dot product between our feature 'X' and weight 'W'
        y_estimated = X.dot(W)
        # calculate the difference between the actual and predicted value
        error = y_estimated - y
        # calculate the cost (Mean squared error - MSE)
        cost = (1 / 2 * m) * np.sum(error ** 2) + reg_coef * W.T.dot(W)
        gradient = (1 / m) * X.T.dot(error)
        # Now we have to update our weights
        W = W - lr * gradient
        if current_iteration % 10 == 0:
            zzz = 3
            # print(f"cost:{cost} \t iteration: {current_iteration}")
            # keep track the cost as it changes in each iteration
        if current_iteration > 0:
            if cost > cost_history_list[-1] * COST_PERCENT / 100:
                cost_history_list.append(cost)
                break
            else:
                cost_history_list.append(cost)
        else:
            cost_history_list.append(cost)

    return W, cost_history_list



def find_next_candid(X,w,how_many_samples=100):
    S = np.linalg.inv(X.T.dot(X))
    candid_random = np.random.rand(how_many_samples,w_guess.shape[0])
    candid_samples = np.array((candid_random > 0.5*(1+w.ravel())).astype(int))
    best_candid = None
    best_cov = -9999999999999
    for candid in candid_samples:
        var = np.matmul(candid.reshape(1,-1),np.matmul(S,candid.reshape(-1,1)))
        if var > best_cov:
            best_cov = var
            best_candid = candid
    return best_candid

def cost_func(x_i,y_i,w_i):
    return (y_i - x_i.dot(w_i))**2


def selector(i,l):
    x = np.arange(l)
    y = np.delete(x,i)
    return y

def pick_lr(A, y, pred_w, reg_coef=0.05, epochs=100):
    bestCost = 999999999
    bestEta = 0
    eta_range = np.logspace(-5,-1,100)
    for i,currEta in enumerate(eta_range):
        currCost = 0
        for i in range(len(y)):
            sifter = selector(i,len(y))
            w_i, g_costs = gradient_descent(A[sifter,:], y[sifter], pred_w, lr=currEta, reg_coef=reg_coef, max_epochs=epochs)
            w_i = np.trunc(w_i*100)/100.0
            currCost += cost_func(A[i],y[i],w_i)
        if currCost < bestCost:
            bestCost = currCost
            bestEta = currEta
    return bestEta,bestCost




def get_valid_prob(n):
    alpha = np.random.random(n)
    p = np.random.dirichlet(alpha)
    p = np.trunc(p*100)/100.0

    # ensure p sums to 1 after rounding
    p[-1] = 1 - np.sum(p[:-1])
    return p


def get_valid_weights(n):
    half = n//2

    a = np.zeros(n)
    a[:half] = get_valid_prob(half)
    a[half:] = -get_valid_prob(n - half)
    return np.around(a, 2)



sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', PORT))

num_string = sock.recv(4).decode("utf-8")
assert num_string.endswith('\n')

num_attr = int(num_string[:-1])


A_init = np.zeros([20,num_attr],dtype='int16')
y_init = np.zeros([20,1])


print('20 Initial candidate scores and attributes:')
for i in range(20):
    # score digits + binary labels + commas + exclamation
    data = sock.recv(8 + 2*num_attr).decode("utf-8")
    y_init[i] = float(data.split(sep=':')[0])
    A_init[i,:] = np.array([int(_) for _ in data.split(sep=':')[1].split(sep='\n')[0].split(sep=',')])
    print('Score = %s' % data[:8])
    assert data[-1] == '\n'


w_guess = A_init[np.argmax(y_init),:].reshape(-1,1)

best_lr,curr_cost = pick_lr(A_init,y_init,w_guess,reg_coef= 0,epochs=LR_SEARCH_EPOCHS)
# ## Initial computations
# w_lstq = np.linalg.lstsq(a=A_init,b=y_init)[0]
# w_lstq[np.where(w_lstq>0)] = w_lstq[np.where(w_lstq>0)] / np.sum(w_lstq[np.where(w_lstq>0)])
# w_lstq[np.where(w_lstq<0)] = w_lstq[np.where(w_lstq<0)] / -1 * np.sum(w_lstq[np.where(w_lstq<0)])
#
# best_cost = 99999999999
# best_lr = 999999999999
# w_guess = w_lstq
# for w_idx in range(51):
#     if w_idx == 0:
#         w_try = w_lstq
#     else:
#         w_try = get_valid_weights(num_attr).reshape(-1,1)
#     curr_best_lr,curr_cost = pick_lr(A_init,y_init,w_try,reg_coef= 0,epochs=LR_SEARCH_EPOCHS)
#     if curr_cost < best_cost:
#         w_guess = w_try
#         best_cost = curr_cost
#         best_lr = curr_best_lr

A_all = np.zeros([41,num_attr])
y_all = np.zeros([41,1])
A_all[:20,:] = A_init
y_all[:20] = y_init


# w_guess, g_costs = gradient_descent(A_init, y_init, w_guess, lr=best_lr,reg_coef=0.00, max_epochs=GDSC_EPOCHS)

# next_candit = find_next_candid(A_init,w_guess,how_many_samples=CANDID_SAMPLE_SIZE)

next_candit = np.array(w_guess.ravel() > 0,dtype=int)

## Every round
for i in range(20):
    prev_candid = next_candit.copy()
    overall_idx = 20 + i
    A_all[overall_idx,:] = next_candit
    #Guess Weights
    sock.sendall(floats_to_msg4(next_candit))
    data = sock.recv(8).decode('utf-8')
    assert data[-1] == '\n'
    score = float(data[:-1])
    y_all[overall_idx] = score
    print('Received a score = %f for i = %d ' % (score, i))
    best_lr,curr_cost = pick_lr(A_all[:overall_idx,:], y_all[:overall_idx], w_guess, reg_coef=0, epochs=LR_SEARCH_EPOCHS)
    w_guess, g_costs = gradient_descent(A_all[i:overall_idx,:], y_all[i:overall_idx], w_guess, lr=best_lr, reg_coef=0.00, max_epochs=GDSC_EPOCHS)
    next_candit = np.array(w_guess.ravel() > 0, dtype=int)

    if (prev_candid == next_candit).all():
        next_candit = find_next_candid(A_all[:overall_idx,:], w_guess, how_many_samples=CANDID_SAMPLE_SIZE)
    else:
        next_candit = np.array(w_guess.ravel() > 0,dtype=int)

    # if (i+1) % 1 == 0:
    #     next_candit = np.array(w_guess.ravel() > 0,dtype=int)
    # else:
    #     next_candit = find_next_candid(A_all[:overall_idx,:], w_guess, how_many_samples=10000)


sock.close()




# bestCost = ∞
# 	bestEta = 0
# 	foreach η in eta_range
# 		currCost = 0
# 		foreach candidate i
# 			wi = train(X without xi, Y without yi)
# 			currCost = currCost + C(wi)
# 		if currCost < bestCost then
# 			bestLoss = currCost
# 			bestEta = η




















# naive_candid = np.zeros([num_attr])
# for feature_index in range(num_attr):
#     positive_values = A_init[:,feature_index].reshape(-1,1) * y_init
#     negative_values = (1-A_init[:, feature_index]).reshape(-1, 1) * y_init
#     if sum(positive_values)>sum(negative_values):
#         naive_candid[feature_index] = 1
#     else:
#         naive_candid[feature_index] = 0
#
# if naive_candid in A_init:
#     next_candit = naive_candid
# else:
#     next_candit = w_lstq











