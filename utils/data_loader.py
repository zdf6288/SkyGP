from scipy.io import loadmat

def load_data(dataset='sarcos'):
    if dataset == 'sarcos':
        print("Loading sarcos dataset...")
        train_data = loadmat('dataset/sarcos/Sarcos_train.mat')
        
        X_train = train_data['sarcos_inv'][:, :21]
        Y_train = train_data['sarcos_inv'][:, 21:]
        X_test = train_data['sarcos_inv'][:, :21]
        Y_test = train_data['sarcos_inv'][:, 21:]
        #Normalize
        X_mean = X_train.mean(0)
        Y_mean = Y_train.mean(0)
        X_train = X_train - X_mean
        Y_train = Y_train - Y_mean
        X_test = X_test - X_mean
        Y_test = Y_test - Y_mean
        return X_train, Y_train, X_test, Y_test

    elif dataset == 'pumadyn32nm':
        print("Loading pumadyn32nm dataset...")
        mat = loadmat('dataset/puma/pumadyn32nm.mat')
        X_train = mat['X_tr']
        Y_train = mat['T_tr']
        X_test = mat['X_tst']
        Y_test = mat['T_tst']
        X_mean = X_train.mean(axis=0)
        Y_mean = Y_train.mean(axis=0)
        X_train = X_train - X_mean
        X_test = X_test - X_mean
        Y_train = Y_train - Y_mean
        Y_test = Y_test - Y_mean
        return X_train, Y_train, X_test, Y_test

    elif dataset == 'kin40k':
        print("Loading kin40k dataset...")
        data = loadmat('dataset/kin40k/kin40k.mat')
        X_train = data['X']
        Y_train = data['Y'].reshape(-1)
        X_mean = X_train.mean(axis=0)
        Y_mean = Y_train.mean()

        X_train = X_train - X_mean
        Y_train = Y_train - Y_mean

        X_test = X_train.copy()
        Y_test = Y_train.copy()
        return X_train, Y_train, X_test, Y_test

    elif dataset == 'electric':
        print("Loading electric dataset...")
        mat = loadmat('dataset/electric/electric_data_preprocessed.mat')
        data = mat['data']
        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X = X - X_mean
        Y = Y - Y_mean
        return X, Y, X.copy(), Y.copy()