import matlab.engine
import numpy as np
import sys

class SSVC:
    # Smooth Support Vector Machine for binary Classification
    def __init__(self):
        # starting matlab engine
        print('starting matlab engine..', end='')
        self.engine = matlab.engine.start_matlab()
        print('ok')

        # init parameter
        self.w = None
        self.b = None
        self.c = 1.0

    def print_params(self):
        print('C=', self.c, 'kernel=linear')
        print('w=', self.w)
        print('b=', self.b)

    def get_params(self):
        print('return w,b')
        return self.w, self.b

    @staticmethod
    def check_type(data):
        # convert to ndarray.
        # thought matlab.double(x) only accept list type,
        #   but we will convert array to list in `find_pos_neg_part` function.
        # only accept list/ndarray type as input.
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            print('input data type must be list / ndarray')
            sys.exit(1)

    @staticmethod
    def find_pos_neg_part(x, y):
        # find pos & neg part from all data, and return list of x_pos & x_neg
        x_pos = []
        x_neg = []
        for idx, val in enumerate(y):
            if val == 1:
                x_pos.append(x[idx].tolist())
            elif val == -1:
                x_neg.append(x[idx].tolist())
            else:
                print('label value must be 1/-1')
                sys.exit(1)
        return x_pos, x_neg

    def fit(self, x, y, c=1.0):
        self.c = float(c)
        print("SSVC(C=" + str(self.c) + ", kernel='linear')")

        # find pos & neg part of x
        x = self.check_type(x)
        y = self.check_type(y)
        x_pos, x_neg = self.find_pos_neg_part(x, y)

        # convert list to matlab data type
        x_pos_matlab_type = matlab.double(x_pos)
        x_neg_matlab_type = matlab.double(x_neg)
        [self.w, self.b] = self.engine.ssvc(x_pos_matlab_type, x_neg_matlab_type, self.c, nargout=2)

        # convert weights to array
        self.w = np.array(self.w)
        self.b = np.array(self.b)

    def predict(self, x):
        x = self.check_type(x)
        # print("A*w-b=", np.dot(x, self.w) - self.b)
        return (np.sign(np.dot(x, self.w) - self.b)).ravel()

    def score(self, x, y):
        # find pos & neg part of x
        x = self.check_type(x)
        y = self.check_type(y)
        x_pos, x_neg = self.find_pos_neg_part(x, y)

        # convert list to array
        x_pos = np.array(x_pos)
        x_neg = np.array(x_neg)
        data_size = x_pos.shape[0] + x_neg.shape[0]
        acc_count = 0
        pos_result = self.predict(x_pos)
        neg_result = self.predict(x_neg)
        for i in pos_result:
            if i == 1:
                acc_count += 1
        for i in neg_result:
            if i == -1:
                acc_count += 1
        accuracy = float(acc_count) / data_size
        print('accuracy:', accuracy)
        return accuracy

class SSVR:
    # Smooth Support Vector Machine for binary Classification
    def __init__(self):
        # starting matlab engine
        print('starting matlab engine..', end='')
        self.engine = matlab.engine.start_matlab()
        print('ok')

        # init parameter
        self.w = None
        self.b = None
        self.c = 1.0
        self.epsilon = 0.1

    def print_params(self):
        print('C=', self.c, 'kernel=linear')
        print('w=', self.w)
        print('b=', self.b)

    def get_params(self):
        print('return w,b')
        return self.w, self.b

    @staticmethod
    def check_type(data):
        # convert to list.
        # cause matlab.double(x) only accept list type.
        # only accept list/ndarray type as input.
        if isinstance(data, list):
            return data
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            print('input data type must be list / ndarray')
            sys.exit(1)

    def fit(self, x, y, c=1.0, epsilon=0.1):
        self.c = float(c)
        self.epsilon = float(epsilon)
        print("SSVR(C=" + str(self.c) + ", epsilon=" + str(self.epsilon) + ", kernel='linear')")

        x = self.check_type(x)
        # in ssvr, label shape must be (n, 1), notice that label shape of ssvc is (1, n)
        y = y.reshape(-1, 1)
        y = self.check_type(y)

        # convert list to matlab data type
        x_matlab_type = matlab.double(x)
        y_matlab_type = matlab.double(y)
        [self.w, self.b] = self.engine.ssvr(x_matlab_type, y_matlab_type, self.c, self.epsilon, nargout=2)

        # convert weights to array
        self.w = np.array(self.w)
        self.b = np.array(self.b)

    def predict(self, x):
        x = self.check_type(x)
        # print("A*w-b=", np.dot(x, self.w) - self.b)
        return (np.dot(x, self.w) - self.b).ravel()

    def score(self, x, y):
        data_size = x.shape[0]
        prediction = self.predict(x)
        sse = np.sum(np.power(prediction-y, 2))
        sst = np.sum(np.power(y-np.mean(y), 2))
        r_squared = 1-sse/sst
        mse = np.sum((np.power(prediction-y, 2)) / data_size)

        print('mse:', mse)
        print('r^2:', r_squared)
        return mse, r_squared



