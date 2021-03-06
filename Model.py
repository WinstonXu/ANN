
import numpy as np
from Layer import Layer

def loadFlower(filename = 'iris.csv'):
    '''Loads in iris data, returns independent and dependent matrices'''
    mydata = np.genfromtxt(filename, delimiter=',', skip_header=0)
    #Changed output to be 100, 010, or 001 for the 3 iris types
    y = mydata[:,[4,5,6]]
    X = mydata[:, [0,1,2,3]]
    X_norm = X / X.max(axis=0)
    return X_norm, y

def loadDataset(filename='breast_cancer.csv'):
    '''Loads in breast cancer data, returns independent and dependent matrices'''
    my_data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # The labels of the cases
    # Raw labels are either 4 (cancer) or 2 (no cancer)
    # Normalize these classes to 0/1
    y = (my_data[:, 10] / 2) - 1

    # Case features
    X = my_data[:, :10]

    # Normalize the features to (0, 1)
    X_norm = X / X.max(axis=0)

    return X_norm, y
#Just used to test derivative of error values
def gradientChecker(model, X, y):
    '''Used to check correct derivative error values are being passed down'''
    epsilon = 1E-5
    model.layers[0].weights[0][0] += epsilon
    out1 = model.forward(X)
    err1 = model.calculateError(y, out1)

    model.layers[0].weights[0][0] -= 2*epsilon
    out2 = model.forward(X)
    err2 = model.calculateError(y, out2)

    numeric = (err2 - err1) / (2*epsilon)
    print numeric

    model.layers[0].weights[0][0] += epsilon
    out3 = model.forward(X)
    err3 = model.calculateDerivError(y, out3)
    model.backward(err3)

def kFold(model, X, y, k):
    '''Given model and dataset, run K-Fold cross validation'''
    sets = np.split(X, k)
    # print sets[4].shape
    answers = np.split(y, k)
    oddOneOut = int(np.random.rand()*k)
    # print oddOneOut
    #train for longer than this?
    for i in range(oddOneOut):
        model.train(sets[i], answers[i], 200)
    for i in range(oddOneOut+1, k):
        model.train(sets[i], answers[i], 200)
    print "Test Case Error"
    # print sets[oddOneOut].shape
    test = model.forward(sets[oddOneOut])
    err = model.calculateError(answers[oddOneOut], test)
    model.reportAccuracy(sets[oddOneOut], answers[oddOneOut])
    print err

class Model:

    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def reportAccuracy(self, X, y):
        '''Given input and output, reports %% error'''
        out = self.forward(X)
        y = y.reshape(len(y), out.shape[1])
        out = np.round(out)
        count = np.count_nonzero(y - out)
        #len(X)*y.shape[1] to get correct number of total count ups
        correct = len(X)*y.shape[1] - count
        print "%.4f%%" % (float(correct)*100.0 / (len(X)*y.shape[1]))

    def calculateDerivError(self, y, pred):
        '''Returns derivative error given target and output'''
        # print y.shape, pred.shape
        y = y.reshape(len(y), pred.shape[1])
        return 2*(y - pred)

    def calculateError(self, y, pred):
        '''Given target and output returns total error'''
        y = y.reshape(len(y),pred.shape[1])
        return (np.sum(np.power((y - pred), 2)))

    def train(self, X, y, number_epochs):
        '''Given input, target, and number, run testing for that many generations'''
        for i in range(number_epochs):
            pred = self.forward(X)
            self.reportAccuracy(X, y)
            #Commented out to reduce console spam
            # print self.calculateError(y, pred)
            self.backward(self.calculateDerivError(y, pred))

    def forward(self, X):
        '''Passes input through all layers in model'''
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, error_vector):
        '''BackPropagation algorithm'''
        err = error_vector
        for i in range(len(self.layers)):
            err = self.layers[len(self.layers)-(i+1)].backward(err)


if __name__ == "__main__":
    X, y = loadDataset()
    BC = Model()
    Input = Layer(BC, 10, 25, .375)
    BC.addLayer(Input)
    Hidden = Layer(BC, 25, 1, .0022)
    BC.addLayer(Hidden)
    BC.train(X,y,400)
    #Numpy split likes clean division, just used train b/c otherwise
    #feeds in data row by row
    # kFold(BC, X, y, 683)

    print "\nSecond Data Set\n"
    X2, y2 = loadFlower()
    # print X2
    Fl = Model()
    fInput = Layer(Fl, 4, 6, .29)
    Fl.addLayer(fInput)
    fHidden = Layer(Fl, 6, 3, .03)
    Fl.addLayer(fHidden)
    kFold(Fl, X2, y2, 3)
    # Fl.train(X2, y2, 200)
