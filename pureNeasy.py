print("Will it jam?")

import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt

import metrickitty as kit

Num = 100
RNG=2048

knitters=1000
learningRate = 0.1

def main():
    Trugoy = Pure(RNG)
    X_tr, y_tr = Trugoy.genData(Num)
    X_tst, y_tst = Trugoy.genData(Num)
    mKit = kit.Kat()
    
    Hector = Easy(lr=learningRate, n_iters=knitters, kitty=mKit)
    Hector.fit(X_tr, y_tr)
    
    y_pred = Hector.predict(X_tst)  
    Hector.summarize(y_tst, y_pred, True)
    return

class Pure:
    def __init__(self, rng):
        self.rng = rng
        self.dim = int(1+float(np.log2(self.rng)))
        return
    
    def genData(self, N):
        X = y = np.random.randint(self.rng, size=(N,1))
        return self.__barney(X), y.flatten()

    def __barney(self, X):
        assert np.issubdtype(X.dtype, np.integer)
        barneySuit = 2 ** np.arange(self.dim, dtype=X.dtype)
        return np.fliplr((X & barneySuit.T).astype(bool).astype(int))

class Easy:
    def __init__(self, lr:float=.001, n_iters:int=1000, kitty:kit.Kat=None):
        self.lr, self.n_iters = lr, n_iters
        self.weights, self.bias = None, None
        self.kitty=kitty; self.lossList, self.accList = [],[]
        return
    
    def fit(self, X, y):
        nSamp, nFeat = X.shape
        self.weights = np.zeros(nFeat)
        self.bias = 0
        
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            if self.kitty:
                self.lossList+=[self.kitty.MSE(y, np.around(y_pred))]
                self.accList+=[self.kitty.accuracy(y, np.around(y_pred))]
            
            dw = (1 / nSamp) * np.dot(X.T, (y_pred - y))
            db = (1 / nSamp) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return np.around(y_pred).astype(int)

    def summarize(self, y, y_pred, fuhVizzle:bool=False):
        tstacc = self.kitty.accuracy(y, y_pred)
        tstloss=self.kitty.MSE(y, y_pred)
        print(f"Test Accuracy: {tstacc:.3f}, Loss: {tstloss:.3f}")
        if fuhVizzle: mapper(np.arange(self.n_iters), loss=self.lossList, metric={"acc":self.accList})
        return

def mapper(x, **ys):
    fig, ax = plt.subplots()
    ax.set_title("loss and metrics vs iteration")
    ax.set_xlabel("iteration")
    lbls=[]; lines=[]

    for i, (lbl, vals) in enumerate(ys.items()):
        if lbl=='loss':
            lines += [ax.plot(x, vals, label=lbl, color='red')]
            ax.set_ylabel("Mean Square Error Loss")
            lbls+=[lbl]
        elif lbl == 'metric':
            axm=ax.twinx()
            axm.cm='viridis'
            for l, v in vals.items():
                lines+=[axm.plot(x, v, label=l)]
                lbls+=[l]
            axm.set_ylabel("Various Metrics")
    fig.legend(lbls)
    plt.show()
    return

if __name__ == '__main__':
    main()