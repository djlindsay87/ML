import numpy as np

class Kat:
    def __init__(self):
        return

    def initCM(self, y, yHat):
        self.trueP  = np.sum(y[(indices:=y.nonzero())]==yHat[indices])
        self.falseN = np.sum(y>yHat)
        self.falseP = np.sum(y<yHat)
        self.trueN  = np.sum(y[(indixes:=np.array(y-1).nonzero())]==yHat[indixes])
        
        print(f"[trueP  : {self.trueP}, falseN : {self.falseN}]")
        print(f"[falseP : {self.falseP}, trueN  : {self.trueN}]")
        return

    def accuracy(self, y, yHat)->float:
        return (np.array(y)==np.array(yHat)).mean()

    def recall(self, y, yHat)->float:
        if not (summ:= self.trueP + self.falseN)==0.0:
            return  self.trueP / summ
        else: return 0.0
        
    def precision(self, y, yHat)->float:
        if not (summ:=self.trueP + self.falseP)==0.0:
            return  self.trueP / summ
        else: return 0.0
        
    def f1(self, y, yHat)->float:
        if not (summ:=self.precision(y, yHat) + self.accuracy(y, yHat))==0.0:
            return 2 * self.precision(y, yHat) * self.recall(y, yHat) / summ
        else: return 0.0

    def logLoss(self, y, yHat)->float:
        clippedHat = np.clip(yHat, 10e-6, 1-10e-6)
        loss = -np.mean(y*np.log(clippedHat)+(1-y)*np.log(1-clippedHat))
        return loss
        
    def MSE(self, y, yHat):
        return np.mean((y - yHat) ** 2)

    def __call__(self, y, yHat):
        self.dm = [self.accuracy(y, yHat), self.recall(y, yHat), self.precision(y, yHat), self.f1(y, yHat)]
        print("acc: {:.3f}, rec: {:.3f}, pre: {:.3f}, f1: {:.3f}".format(*self.dm))
        return self.dm
