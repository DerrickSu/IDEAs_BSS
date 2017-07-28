


import numpy as np


# lorenze模組
class lorenze():
    
    def __init__( self ,rho = 28,sigma = 10 , beta = 8.0/3.0):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

        # 設定初始值及預設
        self.state0 = [1.0, 1.0, 1.0]
        self.t = np.arange(0.0, 40.0, 0.01)

        # 亂數序列
        self.seq = [] 
        self.__states()

        
    def f(self , state, t):
        x, y, z = state  # unpack the state vector
        
        # Lorenze 公式
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  # derivatives


    #產生序列
    def __states(self):
        from scipy.integrate import odeint
        # scipy的積分模組
        self.seq = odeint(self.f , self.state0,self.t)


    # 設定初值及更新序列
    def var_set(self , state0 , t):
        self.state0 = state0
        self.t = t
        self.__states()


    # 印出圖形
    def display(self):
        
        import matplotlib.pyplot as plt

        # 印出3D互動的方法
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import interactive

        interactive(True)
        fig = plt.figure()
        ax = fig.gca(projection='3d') # Axe3D的用法
        ax.plot(self.seq[:,0], self.seq[:,1], self.seq[:,2])
        plt.show()


# 測試
if __name__ == "__main__":
    lo_0 = lorenze()
    lo_1 = lorenze(90,25,11.0/3.0)
    var_dict = dict( state0 = [0.5,2,8] , t = np.arange(0,80,0.002) )
    lo_1.var_set(**var_dict)

    print(lo_0.seq.shape)
    print(lo_1.seq.shape)
    lo_0.display()
    lo_1.display()
