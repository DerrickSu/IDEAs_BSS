




"""
Initialization of rCKC.
Some codes are written for original CKC.

讀mat檔
scipy.io.loadmat 

讀CSV檔
x = pd.read_csv(r"F:\IDEAs_BSS\BSS\Data\S1-Delsys-15Class\HC_1.csv",header = None)

或(減少使用套件)
with open(r"F:\IDEAs_BSS\BSS\Data\S1-Delsys-15Class\HC_1.csv","r") as f :
    x = csv.reader(f)
    y = []
    for i,z in enumerate(x):
            if i >= 10:
                    break
            y.append(np.array(z).astype("float"))
    y = np.array(y)
    f.close()
    

"""
import numpy as np

import rCKC
import csv





"""
# Simulation data
# n = 1009
import random as rd

my_seq = np.array([ np.sin(np.pi* i/100) for i in range(1009)  ])
my_rd = np.array(rd.sample(list(range(-20,80)),50)*20+[1,5,2,3,5,8,9,5,1])/50

mix = np.array( [[1 , 0],[0,1]])
Y = (np.c_[my_seq,my_rd]@mix).T

Y = rCKC.Sample_process(Y)

M = 2
# Data end 
"""


def main():
    M=10

    with open(r"F:\IDEAs_BSS\BSS\Data\S1-Delsys-15Class\HC_1.csv","r") as f :
        x = csv.reader(f)
        y = []
        for i,z in enumerate(x):
                if i >= 1009:
                        break
                y.append(np.array(z).astype("float"))
        y = np.array(y)
        f.close()

    Y = rCKC.Sample_process(y.T)

    cov = Y@Y.T
    cov_inv = np.linalg.inv(cov)

    act_idx = np.diag( Y.T@ cov_inv @ Y)
    max_idx = act_idx.argsort()[::-1][0:M]

    listMU = []
    eval_fun = [ "rCKC.MU( Y[:,max_idx[{0}]] , act_idx[max_idx[{0}]] )".format(i) for i in range(M)]
    for i in eval_fun:
        listMU.append( eval(i))
    
    fun = ["listMU[{0}].train(Y,cov_inv,1000)".format(i) for i in range(M)] 
    for i in fun:
        eval(i)

    for i in range(M):
        a = listMU[i]
        print("MU_{0}: {1}".format(i,a.n)  )


    # END main







if __name__ == "__main__":
    main()


