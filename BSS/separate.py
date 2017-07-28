
import numpy as np

def separate_time_stamp(obser):
    
    #做到n-3的切片
    shape = (   len(obser)-4 , 4  )
    x = np.zeros(  shape )

    for i in range(4):
        x[ : , i ] = obser[ i : shape[0]+i ]

    #use reshape to build 1*n vector
    #np.reshape(x,(1,6)  , order = "C"  )    C means reshape by axis x ,F by axis y
    return x

if __name__ == "__main__":
    n = np.arange(0,40,0.1)
    x = separate_time_stamp(n)

    x_re = np.reshape(x,(x.shape[0]*x.shape[1])  , order = "C"  )

    print("original shape : " , n.shape)
    print("separate step : " , x.shape)
    print(x[:10,:4],"\n")
    print("reshape step : " , x_re.shape)
    print("first ten of reshaped x : \n", x_re[:10])
