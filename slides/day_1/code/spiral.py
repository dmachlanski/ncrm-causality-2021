import numpy as np
import matplotlib.pyplot as plt



def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))



def filter_one(X,y,a1_max, a1_min, a2_max, a2_min):
    X_new = []
    y_new = []
    for (a1,a2), c in zip(X,y):
        #print(a1,a2,c)
        if(a1 < a1_max and a1 > a1_min):
            if(a2 < a2_max and a2 > a2_min):
                X_new.append([a1,a2])
                y_new.append(c)
    X = np.array(X_new)
    y = np.array(y_new)
    return X,y


if __name__ == "__main__":
    for n_points in [50,100,200,500,1000,10000]:
        np.random.seed(100)
        #n_points = 10000
        X, y = twospirals(n_points)
        #print(X.max(), X.min())

        a1_max, a1_min, a2_max, a2_min = 0,-5,8,2.3

        X,y = filter_one(X,y, a1_max, a1_min, a2_max, a2_min)
        plt.title('training set')
        axes = plt.gca()
        axes.set_xlim([-13, 13])
        axes.set_ylim([-13, 13])
        plt.scatter(X[y == 0, 0], X[y == 0, 1], 1, label='data',c = "red")
        plt.legend()
        plt.savefig("./intra_spiral_%.2f_%.2f_%.2f_%.2f_%d.pdf"%(a1_max, a1_min, a2_max, a2_min,n_points))
        plt.cla()

    for n_points in [50,100,200,500,1000,10000]:
        np.random.seed(100)
        #n_points = 10000
        X, y = twospirals(n_points)
        #print(X.max(), X.min())

        a1_max, a1_min, a2_max, a2_min = 0,-5,8,2.3

        X,y = filter_one(X,y, a1_max, a1_min, a2_max, a2_min)
        plt.title('training set')
        axes = plt.gca()
        axes.set_xlim([-13, 13])
        axes.set_ylim([-13, 13])
        plt.scatter(X[y == 0, 0], X[y == 0, 1], 1, label='class 1',c = "red")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], 1, label='class 2', c = "blue")
        plt.legend()
        plt.savefig("./intra_spiral_%.2f_%.2f_%.2f_%.2f_%d_class.pdf"%(a1_max, a1_min, a2_max, a2_min,n_points))
        plt.cla()

    np.random.seed(100)
    # n_points = 10000
    X, y = twospirals(10000)


    plt.title('training set')
    axes = plt.gca()
    axes.set_xlim([-13, 13])
    axes.set_ylim([-13, 13])
    plt.scatter(X[y == 0, 0], X[y == 0, 1], 1, label='class 1', c="red")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], 1, label='class 2', c="blue")
    plt.legend()
    plt.savefig("./extra_spiral.pdf" )
    plt.cla()





