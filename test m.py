from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([2,4,6], dtype=np.float64)
ys = np.array([2,4.,1], dtype=np.float64)

def best_fit_slope_and_intercept(x,y):

    mxs = mean(x)
    mys = mean(y)

    #gradient
    
    #m = (((mxs*mys) - mean(x*y))/
     #    ((mxs**2) - (mean(x**2))))

    #m = sum((x-mxs)*(y-mys))/((sum((x-mxs)**2)*sum((y-mys)**2))**0.5)

    #y-intercept
    
    c = mys - (m*mxs)

    return m, c

#for all xs the difference between the ys of the best fit & data is squared

def square_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    square_error_regr = square_error(ys_orig, ys_line)
    square_error_y_mean = square_error(ys_orig, y_mean_line)
    return (1-(square_error_regr/square_error_y_mean))

m,c = best_fit_slope_and_intercept(xs,ys)

#create the data for a line
regression_line = [(m*x)+c for x in xs]
#regression_line_plot = [(m*xs[0])+c , (m*xs[len(xs)-1])+c] #doesnt work

predict_x = 8
predict_y = (m*predict_x)+c

r_square = coefficient_of_determination(ys, regression_line)
print(r_square)

print(m)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y)
plt.plot(xs, regression_line)
plt.show()
