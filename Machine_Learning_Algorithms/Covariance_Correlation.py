'''
How are the variables dependent, linked, and varying agiainst each other?

Covaraince brings about the varition across variables. We use covirance to measure how much two
varibles change with each other.Covaraince defines the directional association between the varibles.

Correlation determine how strongly linked two varibles are to each other.

Ori: https://stackabuse.com/covariance-and-correlation-in-python
Data: https://www.kaggle.com/uciml/iris
A good understand: https://www.zhihu.com/question/20852004
'''


# Load the feature_1: sepal_length; feature_2: sepal_width
with open('Iris.csv', 'r') as f:
    g = f.readlines()
    sep_length = [float(x.split(',')[1]) for x in g[1:]]
    sep_width  = [float(x.split(',')[2]) for x in g[1:]]


# visualize the data
import matplotlib.pyplot as plt
import seaborn as sns
sns.regplot(x = sep_length, y = sep_width)
plt.savefig("filename.png")


# covarince
def covariance(x, y):
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y) / float(len(y))
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = sum([sub_x[i] * sub_y[i] for i in range(len(sub_x))])
    denominator = len(x)-1
    cov = numerator / denominator
    return cov

print("Covarince between x and y is:", covariance(sep_length, sep_width))

# correlation
def correlation(x, y):
    mean_x = sum(x) / float(len(x))
    mean_y = sum(y) / float(len(y))
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = sum([sub_x[i] * sub_y[i] for i in range(len(sub_x))])
    std_deviation_x = sum([sub_x[i] ** 2.0 for i in range(len(sub_x))])
    std_deviation_y = sum([sub_y[i] ** 2.0 for i in range(len(sub_y))])
    denominator = (std_deviation_x * std_deviation_y) ** 0.5
    cor = numerator / denominator
    return cor

print("Correlation between x and y is:", correlation(sep_length, sep_width))
