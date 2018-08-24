import pickle
import sys
import math


def func(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d


def get_size(instance):
    size = instance[3:] if instance[3].isdigit() else instance[4:]
    if not size[-1].isdigit():
        size = size[:-1]
    return int(size)


instance = sys.argv[1].split('.')[0]
size = get_size(instance)
with open('func_coef.dat', 'rb') as coef_file:
    popt = pickle.load(coef_file)
print(*popt)
time = func(size, *popt)
time = 1 if time < 1 else math.ceil(time)
print(time, end='')
