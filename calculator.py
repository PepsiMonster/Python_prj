import numpy as np
import pandas as pd
import matplotlib as mt
import scipy as sc
import faker as fk
import math


def result(ch1, ch2, zn):
    if zn == "+":
        return ch1+ch2
    elif zn == "-":
        return ch1-ch2
    elif zn == "*":
        return ch1*ch2
    else:
        return ch1+ch2


a = int(input())
d = input("Выберите арифметическое действие + - * / : ")
b = int(input())

res = result(a,b,d)
print(a,d,b,"=",res)