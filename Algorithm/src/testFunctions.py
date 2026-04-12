import math


'''
Raccolta delle 20 funzioni di test per l'algoritmo di Piyavski-Shubert.
 
Ogni coppia  (fi, ti)  contiene:
    fi : callable  f(x) -> float
    ti : tuple     (a, b, L)   estremo sx, estremo dx, costante di Lipschitz
'''


# 1
f1 = lambda x: (1/6)*x**6 - (52/25)*x**5 + (39/80)*x**4 + (71/10)*x**3 - (79/20)*x**2 - x + 1/10
t1 = (-1.5, 11, 13870)

# 2
f2 = lambda x: math.sin(x) + math.sin(10*x/3)
t2 = (2.7, 7.5, 4.29)

# 3
def f3(x):
    return -sum(k * math.sin((k+1)*x + k) for k in range(1, 6))
t3 = (-10, 10, 67)

# 4
f4 = lambda x: -(16*x**2 - 24*x + 5) * math.exp(-x)
t4 = (1.9, 3.9, 3)

# 5
f5 = lambda x: (3*x - 1.4) * math.sin(18*x)
t5 = (0, 1.2, 36)

# 6
f6 = lambda x: -(x + math.sin(x)) * math.exp(-x**2)
t6 = (-10, 10, 2.5)

# 7
f7 = lambda x: math.sin(x) + math.sin(10*x/3) + math.log(x) - 0.84*x + 3
t7 = (2.7, 7.5, 6)

# 8
def f8(x):
    return -sum(k * math.cos((k+1)*x + k) for k in range(1, 6))
t8 = (-10, 10, 67)

# 9
f9 = lambda x: math.sin(x) + math.sin(2*x/3)
t9 = (3.1, 20.4, 1.7)

# 10
f10 = lambda x: -x * math.sin(x)
t10 = (0, 10, 11)

# 11
f11 = lambda x: 2*math.cos(x) + math.cos(2*x)
t11 = (-1.57, 6.28, 3)

# 12
f12 = lambda x: math.sin(x)**3 + math.cos(x)**3
t12 = (0, 6.28, 2.2)

# 13
f13 = lambda x: -(x**(2/3)) + math.copysign(abs(x**2 - 1)**(1/3), x**2 - 1)
t13 = (0.001, 0.99, 8.5)

# 14
f14 = lambda x: -math.exp(-x) * math.sin(2 * math.pi * x)
t14 = (0, 4, 6.5)

# 15
f15 = lambda x: (x**2 - 5*x + 6) / (x**2 + 1)
t15 = (-5, 5, 6.5)

# 16
f16 = lambda x: 2*(x - 3)**2 + math.exp(0.5 * x**2)
t16 = (-3, 3, 85)

# 17
f17 = lambda x: x**6 - 15*x**4 + 27*x**2 + 250
t17 = (-4, 4, 2520)

# 18 (piecewise)
def f18(x):
    if x <= 3:
        return (x - 2)**2
    else:
        return 2 * math.log(x - 2) + 1
t18 = (0, 6, 4)

# 19
f19 = lambda x: -x + math.sin(3*x) - 1
t19 = (0, 6.5, 4)

# 20
f20 = lambda x: (math.sin(x) - x) * math.exp(-x**2)
t20 = (-10, 10, 1.3)


# Registro indicizzato (usato da main.py)
TEST_REGISTRY: dict[int, tuple] = {
    1:  (f1,  t1),   2:  (f2,  t2),   3:  (f3,  t3),   4:  (f4,  t4),
    5:  (f5,  t5),   6:  (f6,  t6),   7:  (f7,  t7),   8:  (f8,  t8),
    9:  (f9,  t9),   10: (f10, t10),  11: (f11, t11),  12: (f12, t12),
    13: (f13, t13),  14: (f14, t14),  15: (f15, t15),  16: (f16, t16),
    17: (f17, t17),  18: (f18, t18),  19: (f19, t19),  20: (f20, t20),
}