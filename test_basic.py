import numpy as np

a = np.zeros((3, 3))
print("shape:", a.shape)
print("dtype:", a.dtype)

b = np.ones((3, 3))

# Test direct method call
print("has __add__:", hasattr(a, '__add__'))
try:
    c = a.__add__(b)
    print("__add__ works, shape:", c.shape)
except Exception as e:
    print("__add__ error:", e)

# Test operator
try:
    c = a + b
    print("+ works, shape:", c.shape)
except Exception as e:
    print("+ error:", e)

print("done!")
