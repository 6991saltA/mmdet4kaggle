# coding: utf-8
import os


# newTuple = (1, 2, 3, 4)
# newList = list(newTuple)
# print(type(newList))

# list(newTuple)[2] = 10
# print(type(newTuple))
# print(newTuple)

def list2Tuple(x):
    return tuple(x)


if __name__ == "__main__":
    newList = [1, 2, 3, 4]
    newTuple = list2Tuple(newList)
    print(type(newTuple))