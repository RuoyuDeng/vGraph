# Suppose this is foo.py.

# print("before import")
# import math

# print("before function_a")
# def function_a():
#     print("Function A")
#     print("current __name__ is " + __name__)

# print("before function_b")
# def function_b():
#     print("Function B {}".format(math.sqrt(100)))
#     print("current __name__ is " + __name__)

# print("before __name__ guard", __name__)
# if __name__ == '__main__':
#     function_a()
#     function_b()
# print("after __name__ guard",__name__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--option-1") # will be automattically repharse to an attribute option_1
args = parser.parse_args()
print(args.option_1)