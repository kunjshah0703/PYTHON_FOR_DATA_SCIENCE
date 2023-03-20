# Functions in Pyhton
"""
def square(value) :  # <- Function Header
    new_value = value ** 2   # <- Function Body
    print(new_value)
square(4)
square(5)
"""

# Multiple function parameters
"""
def raise_to_power(value1, value2):
    # ""Raise value1 to the power of value2.""
    new_value = value1 ** value2
    return new_value
result = raise_to_power(2, 3)
print(result)
"""

# Returning multiple values in a function with the help of tuple
"""
def raise_both(value1, value2):
    ""Raise value1 to the power of value2 and vice versa.""
    new_value1 = value1 ** value2
    new_value2 = value2 ** value1

    new_tuple = (new_value1, new_value2)

    return new_tuple
result = raise_both(2, 3)
print(result)
"""

"""
def count_entries(df, col_name):
    langs_count = {}
    col = df[col_name]
    for entry in col : 
        if entry in langs_count.keys():
            langs_count[entry] += 1
        else : 
            langs_count[entry] = 1
    return langs_count
result = count_entries(tweets_df, 'lang')
print(result)    
"""

# Global vs local scope (1)
"""
new_val = 10

def square(value) :
    new_val = value ** 2
    return new_val
square(3)
new_val
"""
# Default Argumnets.

"""
def power(number, pow = 1) : 
    new_value = number ** pow
    return new_value

power(9) # Will return 9 as we have defined default argument to be 1 only
power(9, 2) # Will replace default argument
"""

# Flexible Arguments

"""
def add_all(*args) : 

    #Initialize sum
    sum_all = 0
    
    # Accumulate the sum
    for num in args : 
        sum_all += num

    return sum_all

add_all(2, 3, 5, 6)


def print_all(**kwargs) : 
    for key, value in kwargs.items():
        print(key + ": " + value)

print_all(name = "dumbledore", jon = "headmaster")
"""

# Lambda Functions
"""

raise_to_power = lambda x, y: x ** y
raise_to_power(2, 3)

nums = [48, 6, 9, 21, 1]
sqaure_all = map(lambda num : num ** 2, nums)
print(sqaure_all)
print(list(sqaure_all))
"""
