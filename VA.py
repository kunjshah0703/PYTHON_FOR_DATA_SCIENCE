"""def myfunc(x):
    
    print("The number is",x)

myfunc(10)"""

"""
a\  # If you want to declare in multiple lines use \
=\
10
print(a)
"""

# a = 7;print(a)

"""
if 2>1:
    print("2 is greater")
    print("But 1 is something too")
"""

#Use of f string.
"""
x = 10; printer = "HP"
print(f"I just printed {x} pages with {printer} printer")
"""

# Subsetting Lists
"""
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
fam
fam[3]
fam[-1]
"""

#List Slicing [start : end ]. Start is inclusive
# means it is included. End is exclusive means it is
# not included.
"""
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
fam[3:5]
fam[:4]
fam[5:]
"""

# Changing list elements

"""
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
fam[7] = 1.86
fam

fam[0:2] = ["lisa", 1.74]
fam


"""

# Adding and removing elements
"""
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
fam_ext = fam + ["me", 1.79]
fam_ext

del(fam[2])
fam
fam_ext
"""

# Copying of list
"""
x = ["a", "b", "c"]
y = x
y[1] = "z"
x # value of an element in x got change because y is also
  # referencing to the same list not elements.
"""
# If we want to create y list wirh same values
# We can use list function like y = list(x) or slicing
# y = x[:]

# Functions

"""
fam = [1.73, 1.68, 1.71, 1.89]
fam
tallest = max(fam)
tallest

round(1.68,1)
round(1.68)

"""
"""
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
fam.index("mom")

fam.count(1.73)
"""
"""
sister = 'liz'
sister
sister.capitalize() # First letter is capitalize()

sister.replace("z", "sa")
"""

"""
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
fam
fam.append("me")
fam
fam.append(1.79)
fam
"""

"""
import numpy as np
np.array([1, 2, 3])
"""

"""
from numpy import array
array([1, 2, 3])
"""


# Numpy

"""Numpy can do these calculations easily as it assumes numpy arrays: 
contains only one type.
"""
"""
height = [1.73, 1.68, 1.71, 1.89, 1.79]
weight = [65.4, 59.2, 63.6, 88.4, 68.7]

import numpy as np
np_height = np.array(height)
np_weight = np.array(weight)
print(np_height)
print(np_weight)

bmi = np_weight / np_height ** 2
bmi

bmi[1]
print(bmi[bmi > 23])
"""

# Numpy 2D arrays
"""
import numpy as np
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
type(np_height)
type(np_weight)

np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],[65.4, 59.2, 63.6, 88.4, 68.7]])
np_2d
np_2d.shape
np_2d[0][2]
np_2d[0, 2]

"""

# Data Visualization using matplotlib library
"""
import matplotlib.pyplot as plt
year = [1950, 1970, 1990, 2010]
population_billion = [2.519, 3.692, 5.263, 6.972]
"""
"""
# Line plot
plt.plot(year,population_billion) # year - horizontal axis, population
# - vertical axis
plt.show()
"""
"""
# Scatter plot
plt.scatter(year, population_billion)
plt.xscale('log')
plt.show()
"""

# Histogram
"""
import matplotlib.pyplot as plt
# help(plt.hist)

values = [0, 0.6, 1.4, 1.6, 2.2, 2.5, 2.6, 3.2, 3.5, 3.9, 4.2, 6]
plt.hist(values, bins=3)
plt.show()
"""

# Customization of plots
# Basic Plot
"""
import matplotlib.pyplot as plt
year = [1950, 1951, 1952, 2100]
pop = [2.538, 2.57, 2.62, 10.85]

#Add more data
year = [1800, 1850, 1900] + year
pop = [1.0, 1.262, 1.650] + pop
plt.plot(year, pop)

plt.xlabel('Year')
plt.ylabel('Populatio in Billions')
plt.title('World Population Projections')
plt.yticks([0, 2, 4, 6, 8, 10],
          [0, '2B', '4B', '6B', '8B', '10B'] )


plt.show()
"""

# Dictionaries
"""
world_population = {"afghanistan" : 30.55, "albania" : 2.77, "algeria" : 39.21}
world_population["albania"]

world_population["sealand"] = 0.000028
world_population

del(world_population["sealand"])
world_population
"""

# Pandas

"""
dict = {
    "country" : ["Brazil", "Russia", "India", "China", "South Africa"],
    "capital" : ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
    "area" : [8.516, 17.10, 3.286, 9.597, 1.221],
    "population" : [200.4, 143.5, 1252, 1357, 52.98]
}
import pandas as pd
BRICS = pd.DataFrame(dict)
BRICS
BRICS.index = ["BR", "RU", "IN", "CH", "SA"] # changes index labels
BRICS

BRICS[["country"]]
BRICS[1:4]

# loc - label - based selection in pandas
# iloc - integer position - based selection in pandas

BRICS.loc[["RU", "IN", "CH"]]
BRICS.loc[["RU", "IN", "CH"], ["country", "capital"]]
BRICS.loc[:, ["country", "capital"]]

BRICS.iloc[[1, 2, 3]]
BRICS.iloc[[1, 2, 3],[0, 1]]
BRICS.iloc[:,[0,1]]

"""

# Comparison Operators.

# Control Structures (IF, ELIF, ELSE)
"""
z = 5
if z % 2 == 0 :
    print("checking " + str(z))
    print("z is even")
else :
    print("z is odd")
"""

"""
z = 6
if z % 2 == 0:
    print("z is divisible by 2")
elif z % 3 == 0:
    print("z is divisible by 3")
else : 
    print("z is neither divisible by 2 nor by 3")
"""

# Filtering pandas DataFrames
"""
dict = {
    "country" : ["Brazil", "Russia", "India", "China", "South Africa"],
    "capital" : ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
    "area" : [8.516, 17.10, 3.286, 9.597, 1.221],
    "population" : [200.4, 143.5, 1252, 1357, 52.98]
}
"""
"""
import pandas as pd
BRICS = pd.DataFrame(dict)
BRICS.index = ["BR", "RU", "IN", "CH", "SA"]
BRICS

# Get area of those countries whose area is grater than 8 million
# Step 1 : We will select area column
BRICS.iloc[:, 2] #BRICS.loc[:, "area"], BRICS["area"]

# Step 2 : Compare
is_huge = BRICS.iloc[:, 2] > 8 # BRICS.loc[:, "area"] > 8, BRICS["area"] > 8
is_huge

# Step 3: Subset DF
BRICS[is_huge]

"""

# While loop
"""
error = 50.0
while error > 1 :
    error = error / 4
    print(error)
"""

# For loop
"""
fam = [1.73, 1.68, 1.71, 1.89]
print(fam)

for index, height in enumerate(fam):
    print("index " + str(index) + ": " + str(height))
"""

# For loop for dictionary

"""
world = {
    "afghanistan" : 30.55,
    "albania" : 2.77,
    "algeria" : 39.21
}

for k, v in world.items() : 
    print(k + " -- " + str(v))
"""

# For loop for numpy array

"""
import numpy as np
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])

bmi = np_weight / np_height ** 2

for val in bmi : 
    print(val)

np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
meas = np.array([np_height, np_weight])
for val in np.nditer(meas) : 
    print(val)

"""

# For loop for pandas dataframe

"""
dict = {
    "country" : ["Brazil", "Russia", "India", "China", "South Africa"],
    "capital" : ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
    "area" : [8.516, 17.10, 3.286, 9.597, 1.221],
    "population" : [200.4, 143.5, 1252, 1357, 52.98]
}

import pandas as pd
BRICS = pd.DataFrame(dict)
BRICS.index = ["BR", "RU", "IN", "CH", "SA"]
BRICS

for lab, row in BRICS.iterrows() : 
    print(lab + ": " + row["capital"])

for lab, row in BRICS.iterrows() :
    # - Creating a series on every iteration
    BRICS.loc[lab, "name_length"] = len(row["country"])
print(BRICS)

BRICS["name_lenth1"] = BRICS["country"].apply(len)
print(BRICS)
"""

# Random Generators
"""
import numpy as np
np.random.rand() #Pseudo random numbers

# Coin toss
np.random.seed(123)
coin = np.random.randint(0, 2)
print(coin)
if coin == 0:
    print("heads")
else : 
    print("tails")
"""

"""
import numpy as np
np.random.seed(123)
outcomes = []
for x in range (10) :
    coin = np.random.randint(0, 2)
    if coin == 0:
        outcomes.append("heads")
    else : 
        outcomes.append("tails")
print(outcomes)
"""
"""
import numpy as np
np.random.seed(123)
tails = [0]
for x in range(10) : 
    coin = np.random.randint(0, 2)
    tails.append(tails[x] + coin)
    """


# Distribution
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
final_tails = []
for x in range(10000) : 
    tails = [0]
    for x in range(10) : 
        coin = np.random.randint(0, 2)
        tails.append(tails[x] + coin)
    final_tails.append(tails[-1])
plt.hist(final_tails, bins = 10)
plt.show()
#print(final_tails)
"""