import numpy as np






with open("100num.txt") as file:
    numbers = [int(line.strip()) for line in file.readlines()]
    #print(numbers)
    avg = sum(numbers)/len(numbers)
    var = sum((x - avg) ** 2 for x in numbers) / len(numbers)    
    print(avg,var)

    
    #print(avg)

