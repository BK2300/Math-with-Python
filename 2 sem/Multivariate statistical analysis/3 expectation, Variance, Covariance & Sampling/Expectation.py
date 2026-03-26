"""




"""




# Mulige værdier
x = [1, 2, 3, 4, 5, 6]

# Sandsynligheder
p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

# Expectation
E = sum(xi * pi for xi, pi in zip(x, p))

print(E)


# Weighted
x = [0, 10]
p = [0.9, 0.1]

E = sum(xi * pi for xi, pi in zip(x, p))

print(E)





# Observerede data
data = [2, 6, 1, 4, 6]

# Gennemsnit af observationer
average = sum(data) / len(data)

print(average)


