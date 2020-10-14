import csv

for i in range(1, 101):
    X = str(i)
    X += ".txt"
    with open(X, 'r') as in_file:
        lines = in_file.read().splitlines()
        lines = ",".join(lines)
        X = []
        X.append(lines)
        lines = X
        lines = [line.replace(","," ").split() for line in lines]
        #lines = " ".join(lines)
        #print(X) 
        lines = zip(*[lines]*1)
        with open('ans.csv', 'a') as out_file:
            writer = csv.writer(out_file)
            for group in lines:
                writer.writerows(group)
