import re
data_path = 'snli_0413_formula.txt'#'/home/8/17IA0973/snli_input_data_1214.json'
f = open('snli_0413_test.txt', 'w')
data = []

c = 1
lines = open(data_path)
for line in lines :
    if c > 4000 :
        break;
    line = line.split('#')
    l2 = line[1].rstrip()
    data.append(l2)
    c += 1



for i in data :
    f.write(str(i)+'\n')
f.close()



print(len(data))
