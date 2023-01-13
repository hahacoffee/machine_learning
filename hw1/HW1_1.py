import random
import csv

a=0
b=0
c=0

with open('Wine.csv', newline='') as csvfile:
    CSVDATA = csv.reader(csvfile)
    datas = list(CSVDATA)

for i in range(0,178):
    idx = datas[i][0]
    if idx=='1':
        a+=1
    elif idx=='2':
        b+=1
    elif idx=='3':
        c+=1

R1 = random.sample(range(0,a),18)
R2 = random.sample(range(a,a+b),18)
R3 = random.sample(range(a+b,a+b+c),18)

with open('test.csv', 'w', newline='') as csvfile1, open('train.csv', 'w', newline='') as csvfile2:
    writer1 = csv.writer(csvfile1)
    writer2 = csv.writer(csvfile2)
    
    for i in range(0,a):
        for j in range(0,18):
            if(R1[j]==i):
                writer1.writerow(datas[R1[j]])
                break
            if j == 17:
                writer2.writerow(datas[i])
    
    for i in range(a,a+b):
        for j in range(0,18):
            if(R2[j]==i):
                writer1.writerow(datas[R2[j]])
                break
            if j == 17:
                writer2.writerow(datas[i])

    for i in range(a+b,a+b+c):
        for j in range(0,18):
            if(R3[j]==i):
                writer1.writerow(datas[R3[j]])
                break
            if j == 17:
                writer2.writerow(datas[i])    