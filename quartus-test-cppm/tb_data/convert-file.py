
outFile = open ("data_new.txt","w")

data = []

with open('data_lauri.txt', 'r') as fp:
    for idx, line in enumerate(fp):
      line=line.replace("\n","")
      data.append(str(float(line)/16))

      if idx >= 4:

          outFile.write("{}\n".format(" ".join(data)) )
          data.pop(0)
