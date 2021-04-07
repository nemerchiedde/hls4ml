
outFile = open ("data_hls4lformat.txt","w")

with open('data.txt', 'r') as fp:
    for line in fp:
      line=line.replace("\n","")
      data=(float(line)/16)
      outFile.write("{} {} {} {} {}\n".format(data,data,data,data,data) )
