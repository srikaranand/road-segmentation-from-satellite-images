# This program converts the csv output to images

import csv
from PIL import Image

n_images = 1

images=[]
for _ in range(n_images):
  images.append(Image.new('RGB',[593,593]))
  #print(img1.mode)

with open('output.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        records = row[0].split('_')
        value = int(row[1])*255
        i = int(records[0])-1
        x = int(records[1])
        y = int(records[2])
        img = images[i].load()
        img[x,y] = (value, value,value)

img_num=0
for img1 in images:
  output = Image.new(img1.mode,img1.size)
  output_ = output.load()
  for i in range(img1.size[0]):
    for j in range(img1.size[1]):
      x = (i//16) * 16
      y = (j//16) * 16
      output_[i,j] = img[x,y]
  img_num=img_num+1
  output.save('outputImages/'+str(img_num)+'.png')