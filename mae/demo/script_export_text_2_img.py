f_text = open("Concepts81.txt","r")
ar_text = []
while True:
    line = f_text.readline()
    if line == "":
        break
    line = line.strip()
    ar_text.append(line)
f_text.close()

f_img_name = open("train_filenames.txt","r")
ar_img = []

while True:
    line = f_img_name.readline()
    if line == "":
        break
    line  = line.strip()
    ar_img.append(line)
f_img_name.close()

f_i2t = open("true_img2text","r")
fo1 = open("img2txtx","w")

for i in xrange(2):
    line = f_i2t.readline().strip()
    print line
    fo1.write("%s\n"%ar_img[int(line)])
    line  = f_i2t.readline().strip()
    parts = line.split(" ")[:-1]
    for t in parts:
        fo1.write("%s "%ar_text[int(t)])
    fo1.write("\n")



f_i2t = open("true_text2img","r")
fo1 = open("txt2imgx","w")

for i in xrange(2):
    line = f_i2t.readline().strip()
    fo1.write("%s\n"%ar_text[int(line)])
    line  = f_i2t.readline().strip()
    parts = line.split(" ")[:-1]
    for t in parts:
        fo1.write("%s "%ar_img[int(t)])
    fo1.write("\n")



