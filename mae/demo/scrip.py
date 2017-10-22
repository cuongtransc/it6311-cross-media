fi1 = open("true_text_2_img.dat","r")
fi2 = open("line_ids_traini.dat","r")

fo = open("true_text2img","w")

ar_ids = []
while True:
    line_id = fi2.readline()
    if line_id=="":
        break
    ar_ids.append(line_id.strip())
fi2.close()

v =fi1.readline()

fo.write("%s"%v)
line = fi1.readline().strip()
parts = line.split(" ")
for id in parts[:-1]:
    fo.write("%s "%ar_ids[int(id)-1])
fo.write("\n")

v =fi1.readline()

fo.write("%s"%v)
line = fi1.readline().strip()
parts = line.split(" ")
for id in parts[:-1]:
    fo.write("%s "%ar_ids[int(id)-1])
fo.write("\n")

fo.close()
fi1.close()
