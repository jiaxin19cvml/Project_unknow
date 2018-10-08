import os
import os.path as op

ROOT_DIR = "/home/jiaxinchen/Project/3DXRetrieval"
DATASET = "SHREC14"
PROJMETHOD = "C4RAND"
MODALITY = "SHAPE"
NUMB_CHANNLES = 4

class_files = os.listdir( op.join(ROOT_DIR, "data", DATASET, DATASET+"_"+MODALITY+"_"+PROJMETHOD+"_Lists") )
numb_class = len(class_files)
with open( op.join(ROOT_DIR, "data", DATASET, DATASET+"_"+MODALITY+"_"+PROJMETHOD+"_Train_Lists.txt" ), "w" )as f:
    for i in range(numb_class ):
        txtFileNames = os.listdir( op.join(ROOT_DIR, "data", DATASET, DATASET+"_"+MODALITY+"_"+PROJMETHOD+"_Lists", class_files[i]) )
        numb_texts = len(txtFileNames)
        if numb_texts>0:
            with open( op.join(ROOT_DIR, "data", DATASET, DATASET+"_"+MODALITY+"_"+PROJMETHOD+"_Lists", class_files[i], txtFileNames[0]), "r" ) as temp_f:
                texts=temp_f.readlines()
                first_line = texts[0].split()
                label = first_line[1]
            temp_f.close()
        for j in range(numb_texts):
            f.write( op.join( "/data", DATASET,  DATASET+"_"+MODALITY+"_"+PROJMETHOD+"_Lists", class_files[i], txtFileNames[j]) )
            f.write("\t")
            f.write(label)
            f.write("\n")
f.close()


