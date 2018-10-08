import os
import os.path as op
import glob
import re
import time
import random
import numpy as np


ROOT_DIR = "/home/jiaxinchen/Project/3DXRetrieval"
DATASET = "SHREC14"
PROJMETHOD = "C4RAND"
MODALITY = "SHAPE"
NUMB_CHANNLES = 4
FLAGS_RANDOM_SAMPLING = False

def outputTxt(meta, OUT_DIR):
    numb_obj = len(meta)
    numb_chan = len(meta[0])
    #if not op.exists(OUT_DIR):
     #   os.mkdir(OUT_DIR)

    for i in range(numb_obj):
        paths_proj_file = meta[i][0][0]
        label = meta[i][0][1]
        classname = meta[i][0][2]
        shapename = meta[i][0][3]
        numb_samples_obj = np.zeros(NUMB_CHANNLES)
        for pp in range(NUMB_CHANNLES):
            numb_samples_obj[pp]=len(meta[i][pp][0])
        numb_samples_obj= int(numb_samples_obj.min())
        if FLAGS_RANDOM_SAMPLING:
            for j in range(numb_chan):
                indx_rand = random.randint(0, len(meta[i][j][0]))
                meta[i][j][0] = meta[i][j][0][indx_rand]
        if not op.exists(op.join(OUT_DIR, classname)):
            os.makedirs(op.join(OUT_DIR, classname))
        with open( op.join(OUT_DIR, classname, classname+"_"+shapename+".off.txt"), 'w' ) as f:
            f.write( classname+'\t'+label+'\n' )
            f.write( "sections\t"+str(NUMB_CHANNLES)+'\n')
            f.write("number_of_samples\t" + str(numb_samples_obj)+'\n')
            f.write('\n')
            for k in range( numb_samples_obj ):
                for p in range(NUMB_CHANNLES):
                    f.write(meta[i][p][0][k]+'\n' )
                f.write('\n')
        f.close()

        #with open( op.join(OUT_DIR, classname, DATASET+"_"+) )
        #for k in range(numb_samples_obj):
         #   print(0)


def main():
    ## Check if each category has NUMB_CHANNLES files
    time_start = time.time()
    pkg_filenames = os.listdir( op.join( ROOT_DIR, 'dataset', DATASET, MODALITY +"_"+PROJMETHOD ) )
    if len(pkg_filenames)!=0:
        class_names = os.listdir( op.join( ROOT_DIR, 'dataset', DATASET, MODALITY +"_"+PROJMETHOD, pkg_filenames[0] ) )
        if len(class_names)!=0:
            for ii in range( len(pkg_filenames) ):
                for jj in range( len(class_names) ):
                    chan_names = os.listdir( op.join( ROOT_DIR, 'dataset', DATASET, MODALITY +"_"+PROJMETHOD, pkg_filenames[ii], class_names[jj] ) )
                    if NUMB_CHANNLES!=len(chan_names):
                        raise Exception("Invalid files for %s in %s : exists %d channels (%d channels required)"%( class_names[jj], pkg_filenames[ii], len(chan_names), NUMB_CHANNLES ))
        else:
            raise Exception("Empty classes")
    else:
        raise Exception("No data packages")
    ## Read from .cla files and get the file names, labels for each projection channel per class
    numb_pkgs = len( pkg_filenames )
    with open( op.join(ROOT_DIR, 'dataset', DATASET, DATASET+"_"+MODALITY+".cla"), "r" ) as f:
        texts=f.readlines()
        numb_class, numb_shapes = texts[1].split()
        numb_class = int(numb_class)
        numb_shapes = int(numb_shapes)
        meta_info = list([])
        counter_class = 0

        for i in range(2,len(texts)):
            text=texts[i].split()
            if len(text)==3:
                counter_class = counter_class+1
                classname = text[0]
                label = str(counter_class)
                numb_samples_in_class = int(text[2])
                print("Processing %d-%d-th class: \"%s\"" % (counter_class, numb_class, classname))
            elif len(text)==1:
                shape_filename = "M"+text[0]
                meta_info_chan = list([])
                for i in range(NUMB_CHANNLES):
                    shape_projs = list([])
                    meta_info_pkg = list([])
                    for j in range(numb_pkgs):
                        abs_temp_imgfiles = glob.glob( op.join( ROOT_DIR, 'dataset', DATASET, MODALITY +"_"+PROJMETHOD, pkg_filenames[j], classname, "chan"+str(i+1), shape_filename+"*.jpg" ) )
                        temp_imgfiles = list([])
                        for k in range(len(abs_temp_imgfiles)):
                            indx_temp=re.search("/dataset", abs_temp_imgfiles[k] )
                            temp_imgfiles.extend([abs_temp_imgfiles[k][indx_temp.regs[0][0]:]])
                        #temp_imgfiles = map( i[(re.search(i, '/pkg')).regs[0]:], temp_imgfiles )
                        shape_projs.extend(temp_imgfiles)
                    meta_info_pkg.extend( [shape_projs, label, classname, shape_filename])
                    meta_info_chan.append(meta_info_pkg)
                meta_info.append(meta_info_chan)
    f.close()
    time_end = time.time()
    print("Take %f sec"%(time_end-time_start))
    ## Output the file names and labels to .txt files
    OUT_DIR = op.join(ROOT_DIR,'data', DATASET, DATASET+"_"+MODALITY+"_"+PROJMETHOD+"_Lists/")
    outputTxt(meta_info, OUT_DIR)


if __name__ == '__main__':
    main()
