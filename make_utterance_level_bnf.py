import numpy as np


defaultLocation="exp/make_bnf/"
fileNamePrefix="raw_bnfeat_cv_finaltest_nz_hires."
defaultSuffix=".txt"

completeDict=dict()
numberOfFrames=dict()
for i in range(1, 21):
    name=defaultLocation+fileNamePrefix + str(i) + defaultSuffix
    print ("==============InputFile:",name,"=================")
    print("READING")
    myDict=dict()
    with open(name) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        uttId=""
        uttEnd=0
        allFrames=[]
        count=0
        for line in content:
            if line.find("sample") >= 0:
                tempSplit=line.split()
                uttId=tempSplit[0]
                #print(uttId)
            else:
                count=count+1
                t=line.split()
                bnfs= [t[p] for p in range(300)]
                if len(t)==301:
                    #print("LastFrame")
                    uttEnd=1
                
                allFrames.append(bnfs)
            
            if uttEnd==1:
                allFramesNumpyTemp=np.array(allFrames)
                allFramesNumpy=allFramesNumpyTemp.astype(float)
                myDict[uttId]=allFramesNumpy
                numberOfFrames[uttId]=count;
                #print(allFramesNumpy.shape)
                allFrames=[]
                count=0
                uttEnd=0
                uttId=""
        
    averDict=dict()
    for key,value in myDict.items():
        averDict[key]=np.mean(value, axis=0)
        
    outputFileName=defaultLocation+"test/"+fileNamePrefix + str(i) + defaultSuffix    
    print ("==============OutputFile:",outputFileName,"=================")
    print("Writing")
    for key,value in averDict.items():
        times=numberOfFrames[key]
        #print(key)
        with open(outputFileName, "a") as file:
            temp=value.astype(np.float16)
            string_to_write=key+"  [\n"
            file.write(string_to_write)
            for x in range(times):
                string_to_write=" "
                #print(x+1)
                for dim in range(300):
                    if x!=times-1:
                        if dim!=299:
                            #print(value[i])
                            string_to_write=string_to_write+str(temp[dim])+" "
                        else:
                            string_to_write=string_to_write+str(temp[dim])+"\n"
                    else:
                        if dim!=299:
                            string_to_write=string_to_write+str(temp[dim])+" "
                        else:
                            string_to_write=string_to_write+str(temp[dim])+" ]\n"
                file.write(string_to_write)

    
print("All Set")


# run.pl JOB=1:30 exp/make_bnf/test/log/convert.JOB.log copy-feats ark,t:exp/make_bnf/test/raw_bnfeat_cv_finaltest_nz_hires.JOB.txt ark,scp:exp/make_bnf/test/arks/raw_bnfeat_cv_finaltest_nz_hires.JOB.ark,exp/make_bnf/test/arks/raw_bnfeat_cv_finaltest_nz_hires.JOB.scp



# for n in $(seq 30); do  cat exp/make_bnf/test/arks/raw_bnfeat_cv_finaltest_nz_hires.$n.scp; done > exp/make_bnf/test/arks/feats.scp
