import os

directory = '../../../../nodeserver/data/grades'
for folder in os.listdir(directory):
    index = 1
    for filename in os.listdir(directory +'/' +folder):
        os.rename(directory+'/'+folder+'/'+filename, directory+'/'+folder+'/'+ f"{folder}-{index:04d}.jpg")
        print(f"{folder}-{index:04d}")
        index+=1