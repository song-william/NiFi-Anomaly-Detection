#!/usr/bin/python
import os

dirPathList = ['/Users/wsong/Desktop/nifi/nifi-0.6.1/content_repository', '/Users/wsong/Desktop/nifi/nifi-0.6.1/flowfile_repository', '/Users/wsong/Desktop/nifi/nifi-0.6.1/provenance_repository']
for dirPath in dirPathList:
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath+"/"+fileName)
print "NIFI repositories are cleared"
