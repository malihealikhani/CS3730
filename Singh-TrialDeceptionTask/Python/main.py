import nltk
import os
import sys
import shutil
import subprocess
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk.tree import Tree
import re



from itertools import combinations

from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#handle parsing of the data
#copy all the files to the correct location.
srcPath = "../Data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Transcription"
cwd = os.getcwd()
rstPath = "gCRF_dist/texts"
pdtbPath = "pdtb-parser/examples/data"

fh = open("gCRF_dist/texts/fileList.txt", 'w')
y = []
x = []
fnames = []
for f in os.listdir(srcPath):
    for p in os.listdir(os.path.join(srcPath, f)):
        #copy the file to the correct locations
        src = os.path.join(srcPath, f, p)
        dst1 = os.path.join(rstPath, p)
        dst2 = os.path.join(pdtbPath, p)
        shutil.copyfile(src, dst1)
        shutil.copyfile(src,dst2)
        fh.write(os.path.join(cwd,dst1) + "\n")

        y.append( int('truth' in p) )
        fnames.append(p[:-3] + "mp4")
        fs = open(src, 'r')
        a = fs.read()
        x.append(a)

fh.close()
df = pd.read_csv("../Data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv", index_col="id").reindex(fnames).drop(columns=['class'])


result = subprocess.run(["bash", "rst_parse.sh"])
#print(result)
result = subprocess.run(["bash", "pdtbParse.sh"])
#print(result)

#looc and svm
#lets just do svm and the behavioral features

#loop through the RST trees and extract info
rstFeats = []
for fn in fnames:
    #read in file and save tree
    path = os.path.join("gCRF_dist/texts/results", fn[:-3] +"txt.tree")
    t = Tree.fromstring( open(path, 'r').read(), leaf_pattern="_!.*!_")

    toAppend = []
    for prod in t.productions():
        toAdd = re.sub( " ", "=", str(prod))
        if("_!" not in toAdd):
            toAppend.append(toAdd)

    rstFeats.append(" ".join(toAppend))

print(len(rstFeats))
pdtbFeats = []
for fn in fnames:
    path = os.path.join("pdtb-parser/examples/data/output", fn[:-3] +"txt.pipe")
    data = open(path, 'r').read()
    toAppend = []
    for line in data.split("\n"):
        if(len(line)>1):
            vals = line.split("|")
            toAppend.append( vals[0]+"="+vals[11])
    pdtbFeats.append(" ".join(toAppend))
#tokenzie the data.
for i in range(len(x)):
    sub = [word_tokenize(a) for a in sent_tokenize(x[i])]
    x[i] = " ".join([j for p in sub for j in p])

unigram = CountVectorizer( ngram_range=(1,1))
bigram = CountVectorizer(ngram_range=(2,2))
rstGram = CountVectorizer(token_pattern="\S+")
pdtbGram = CountVectorizer(token_pattern="\S+")

uniX = unigram.fit_transform(x).toarray()
biX = bigram.fit_transform(x).toarray()
behavFeatures = df.to_numpy()
rstX = rstGram.fit_transform(rstFeats).toarray()
pdtbX = pdtbGram.fit_transform(pdtbFeats).toarray()

print(rstX.shape)
print(pdtbX.shape)

print("Printing out the analysis of RST and PDTB features")

rstSum = np.zeros((2,rstX.shape[1]))
pdtbSum = np.zeros((2,pdtbX.shape[1]))

for idx,c in enumerate(y):
    rstSum[c] += rstX[idx]
    pdtbSum[c] += pdtbX[idx]


rstLabels = np.array([x[1] for x in sorted([(v,k) for k,v in rstGram.vocabulary_.items()])])
k=5
indD = np.argpartition(rstSum[0], -k)[-k:]
rstD = list(reversed(sorted([( rstSum[0][i],rstLabels[i]) for i in indD])))
indT = np.argpartition(rstSum[1], -k)[-k:]
rstT = list(reversed(sorted([(rstSum[1][i], rstLabels[i]) for i in indT])))
for i in range(k):
    print( "& ".join([str(x) for x in [rstT[i][1], rstT[i][0], rstD[i][1], rstD[i][0]]]))

pdtbLabels = np.array([x[1] for x in sorted([(v,k) for k,v in pdtbGram.vocabulary_.items()])])
k=4
indD = list(reversed(np.argpartition(pdtbSum[0], -k)[-k:]))
pdtbD = list(reversed(sorted([( pdtbSum[0][i], pdtbLabels[i]) for i in indD])))

indT = list(reversed(np.argpartition(pdtbSum[1], -k)[-k:]))
pdtbT = list(reversed(sorted([(pdtbSum[1][i], pdtbLabels[i]) for i in indT])))
for i in range(k):
    print( "& ".join([str(x) for x in [pdtbT[i][1], pdtbT[i][0], pdtbD[i][1], pdtbD[i][0]]]))

raise SystemExit
featsets = dict(
    uniX = uniX,
    biX = biX,
    behavFeatures = behavFeatures,
    rstX = rstX,
    pdtbX = pdtbX,
)

names = list(featsets.keys())
for i in range(2, len(names)+1):
    for comb in combinations(names,i):
        featsets["_".join(comb)] = np.concatenate( [featsets[fset] for fset in comb], axis=1)


looc = list(LeaveOneOut().split(x,y))


fh = open("../Results/experimentResults.csv", 'w')
fh.write( ",".join(["Feature Set", "Model", "Accuracy", "Precision", "Recall", "F1"]) + "\n")

models = [LinearSVC(), DecisionTreeClassifier(), LogisticRegression()]

mFset = ""
mFsetVal = 0
mFsetModel = ""
for k in featsets:
    v = featsets[k]
    for model in models:
        print("Running experiment for %s %s" %(k, model))
        scores = cross_validate(model, v, y, cv = looc, scoring=('accuracy', 'precision', 'recall', 'f1'))

        acc = np.average(scores['test_accuracy'])
        pre = np.average(scores['test_precision'])
        rec = np.average(scores['test_recall'])
        f1 = np.average(scores['test_f1'])

        print("Acc    |Prec   |Recall  |F1")
        print("%s |%s |%.6s  |%s\n" %(acc,pre,rec,f1))

        fh.write(",".join([k, str(model), str(acc), str(pre), str(rec), str(f1)]) +"\n")


        if(acc > mFsetVal):
            mFset = k
            mFsetVal = acc
            mFsetModel = model

fh.close()
print(mFset, mFsetVal, mFsetModel)


