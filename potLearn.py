#!/bin/python

import potlist as pl
# import external libraries
import numpy as np
import os, sys, pickle
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv

#scikit libraries
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
# for now, input is a directory containing some number of potentials (e.g. set.01)
CLASSIFIER 	 = "SVM"
Nrepeat 	 = 10
trim = 100.0
if len(sys.argv) != 1:

	database = pl.potList()
	database.find_in_path(sys.argv[1::])
	
	with open('database.pickle', 'wb') as f:
		pickle.dump(database, f)

else:
	
	with open('database.pickle', 'rb') as f:
		database = pickle.load(f)
		

print "Initial size: ", database.get_length()
print "Trimming by rms < ", trim
database.trim_list_by_rms(trim)
print "New size: ", database.get_length()
print "Trimming empty data"
database.trim_list_by_empty_subs(["a_bcc", "a_fcc", "a_hcp", "c_hcp", "a_wti", "c_wti", "a_bw", "a_bta", "C11", "C12", "C44", "dE_fcc", "dE_hcp", "dE_wti", "dE_bta", "dE_bw", "vacfor", "vacmig", "vacact"])
print "New size: ", database.get_length()
database.trim_by_unique()
print "Removing duplicates..."
print "New size: ", database.get_length()
#print "Full database: "
#print "Mean RMS error, std dev:"
#print database.global_rms_error()
#print "\t Worst potential:"
#print "\t", database.get_max_rms(), database.get_max_rms_name()

# split database into meam and eam parts

eambase, meambase = database.split_by_type()

#print "EAM database: "
#print "Mean RMS error, std dev:"
#print eambase.global_rms_error()
#print "\t Worst EAM potential:"
#print "\t", eambase.get_max_rms(), eambase.get_max_rms_name()

#print "MEAM database: "
#print "Mean RMS error, std dev:"
#print meambase.global_rms_error()
#print "\t Worst MEAM potential:"
#print "\t", meambase.get_max_rms(), meambase.get_max_rms_name()

l1 = matplotlib.patches.Patch(color='red', label='EAM', alpha=1.0)
l2 = matplotlib.patches.Patch(color='green', label='MEAM', alpha=1.0)
l3 = matplotlib.patches.Patch(color='blue', label='both', alpha=1.0)

a_bccs = (database.get_prop_rms('a_bcc'), eambase.get_prop_rms('a_bcc'), meambase.get_prop_rms('a_bcc'))
a_fccs = (database.get_prop_rms('a_fcc'), eambase.get_prop_rms('a_fcc'), meambase.get_prop_rms('a_fcc'))
a_hcps = (database.get_prop_rms('a_hcp'), eambase.get_prop_rms('a_hcp'), meambase.get_prop_rms('a_hcp'))
c_hcps = (database.get_prop_rms('c_hcp'), eambase.get_prop_rms('c_hcp'), meambase.get_prop_rms('c_hcp'))
a_wtis = (database.get_prop_rms('a_wti'), eambase.get_prop_rms('a_wti'), meambase.get_prop_rms('a_wti'))
c_wtis = (database.get_prop_rms('c_wti'), eambase.get_prop_rms('c_wti'), meambase.get_prop_rms('c_wti'))
a_bws = (database.get_prop_rms('a_bw'), eambase.get_prop_rms('a_bw'), meambase.get_prop_rms('a_bw'))
a_btas = (database.get_prop_rms('a_bta'), eambase.get_prop_rms('a_bta'), meambase.get_prop_rms('a_bta'))

C11s = (database.get_prop_rms('C11'), eambase.get_prop_rms('C11'), meambase.get_prop_rms('C11'))
C12s = (database.get_prop_rms('C12'), eambase.get_prop_rms('C12'), meambase.get_prop_rms('C12'))
C44s = (database.get_prop_rms('C44'), eambase.get_prop_rms('C44'), meambase.get_prop_rms('C44'))
dE_fccs = (database.get_prop_rms('dE_fcc'), eambase.get_prop_rms('dE_fcc'), meambase.get_prop_rms('dE_fcc'))
dE_hcps = (database.get_prop_rms('dE_hcp'), eambase.get_prop_rms('dE_hcp'), meambase.get_prop_rms('dE_hcp'))
dE_wtis = (database.get_prop_rms('dE_wti'), eambase.get_prop_rms('dE_wti'), meambase.get_prop_rms('dE_wti'))
dE_bws = (database.get_prop_rms('dE_bw'), eambase.get_prop_rms('dE_bw'), meambase.get_prop_rms('dE_bw'))
dE_btas = (database.get_prop_rms('dE_bta'), eambase.get_prop_rms('dE_bta'), meambase.get_prop_rms('dE_bta'))

alls = (database.get_prop_rms('a_bcc'), database.get_prop_rms('a_fcc'), database.get_prop_rms('a_hcp'),
	database.get_prop_rms('c_hcp'), database.get_prop_rms('a_wti'), database.get_prop_rms('c_wti'),
	database.get_prop_rms('a_bw'), database.get_prop_rms('a_bta'),
	database.get_prop_rms('C11'), database.get_prop_rms('C12'), database.get_prop_rms('C44'),
        database.get_prop_rms('dE_fcc'), database.get_prop_rms('dE_hcp'), database.get_prop_rms('dE_wti'), database.get_prop_rms('dE_bw'),
        database.get_prop_rms('dE_bta'), database.get_prop_rms('vacfor'), database.get_prop_rms('vacmig'), database.get_prop_rms('vacact'))

eams = (eambase.get_prop_rms('a_bcc'), eambase.get_prop_rms('a_fcc'), eambase.get_prop_rms('a_hcp'),
	eambase.get_prop_rms('c_hcp'), eambase.get_prop_rms('a_wti'), eambase.get_prop_rms('c_wti'),
	eambase.get_prop_rms('a_bw'), eambase.get_prop_rms('a_bta'),
	eambase.get_prop_rms('C11'), eambase.get_prop_rms('C12'), eambase.get_prop_rms('C44'),
        eambase.get_prop_rms('dE_fcc'), eambase.get_prop_rms('dE_hcp'), eambase.get_prop_rms('dE_wti'), eambase.get_prop_rms('dE_bw'),
        eambase.get_prop_rms('dE_bta'), eambase.get_prop_rms('vacfor'), eambase.get_prop_rms('vacmig'), eambase.get_prop_rms('vacact'))

meams= (meambase.get_prop_rms('a_bcc'), meambase.get_prop_rms('a_fcc'), meambase.get_prop_rms('a_hcp'),
	meambase.get_prop_rms('c_hcp'), meambase.get_prop_rms('a_wti'), meambase.get_prop_rms('c_wti'),
	meambase.get_prop_rms('a_bw'), meambase.get_prop_rms('a_bta'),
	meambase.get_prop_rms('C11'), meambase.get_prop_rms('C12'), meambase.get_prop_rms('C44'),
        meambase.get_prop_rms('dE_fcc'), meambase.get_prop_rms('dE_hcp'), meambase.get_prop_rms('dE_wti'), meambase.get_prop_rms('dE_bw'),
        meambase.get_prop_rms('dE_bta'), meambase.get_prop_rms('vacfor'), meambase.get_prop_rms('vacmig'), meambase.get_prop_rms('vacact'))

# here we construct the cross validation error versus number of properties considered to establish how many
# crystal properties we need to distingush between EAM and MEAM

full_subs = ["a_bcc", "a_fcc", "a_hcp", "c_hcp", "a_wti", "c_wti", "a_bw", "a_bta", "C11", "C12", "C44", "dE_fcc", "dE_hcp", "dE_wti", "dE_bw", "dE_bta", "vacfor", "vacmig", "vacact"]

N = len(full_subs) 
ind = np.arange(N)
width = 1./6

CLASSIFIERS 	 = {'SVM':svm.SVC(probability=True, kernel='rbf'), 'TREE':tree.DecisionTreeClassifier(max_depth=5), 'NN':neural_network.MLPClassifier(alpha=1)}
CLASSIFIER_NAMES = {'SVM':"RBF SVM", 'TREE':"Decision Tree", 'NN':"Neural Network"}

best = 0
avscores = np.zeros(len(full_subs))
avstdevs = np.zeros(len(full_subs))
for i in range(Nrepeat):
	scores = []
	square = []
	print i
	for Nsub in range(1,len(full_subs)+1):
		
		subs = full_subs[:Nsub]
	
		eamerrs  =  eambase.get_err_lists_subs(subs)
		meamerrs = meambase.get_err_lists_subs(subs)
		nprops = len(subs)
	
		samsiz = len(eamerrs)
		eamidx  = range(len(eamerrs))
		meamidx = range(len(meamerrs))
		
		eamsel   = np.random.choice(eamidx,  samsiz)
		meamsel  = np.random.choice(meamidx, samsiz)
		eamsub   = [eamerrs[i] for i in eamsel]
		meamsub  = [meamerrs[i] for i in meamsel]
		
		X = sum([eamsub, meamsub], [])
		Y = sum([[0 for i in range(len(eamsub))],[1 for i in range(len(meamsub))]], [])
		
		X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.1)
		
	#	gm = 300
	#	lp = np.linspace(-gm, gm, 5)
	#	grid = np.meshgrid(*[lp for i in range(nprops)], indexing='xy', sparse=False)[0]
		
		
#		clf = svm.SVC(probability=True, kernel='rbf')
		clf = CLASSIFIERS[CLASSIFIER]
		clf.fit(X_train, Y_train)
		
		score = clf.score(X_test, Y_test)

		if Nsub == len(full_subs):
			if score > best:
				clf_best = clf
				best = score

		scores.append(score)
		square.append(score**2)
#		print "Cross-validation score using ", Nsub, " properties: ", score
	avscores += np.array(scores)
	avstdevs += np.array(square) 

avstdevs  = np.sqrt((1./Nrepeat)*(avstdevs-(avscores**2)/(Nrepeat)))
avscores /= (Nrepeat)

print "Exporting trimmed database to database.csv ..."
with open('database.csv', 'w') as fo:
	writer = csv.writer(fo, lineterminator='\n')
	for pot in database.plist:
		writer.writerow(np.concatenate(([pot.get_type_int()], pot.get_err_list_sub(full_subs))))

#eamerrs   = eambase.get_err_lists()
#meamerrs  = meambase.get_err_lists()
#eamerrs  =  eambase.get_err_lists_subs(["C11", "C12", "C44", "dE_fcc", "dE_hcp", "dE_wti", "dE_bta", "dE_bw", "vacfor", "vacmig", "vacact"])
#meamerrs = meambase.get_err_lists_subs(["C11", "C12", "C44", "dE_fcc", "dE_hcp", "dE_wti", "dE_bta", "dE_bw", "vacfor", "vacmig", "vacact"])
#nprops   = 11
#eamerrs  =  eambase.get_err_lists_subs(["C11", "C12", "C44", "dE_fcc", "dE_hcp", "dE_wti", "dE_bta", "dE_bw"])
#meamerrs = meambase.get_err_lists_subs(["C11", "C12", "C44", "dE_fcc", "dE_hcp", "dE_wti", "dE_bta", "dE_bw"])

# now compute probability densities on various planes for visualization
gm = 50
lp = np.linspace(-gm, gm, 300)
xx = np.meshgrid(lp, lp)

#C11 vs C44
#clf1  = svm.SVC(probability=True, kernel='poly', degree=2, coef0=0)
#clf1  = svm.SVC(probability=True, kernel='rbf')
#subs1 = ['C11', 'C44']
#eam1  =  eambase.get_err_lists_subs(subs1)
#meam1 = meambase.get_err_lists_subs(subs1)

#X1 = sum([eam1, meam1], [])
#Y1 = sum([[0 for i in range(len(eam1))], [1 for i in range(len(meam1))]], [])
#X1_T, X1_t, Y1_T, Y1_t = cross_validation.train_test_split(X1, Y1, test_size = 0.99, random_state = 0)
#clf1.fit(X1_T, Y1_T)
#print xx[0].ravel()
#Z1 = clf1.predict_proba(np.c_[xx[0].ravel(), xx[1].ravel()])[:,1]
Z1 = clf_best.predict_proba(np.c_[
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     xx[0].ravel(), np.zeros(len(xx[0].ravel())), xx[1].ravel(),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel()))])[:,1]

#C12 vs C44
#clf2  = svm.SVC(probability=True, kernel='poly', degree=2, coef0=0)
#clf2  = svm.SVC(probability=True, kernel='rbf')
#subs2 = ['C12', 'C44']
#eam2  =  eambase.get_err_lists_subs(subs2)
#meam2 = meambase.get_err_lists_subs(subs2)

#X2 = sum([eam2, meam2], [])
#Y2 = sum([[0 for i in range(len(eam2))], [1 for i in range(len(meam2))]], [])
#X2_T, X2_t, Y2_T, Y2_t = cross_validation.train_test_split(X2, Y2, test_size = 0.99, random_state = 0)
#clf2.fit(X2_T, Y2_T)
#Z2 = clf2.predict_proba(np.c_[xx[0].ravel(), xx[1].ravel()])[:,1]
Z2 = clf_best.predict_proba(np.c_[
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())), xx[0].ravel(), xx[1].ravel(),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel()))])[:,1]
print Z2
#C11 vs C12
#clf3  = svm.SVC(probability=True, kernel='rbf')
#subs3 = ['C11', 'C12', 'C44']
#eam3  =  eambase.get_err_lists_subs(subs3)
#meam3 = meambase.get_err_lists_subs(subs3)
#xxx = np.meshgrid(lp, lp, lp)
#X3 = sum([eam3, meam3], [])
#Y3 = sum([[0 for i in range(len(eam3))], [1 for i in range(len(meam3))]], [])
#X3_T, X3_t, Y3_T, Y3_t = cross_validation.train_test_split(X3, Y3, test_size = 0.99, random_state = 0)
#clf3.fit(X3_T, Y3_T)
#print xxx[0].ravel()
#print np.zeros(len(lp))
Z3 = clf_best.predict_proba(np.c_[
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     xx[0].ravel(), xx[1].ravel(), np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel())),
     np.zeros(len(xx[0].ravel()))])[:,1]

# plot section #

plt.figure(1)
plt.subplot(321)

n2, bins2, patches2 = plt.hist( eambase.get_rms_list(), int(trim), range=(0,trim), normed=1, facecolor='red',   alpha=1.0)
n3, bins3, patches3 = plt.hist(meambase.get_rms_list(), int(trim), range=(0,trim), normed=1, facecolor='green', alpha=1.0)
n1, bins1, patches1 = plt.hist(database.get_rms_list(), int(trim), range=(0,trim), normed=1, facecolor='blue',  alpha=1.0)

plt.legend(handles=[l1, l2, l3])

plt.xlabel('RMS error (%)')
plt.ylabel('Frequency (arb.)')
plt.xlim([0,100])


plt.subplot(322)


plt.xlim([0,len(full_subs)-0.5])
plt.ylabel('Average RMS error (%)')
plt.bar(ind, alls, color='blue', alpha=1.0,width=1./6)
plt.bar(ind+width, eams, color='red', alpha=1.0, width=1./6)
plt.bar(ind+2*width, meams, color='green', alpha=1.0, width=1./6)
labels = ['$a_{bcc}$', '$a_{fcc}$', '$a_{hcp}$', '$c_{hcp}$', '$a_{\\omega Ti}$', '$c_{\\omega Ti}$', '$a_{\\beta W}$', '$a_{\\beta Ta}$', '$C_{11}$','$C_{12}$','$C_{44}$','$\\Delta E_{fcc}$', '$\\Delta E_{hcp}$', '$\\Delta E_{\\omega Ti}$', '$\\Delta E_{\\beta W}$', '$\\Delta E_{\\beta Ta}$',
	  '$E{_{f}}^{vac}$', '$E{_{m}}^{vac}$', '$E{_{a}}^{vac}$']
plt.xticks(ind+0.25, labels, rotation=45)

plt.subplot(323)
plt.title(CLASSIFIER_NAMES[CLASSIFIER] + " with " + str(Nrepeat) + " models")
plt.xlabel('# Properties')
plt.ylabel('10% Cross-\nvalidation score')
plt.xticks([i+1 for i in range(0,len(full_subs))],[str(i+1)+"\n"+labels[i] for i in range(0,len(full_subs))])
plt.xlim(0.5,len(full_subs)+0.5)
#plt.errorbar(np.arange(10)+1, avscores, yerr=avstdevs, lw=2)
plt.plot(np.arange(len(full_subs))+1, avscores, linewidth=2, color='k')
plt.fill_between(np.arange(len(full_subs))+1, avscores+avstdevs, avscores-avstdevs, facecolor='blue', alpha=0.5)
plt.grid()

plt.subplot(324)

plt.xlabel('C11 Error (%)')
plt.ylabel('C44 Error (%)')
plt.xlim(-gm, gm)
plt.ylim(-gm, gm)
plt.grid()
plt.axhline(y=0, xmin=-gm, xmax=gm, linewidth=1, color='k')
plt.axvline(x=0, ymin=-gm, ymax=gm, linewidth=1, color='k')
plt.contourf(xx[0], xx[1], Z1.reshape(xx[0].shape), 50, cmap=plt.cm.RdYlGn, alpha=0.5)
plt.scatter( eambase.get_prop_list('C11', 1), eambase.get_prop_list('C44', 1), color='red',     alpha=1.00, s=1.5, marker="v")
plt.scatter(meambase.get_prop_list('C11', 1),meambase.get_prop_list('C44', 1), color='green',   alpha=1.00, s=1.5, marker="^")
#print Y1_t
#plt.scatter(X1_t[:,0], X1_t[:,1], c=Y1_t)

plt.subplot(325)

plt.xlabel('C12 Error (%)')
plt.ylabel('C44 Error (%)')
plt.xlim(-gm, gm)
plt.ylim(-gm, gm)
plt.grid()
plt.axhline(y=0, xmin=-gm, xmax=gm, linewidth=1, color='k')
plt.axvline(x=0, ymin=-gm, ymax=gm, linewidth=1, color='k')
plt.contourf(xx[0], xx[1], Z2.reshape(xx[0].shape), 50, cmap=plt.cm.RdYlGn, alpha=0.5)
plt.scatter( eambase.get_prop_list('C12', 1), eambase.get_prop_list('C44', 1), color='red',     alpha=1.00, s=1.5, marker="v")
plt.scatter(meambase.get_prop_list('C12', 1),meambase.get_prop_list('C44', 1), color='green',   alpha=1.00, s=1.5, marker="^")

plt.subplot(326)

plt.xlabel('C11 Error (%)')
plt.ylabel('C12 Error (%)')
plt.xlim(-gm, gm)
plt.ylim(-gm, gm)
plt.grid()
plt.axhline(y=0, xmin=-gm, xmax=gm, linewidth=1, color='k')
plt.axvline(x=0, ymin=-gm, ymax=gm, linewidth=1, color='k')
plt.contourf(xx[0], xx[1], Z3.reshape(xx[0].shape), 50, cmap=plt.cm.RdYlGn, alpha=0.5)
plt.scatter( eambase.get_prop_list('C11', 1), eambase.get_prop_list('C12', 1), color='red',     alpha=1.00, s=1.5, marker="v")
plt.scatter(meambase.get_prop_list('C11', 1),meambase.get_prop_list('C12', 1), color='green',   alpha=1.00, s=1.5, marker="^")

figure = plt.gcf()
figure.set_size_inches(16,14)
figure.tight_layout()
#figure.suptitle(CLASSIFIER_NAMES[CLASSIFIER] + " with " + str(Nrepeat) + " models")
plt.savefig('./potlearn.png', dpi=300)
plt.show()
