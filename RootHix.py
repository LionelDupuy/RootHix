#!/usr/bin/python
# -*- coding: latin-1 -*-
"""	 Main authors: L. Dupuy
	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License along
	with this program; if not, write to the Free Software Foundation, Inc.,
	51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA."""
	
import numpy.random as rnd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import argrelextrema

#np.seterr(all = 'ignore')
#import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from xml.dom import minidom
import pylab as plt
#import pylab as plt
#plt.subplots_adjust( hspace=0.5 )

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore")

import wx
import os
# TODO
# incorporate physical scale into 1 - spatial coordinates / saved data and 2 - frequency data
# label axes with units
# why some results are not scale to Z = 1

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

pi = 3.14159
duration = 1
fs	= 100.0
frac_spect = 0.15						# fraction of the spectrum to be analysed
max_freq = 15.0							# max frequency shown is 1/T


		
class helical_transform:
	# Various settings
	path = ""
	RBF_param =[]
	freq_profile = []
	freq_bounds = []
	max_freq = max_freq
	f_mult = frac_spect*2.
	
	# physical and temporal size of the experiment
	scaling = [1.,1.]
	
	# Min and max value of z
	z_bounds = []
	
	# input / output
	RES = []
	IN = [[],[],[]]

	# raw data before transformed
	X_base = []
	Y_base = []
	Z_base = []

	# overall trend of the trajectory
	bckg_x = []
	bckg_y = []
	bckg_z = []	
	

	def spirl(self,k,t,phi = 0):
		if k != 0:
			x = np.exp( -1j* (t*2*pi*k +phi))
			y = np.sin(t*2*pi*k + phi) + 1j* np.cos(t*2*pi*k + phi) #np.conj(np.exp( 1j*(t*2*pi*k + phi) ))
			z = 0*t
		else:
			x = 0*t
			y = 0*t
			z = np.sqrt(3)*t	
		RES = np.concatenate((np.array([x]),np.array([y]),np.array([z])),axis = 0)
		return RES	
	
	def proj(self, k, x, y, z):
		dltot = np.sqrt((self.X_base[1:]-self.X_base[:-1])**2 + (self.Y_base[1:]-self.Y_base[:-1])**2 + (self.Z_base[1:]-self.Z_base[:-1])**2)
		dl = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2 + (z[1:]-z[:-1])**2)
		exdl = np.concatenate((np.array([0]),dl))#, axis = 1)
		exdltot = np.concatenate((np.array([0]),dltot))
		cuml = np.cumsum(exdl, dtype=float)
		cumltot = np.cumsum(exdltot, dtype=float)
		L = np.sum(dl)
		Ltot = np.sum(dltot)
		n = len(x)
		ntot = len(self.X_base)
		T = L / Ltot
		if k==0:
			A = 1./ T**2 * self.spirl(k,cuml/Ltot)*np.array([x,y,(z - z[0])/(z[-1]- z[0])]) / float(ntot) 
		else:
			A = 1. / T * self.spirl(k,cuml/Ltot)*np.array([x,y,z/(z[-1]- z[0])]) / float(ntot) 
		return np.sum(np.sum(A,axis=0))
		
	def transform(self,x, y, z):
		'''
		Return coefficients for a helix achieved for 
		* unit duration T=1 and unit length z_max - Z_min =1
		* OR for a cropped version of the initial T=1 trajectory'''

		freq = np.arange(0.01,self.max_freq,self.f_mult * 1./self.max_freq)
#		freq = np.arange(frac_spect,len(x)*frac_spect,f_mult)
		ak = []
		mk = []
		i=0
		for k in freq:
			ak.append(self.proj(k, x, y, z))
			mk.append(self.proj(-k, x, y, z))
			i += 1
			
				
		ab0 = self.proj(0, x, y, z)
		freq = freq
		return [freq, ak, ab0, mk]
	
#################################################################################################
# gaussian mixture regression of frequency data
#################################################################################################
	def RBF(self,x):
		c1, s1, c2, s2, m2 = x
		Y2 = c1*np.exp(-(self.RES[0]**2)/s1) + c2*np.exp(-((self.RES[0]-m2)**2)/s2) 
		return Y2
	def RBF_sim(self,x, X):
		c1, s1, c2, s2, m2 = x
		Y2 = c1*np.exp(-(X**2)/s1) + c2*np.exp(-((X-m2)**2)/s2) 
		return Y2		
	def RBF_error(self,x):
		pred = self.RBF(x)	
		profile = np.real(self.freq_profile)#+ np.real(self.RES[3] * np.conj(self.RES[3]))
		E = np.sum((pred - profile) **2)
		return E
	def RBF_fit(self, mini = 0):	
		x0 = [1,1,1,1,5]
		nf = (self.RES[3] * np.conj(self.RES[3]))
		pf = ((self.RES[1] * np.conj(self.RES[1])))
		max_nf =  np.real(nf).max()
		max_pf = np.real(pf).max()

		
		is_neg_freq = False
		if mini == 0:
			if (   max_pf >= max_nf  ):
				self.freq_profile = np.real(pf )
				a = self.freq_profile >= max_pf
				b = np.array(range(len(self.freq_profile)))
				c = (a*b).max() * self.max_freq / len(a)
				x0 = [1,1,max_pf,1,c]
			else:
				self.freq_profile = np.real(nf)
				a = self.freq_profile >= max_nf
				b = np.array(range(len(self.freq_profile)))
				c = (a*b).max() * self.max_freq / len(a)
				x0 = [1,1,max_nf,1,c]
				is_neg_freq = True
		else:
			if mini>0:
				self.freq_profile = np.real(self.RES[1] * np.conj(self.RES[1]) )
				index = int(mini / self.max_freq * len(self.freq_profile))
				x0 = [1,1,self.freq_profile[index],1,mini]

				
			else:
				self.freq_profile = np.real(self.RES[3] * np.conj(self.RES[3]) )
				index = int(mini / self.max_freq * len(self.freq_profile))
				x0 = [1,1,self.freq_profile[index],1,np.abs(mini)]
				is_neg_freq = True
		res = minimize(self.RBF_error, x0, method='Powell', tol=1e-4)
		self.RBF_param = res.x
#‘Nelder-Mead’ (see here)
#‘Powell’ (see here)
#‘CG’ (see here)
#‘BFGS’ (see here)
#‘Newton-CG’ (see here)
#‘L-BFGS-B’ (see here)
#‘TNC’ (see here)
#‘COBYLA’ (see here)
#‘SLSQP’ (see here)
#‘dogleg’ (see here)
#‘trust-ncg’ (see here)
		if mini < 0 or (mini==0 and max_pf < max_nf):
			self.RBF_param[4] = -self.RBF_param[4]
		
#################################################################################################
# test function
#################################################################################################
	def test_function(self):
		w	= -5.
		phi = 0.2			#5.*pi/10.
		a0	= 0.2
		a1	= 0
		r0	= 0.005
		t	= np.arange(0,1.0, 1./fs)

		n = len(t)
		rd = rnd.random([n,3])
		rd = np.cumsum(rd, dtype=float, axis = 1)
		X = a0*np.cos(2*pi*w*t+phi) + a1*np.cos(4*2*pi*w*t+phi) + r0*rd[:,0]# + 1.5*t
		Y = a0*np.sin(2*pi*w*t+phi) + a1*np.sin(4*2*pi*w*t+phi) + r0*rd[:,1]
		Z = 3*t + r0*rd[:,2]
		return [X,Y,Z]
#################################################################################################
# I/O
#################################################################################################
		
	def read(self, path):
		self.path = path
		self.f_mult = frac_spect*2.
		X = []
		Y = []
		Z = []
		buf, file_ext = os.path.splitext(path)
		if file_ext == ".txt":
			f = open(path, "r")
			for line in f:
				row = line.split(",")
				if len(row) >0 and len(row) < 5:
					X.append(float(row[0])*self.scaling[0])
					Y.append(float(row[1])*self.scaling[0])
					Z.append(float(row[2])*self.scaling[0])
					


			f.close()
			X = np.array(X)
			Y = np.array(Y)
			Z = np.abs(np.array(Z))

		
		elif  file_ext == ".xml":
			xmldoc = minidom.parse(path)
			itemlist = xmldoc.getElementsByTagName('pos')

			for s in itemlist:
				row = (getText(s.childNodes)).split(" ")
				X.append(float(row[0])*self.scaling[0])
				Y.append(float(row[1])*self.scaling[0])
				Z.append(float(row[2])*self.scaling[0])
			X = np.array(X)
			Y = np.array(Y)
			Z = np.abs(np.array(Z))

		
		Z = Z-Z.min()
		if Z[0] > Z[-1]:
			Z = Z[::-1]
		splx = UnivariateSpline(Z, X)
		sply = UnivariateSpline(Z, Y)
		splx.set_smoothing_factor(1000000)
		sply.set_smoothing_factor(1000000)

		spacing_knots = np.abs(Z[0]-Z[-1]) / 10.
		T = np.array([(Z[0]+Z[-1])/2.])#sply.get_knots()#np.linspace(Z[0], Z[-1], np.abs(Z[0]-Z[-1]) / spacing_knots)

		splx = LSQUnivariateSpline(Z, X, T, bbox = [Z[0], Z[-1]])
		sply = LSQUnivariateSpline(Z, Y, T, bbox = [Z[0], Z[-1]])

		bckg_x = splx(Z)
		bckg_y = sply(Z)
		zz = [0]
		for k in range(1,len(bckg_x)):
			if np.isnan(bckg_x[k]):
				bckg_x[k] = bckg_x[k-1]
				bckg_y[k] = bckg_y[k-1]
			zz.append(np.sqrt( (Z[k] - Z[k-1])**2 + (bckg_x[k] - bckg_x[k-1])**2 + (bckg_y[k] - bckg_y[k-1])**2) )
		zz = np.cumsum(zz)
		

		bckg_x = np.nan_to_num(bckg_x)
		bckg_y = np.nan_to_num(bckg_y)
		self.bckg_x = bckg_x - bckg_x[0]
		self.bckg_y = bckg_y - bckg_y[0]
		self.bckg_z = Z - Z.min()
		self.IN = [X -  bckg_x,Y - bckg_y, zz - zz.min()]

		self.X_base = np.array(self.IN[0])
		self.Y_base = np.array(self.IN[1])
		self.Z_base = np.array(self.IN[2])
		
		self.freq_bounds = [0, self.max_freq]#len(X) * frac_spect]
		self.z_bounds = [0, self.IN[2].max()]
		return self.IN
		
	def reload(self):
		if self.path == "":
			pass
		else:
			self.read(self.path)
		self.run()	
	def export_test(self, path, X,Y,Z):
		f = open(path, "w")
		for i in range(len(X)):
			row = str(X[i]) + " , " +str(Y[i]) + " , " +str(Z[i]) + "\n"
			f.write(row)
		f.close()
	def export_res(self, path):
		exp_file = (path.split('.'))[0] + "_res.csv"
		f = open(exp_file, "w")
		RES = self.RES
		
		f.write("Frequency Analysis\n")
		f.write("Freq,Power\n")
		spectr1 = np.real(self.RES[1] * np.conj(self.RES[1]))
		spectr2 = ( np.real(self.RES[3] * np.conj(self.RES[3]) ))[::-1] 
		freq = np.concatenate(((-RES[0]) [::-1] , RES[0]))
		spectrum = np.concatenate((spectr2,spectr1))
		for i in range(len(freq)):
			#print freq[i], " , ", spectrum[i]
			row = str(freq[i]) + " , " +str(spectrum[i]) + "\n"
			f.write(row)
			
		f.write("Growth Factor:," + str(RES[2]) + "\n")
		
		f.write("Helix Properties,\n")
		if self.RBF_param[4] <0:
			f.write("Rotation: , Anticlockwise\n")
		if self.RBF_param[4] >0:
			f.write("Rotation: , Clockwise\n")
		f.write("Frequency:," + str(self.RBF_param[4]) + "\n") 
		f.write("Amplitude Oscillation:," + str(np.sqrt(self.RBF_param[2])) + "\n") 
		
		f.close()
		
		# Export trajectory
		exp_file = (path.split('.'))[0] + "_traj.txt"		
		f = open(exp_file, "w")
		X, Y, Z = self.IN
		for i in range(len(X)):
			f.write(str(X[i]) + " , " + str(Y[i]) + " , " + str(Z[i])+"\n")
		f.close()
		
	def get_curvature(self):
		CURV = []
		X, Y, Z = self.IN
		for i in range(len(X)-2):
			X0 = X[i+0]
			X1 = X[i+1]
			X2 = X[i+2]
			Y0 = Y[i+0]
			Y1 = Y[i+1]
			Y2 = Y[i+2]
			Z0 = Z[i+0]
			Z1 = Z[i+1]
			Z2 = Z[i+2]
			
			V1 = np.array([X1-X0, Y1-Y0, Y1-Y0])
			V2 = np.array([X2-X1, Y2-Y1, Y2-Y1])
			L1 = np.sqrt(np.sum(V1*V1))
			L2 = np.sqrt(np.sum(V2*V2))
			
			cross = np.cross(V1,V2) / (L1*L2)
			curvature = np.sqrt(np.sum(cross*cross)) / (L1+L2)
			CURV.append(curvature)

		CURV = np.array(CURV)	
		P = argrelextrema(CURV, np.greater, order = 3)
		POS = np.array(P[0], int)
		
		return [CURV, POS]
#######################################################################################################
# transform
#######################################################################################################
	def run(self):
		if len(self.IN) < 3:
			X,Y,Z = self.test_function()
			self.IN = [X,Y,Z]	
		elif len(self.IN[0])<2:
			X,Y,Z = self.test_function()
			self.IN = [X,Y,Z]	
		self.RES = self.transform(self.IN[0],self.IN[1],self.IN[2])
		self.RBF_fit()


#############################################################################		
# Run the interface
#############################################################################
class CanvasPanel(wx.Panel):
	def __init__(self, parent):
		self.parent = parent
		self.IN = []
		self.RES = [[],[],[],[]]
		wx.Panel.__init__(self, parent)
		self.SetBackgroundColour((255,255,255))
		self.figure = Figure(facecolor='white')
		self.is_zaxis_inv = False
		self.figure2 = Figure(facecolor='white')
		self.canvas = FigureCanvas(self, -1, self.figure)
		self.toolbar1 = Toolbar(self.canvas)
		self.toolbar1.Realize()		
		self.canvas2 = FigureCanvas(self, -1, self.figure2)
		self.toolbar2 = Toolbar(self.canvas2)
		self.toolbar2.Realize()		
		

		
		#self.sizerb = wx.BoxSizer(wx.HORIZONTAL)
		# BUTTONS CROPPING / DRAWING
#		self.b1 = wx.Button(self, 1, "RELOAD", (20, 20))
#		self.Bind(wx.EVT_BUTTON, self.OnReload, self.b1)
#		self.b2 = wx.Button(self, 2, "TOP", (20, 20))
#		self.Bind(wx.EVT_BUTTON, self.OnCropTop, self.b2)
#		self.b3 = wx.Button(self, 3, "BOTTOM", (20, 20))
#		self.Bind(wx.EVT_BUTTON, self.OnCropBottom, self.b3)
#		self.b4 = wx.Button(self, 4, "SAVE", (20, 20))
#		self.Bind(wx.EVT_BUTTON, self.Onsave, self.b4)
#		self.sizerb = wx.BoxSizer(wx.HORIZONTAL)
#		self.sizerb.Add(self.b1, 1, wx.LEFT | wx.TOP )
#		self.sizerb.Add(self.b2, 1, wx.LEFT | wx.TOP )
#		self.sizerb.Add(self.b3, 1, wx.LEFT | wx.TOP )
#		self.sizerb.Add(self.b4, 1, wx.LEFT | wx.TOP )
		
		# SIZER
		# position of the HELIX
		self.slider_hel_position = wx.Slider(self, -1, 0, 0, 1000)
		self.sldmin = -1150
		self.sldmax = 1000
		self.sldpos = 0
		self.sldscale = 100. * self.parent.C.max_freq / 10.
		self.slider_hel_position.SetRange(self.sldmin, self.sldmax)
		self.Bind(wx.EVT_SCROLL_CHANGED, self.Onslider, self.slider_hel_position)
		
		# position of directional changes
		self.slider_curve_position = wx.Slider(self, -1, 0, 0, 1000)
		self.slider_curve_position.SetRange(0, 100)
		#self.Bind(wx.EVT_SCROLL_CHANGED, self.Onslider, self.slider_curve_position)		
		
		
		self.sizergraphs = wx.BoxSizer(wx.HORIZONTAL)
		self.sizer3D = wx.BoxSizer(wx.VERTICAL)
		self.sizer3D.Add(self.toolbar1, 1, wx.LEFT | wx.TOP, 0)#  | wx.GROW
		self.sizer3D.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW, 0)#, wx.LEFT | wx.TOP | wx.GROW)
		self.sizer3D.Add(self.slider_curve_position, 1, wx.LEFT | wx.TOP | wx.GROW, 0)
		
		
		self.sizergraphs.Add(self.sizer3D, 2, wx.LEFT | wx.TOP | wx.GROW)
		self.sizerfreq = wx.BoxSizer(wx.VERTICAL)
		self.sizerfreq.Add(self.toolbar2, 1, wx.LEFT | wx.TOP, 0)#, wx.LEFT | wx.TOP | wx.GROW)
		self.sizerfreq.Add(self.canvas2, 1, wx.LEFT | wx.TOP | wx.GROW, 0)
		self.sizerfreq.Add(self.slider_hel_position, 2, wx.ALL|wx.EXPAND, 0)
		self.sizergraphs.Add(self.sizerfreq, 2, wx.LEFT | wx.TOP | wx.GROW)

		self.sizer = wx.BoxSizer(wx.VERTICAL)
		#self.sizer.Add(self.sizerb, 2, wx.LEFT | wx.TOP | wx.GROW)
		self.sizer.Add(self.sizergraphs, 3, wx.LEFT | wx.TOP | wx.GROW)
		self.SetSizer(self.sizer)
		self.Fit()

		#self.OnCropTop(None)
		#self.OnReload(None)
		
	def draw(self):
		self.axes1 = self.figure.add_subplot(111, projection='3d')
		self.axes2 = self.figure2.add_subplot(311)
		self.axes22 = self.figure2.add_subplot(312)
		self.axes3 = self.figure2.add_subplot(313)
		CC = self.parent.C

	
		IN = self.parent.C.IN#RES1
		RES = self.parent.C.RES
		
	
		self.figure.set_canvas(self.canvas)
		self.figure2.set_canvas(self.canvas2)
		#self.axes1.clear()
		self.axes3.clear()
		self.axes2.clear()	
		self.axes22.clear()	
		
		#Axes3D.mouse_init()
		
		# Get the modelled frequency data
		X2 = np.concatenate((-RES[0][::-1], RES[0]))
		Y2= CC.RBF_sim(CC.RBF_param, X2)		
		# Get the modelled helix
		
		c = np.real(CC.RBF_param[2])
		m = CC.RBF_param[4]
		c0 = RES[2]
		t = np.arange(0,1.,1./len(IN[0]))
		
		angle = 1#


		maxl = len(RES[1])-1

		angle = np.angle(RES[1])[int(min(maxl,np.round(c/np.real(RES[0][0]))))]
		angle = np.nan_to_num(angle) 
		c = np.sqrt(np.abs(c))
		
		Xtraj,Ytraj,Ztraj = c*np.real(self.parent.C.spirl(m,t,phi = angle)) + c0*np.real(self.parent.C.spirl(0,t))
		self.axes1.plot (IN[0]/CC.z_bounds[1],IN[1]/CC.z_bounds[1],IN[2]/CC.z_bounds[1], 'k', linewidth=2.0)
		self.axes1. plot (Xtraj/CC.z_bounds[1],Ytraj/CC.z_bounds[1],Ztraj, 'r', linewidth=0.5)
		CURV, POS = CC.get_curvature()
		self.axes1. plot (IN[0][POS]/CC.z_bounds[1],IN[1][POS]/CC.z_bounds[1],IN[2][POS]/CC.z_bounds[1], 'go')
		
		
		#self.axes1.plot (CC.bckg_x/CC.z_bounds[1],CC.bckg_y/CC.z_bounds[1],CC.bckg_z/CC.z_bounds[1], 'g', linewidth=0.5)

		self.axes1.set_title('Normalised trajectory \n scale: 1=' + '% 6.2f' % CC.z_bounds[1] + 'mm')
		self.axes1.set_xlabel('X' )
		self.axes1.set_ylabel('Y' )
		self.axes1.set_zlabel('Z' )
		
		self.axes2.plot(IN[2]/self.parent.C.z_bounds[1],IN[1], 'k', linewidth=2.0)
		self.axes2. plot (IN[2][POS]/CC.z_bounds[1],IN[1][POS], 'go')
		self.axes2.plot(Ztraj,Ytraj,'r',linestyle='--', linewidth=0.5)
		self.axes2.set_xlabel('Normalized depth [0-1])')
		self.axes2.set_ylabel('Normalized X')	
		
		self.axes22.plot(IN[2]/self.parent.C.z_bounds[1],IN[0], 'k', linewidth=2.0)
		self.axes22. plot (IN[2][POS]/CC.z_bounds[1],IN[0][POS], 'go')
		self.axes22.plot(Ztraj,Xtraj,'r',linestyle='--', linewidth=0.5)

		self.axes22.set_xlabel('Normalized depth [0-1]')
		self.axes22.set_ylabel('Normalized Y')	
		
		self.axes3.plot(RES[0],  np.real(RES[1]*np.conj(RES[1])), linewidth=2.0) #RES[1]*np.conj(RES[1]) # np.real(RES[1])
		self.axes3.plot(-RES[0], np.real(RES[3]*np.conj(RES[3])), linewidth=2.0) #RES[3]*np.conj(RES[3]) # np.real(RES[3]
		self.axes3.plot(X2,Y2,'r',linestyle='--')
		self.axes3.plot([m,m],[0,1.1*c*c],'r',linestyle='--')

		self.axes3.text(-self.parent.C.freq_bounds[1]*0.95, 0.15*c**2, 'anti-clockwise', style='italic',bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
		self.axes3.text(self.parent.C.freq_bounds[1]*0.65, 0.15*c**2, 'clockwise', style='italic', bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})		
		self.axes3.set_xlim([-self.parent.C.freq_bounds[1],self.parent.C.freq_bounds[1]])
		self.axes3.set_xlabel('Helix Frequency (mm-1)')
		self.axes3.set_ylabel('Power (mm^2)')
		max_range = 1.
		sxy = 2.

		self.axes1.invert_zaxis()
		X,Y,Z = IN/self.parent.C.z_bounds[1]
		max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
		mean_x = X.mean()
		mean_y = Y.mean()
		mean_z = Z.mean()
		self.axes1.set_xlim(mean_x - max_range/2./sxy, mean_x + max_range/2./sxy)
		self.axes1.set_ylim(mean_y - max_range/2./sxy, mean_y + max_range/2./sxy)
		self.axes1.set_zlim(0 , 1 )
		self.figure2.tight_layout(pad=0.4, w_pad=0.0, h_pad=1.0)
		#self.axes1.set_aspect('equal')

		
		self.canvas.draw()
		self.canvas2.draw()
		self.canvas.draw()
		self.canvas2.draw()
		
	def OnReload(self,event):
		self.parent.C.reload()
		self.IN = self.parent.C.IN
		self.RES = self.parent.C.RES
		self.draw()
	def Onslider(self,event):
		self.sldpos = self.slider_hel_position.GetValue()
		#print float(self.sldpos)/float(self.sldscale)
		self.parent.C.RBF_fit(mini = float(self.sldpos)/float(self.sldscale))
		self.draw()
	def OnCropTop(self, event):
		# Crop top
		#print "MERDE", len(self.parent.C.IN[0]), " / ", len(self.parent.C.IN[0])
		if len(self.parent.C.IN[0]) > 10:
			IN0 = np.delete(self.parent.C.IN[0], 0,0)
			self.parent.C.IN[0] = IN0
			IN1 = np.delete(self.parent.C.IN[1], 0,0)
			self.parent.C.IN[1] = IN1
			IN2 = np.delete(self.parent.C.IN[2], 0,0)
			#print "coucou2: ", self.RES1[2][-1]
			#IN2 = IN2 - IN2.min()
			self.parent.C.IN[2] = IN2
			#print "coucou3: ", self.RES1[2][-1]
			self.parent.C.run()
			#print "coucou4: ", self.RES1[2][-1]
			self.RES = self.parent.C.RES
			self.draw()
			
	def OnCropBottom(self, event):
		# Crop top
		if len(self.parent.C.IN[0]) > 10:
			IN0 = np.delete(self.parent.C.IN[0], -1,0)
			self.parent.C.IN[0] = IN0
			IN1 = np.delete(self.parent.C.IN[1], -1,0)
			self.parent.C.IN[1] = IN1
			IN2 = np.delete(self.parent.C.IN[2], -1,0)
			#IN2 = IN2 - IN2.min()
			self.parent.C.IN[2] = IN2
			self.parent.C.run()
			self.RES = self.parent.C.RES
			self.draw()
			
	def Onsave(self, event):				
		# Save button
		if True: #event.GetId() == self.b4.GetId():
			dlg = wx.FileDialog(self, message="Save file as ...", 
				defaultFile="", style=wx.SAVE)
			if dlg.ShowModal() == wx.ID_OK:
				paths = dlg.GetPath()
				self.parent.C.export_res(self.parent.C.path)
#   9     def __init__(self, parent, id, title):
#  10 
#  11         wx.Dialog.__init__(self, parent, id, title, size=(300, 300))			
class MySettingFrame(wx.Dialog):
	def __init__(self, parent, id, title = 'scaling', size = (200,200), p = ""):
		wx.Dialog.__init__(self, None, id, title, size)	
		
		self.parent = parent
		# INPUT DATA SCALING
		self.txt1 = wx.StaticText(self, -1, "Image Scale (mm/pix)", (50,50))
		buf, file_ext = os.path.splitext(p)
		if file_ext == ".xml":
			self.control1 = wx.TextCtrl(self, -1, "0.05", (50,50))
		else:
			self.control1 = wx.TextCtrl(self, -1, "1.0", (50,50))
		self.txt2 = wx.StaticText(self, -1, "Growth rate (cm/day)", (50,50))
		self.control2 = wx.TextCtrl(self,-1, "1", (50,50))	
		self.txt3 = wx.StaticText(self, -1, "Max Frequency (mm-1)", (50,50))
		self.control3 = wx.TextCtrl(self,-1, str(parent.C.max_freq), (50,50))	
		self.b1 = wx.Button(self, 1, "OK", (50, 50))
		self.Bind(wx.EVT_BUTTON, self.OnOK, self.b1)

		
		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.sizer.Add(self.txt1, 1, wx.ALL|wx.EXPAND )
		self.sizer.Add(self.control1, 1, wx.ALL|wx.EXPAND)
		self.sizer.Add(self.txt2, 1, wx.ALL|wx.EXPAND )
		self.sizer.Add(self.control2, 1, wx.ALL|wx.EXPAND )
		self.sizer.Add(self.txt3, 1, wx.ALL|wx.EXPAND )
		self.sizer.Add(self.control3, 1, wx.ALL|wx.EXPAND )
		self.sizer.Add(wx.StaticText(self, -1, "", (50,50)), 1, wx.ALL|wx.EXPAND )
		self.sizer.Add(wx.StaticText(self, -1, "", (50,50)), 1, wx.ALL|wx.EXPAND )
		self.sizer.Add(wx.StaticText(self, -1, "", (50,50)), 1, wx.ALL|wx.EXPAND )		
		
		
		self.sizer.Add(self.b1, 1, wx.ALL|wx.EXPAND )

		self.SetSizer(self.sizer)
		self.Fit()

	def OnOK(self, event):				
		# Save button
		self.parent.C.scaling = [float(self.control1.GetValue()), float(self.control2.GetValue())]	
		self.parent.C.max_freq = float(self.control3.GetValue())
		#self.parent.panel.OnReload(None)
		
		self.Close()
		
class MyFrame(wx.Frame):
		def __init__(self, id,  title = "", size = (200,200)):
			wx.Frame.__init__(self, None, id, title, size=size)			
			self.C = helical_transform()		
			
			# Panel for drawing
			self.panel = CanvasPanel(self)
			
			# Prepare the menu bar
			menuBar = wx.MenuBar()
		
			# 1st menu from left
			menu1 = wx.Menu()
			menu1.Append(101, "&Open File (.txt or xml)")
			self.Bind(wx.EVT_MENU, self.OpenFile, id=101)			
			#menuBar.Append(menu1, "&File")
			
			menu1.Append(301, "&Export Data (.csv)")
			self.Bind(wx.EVT_MENU, self.panel.Onsave, id=301)			
			menuBar.Append(menu1, "&File")
			

			menu2 = wx.Menu()
			menu2.Append(201, "&Settings")
			self.Bind(wx.EVT_MENU, self.OnSettings, id=201)			
			menuBar.Append(menu2, "&Settings")
			
			
			self.SetMenuBar(menuBar)
			
		def OnSettings(self, event):
			pass
		def OpenFile(self, event):

			dlg = wx.FileDialog(
						None, message="Open Trajecotry file",
						defaultDir=os.getcwd(), 
						defaultFile="",
						style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
						)

			if dlg.ShowModal() == wx.ID_OK:
				paths = dlg.GetPaths()
				for path in paths:
					self.C = helical_transform()
					sett = MySettingFrame(self,-1, title = 'Scaling Parameters', p = path)
					sett.SetSize((500,500))
					WIS = sett.ShowModal()
					sett.Destroy()
					
					self.C.read(path)
					self.C.run()
					self.panel.draw()
					self.SetTitle('ROOTHIX - ' + os.path.basename(path))

		
app = wx.App(False)

fr = MyFrame(-1, title='ROOTHIX', size = (1200,680))
#, C)
#panel.draw()
fr.Show()
app.MainLoop()