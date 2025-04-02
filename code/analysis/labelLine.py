import numpy as np
# from matplotlib.dates import date2num
# from datetime import datetime
import math
from math import atan2


OFFDEF=0.1
ALDEF=True
#Label line with line2D label data
def labelLine(line, x, label, ang=None, align=ALDEF, y_offset=OFFDEF, **kwargs):

	ax = line.axes
	xdata = line.get_xdata()
	ydata = line.get_ydata()
	order=np.argsort(xdata)
	xdata=xdata[order]
	ydata=ydata[order]
	# print(xdata[0],xdata[-1])
	if (x < xdata[0]) or (x > xdata[-1]):
		print('x label location is outside data range!')
		return

	axis_to_data = ax.transAxes + ax.transData.inverted()
	data_to_axis = axis_to_data.inverted()
	##Transforming to axis coordinates
	ax_dat=data_to_axis.transform(np.transpose([xdata, ydata]))
	#Find corresponding y co-ordinate and angle of the
	ip = 1
	for i in range(len(xdata)):
		if x < xdata[i]:
			ip = i
			break
	y = (ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1]))
	y = axis_to_data.transform([0, data_to_axis.transform([0,y])[1]+y_offset])[1]

	if not label:
		label = line.get_label()

	if align and not ang:
		#Compute the slope
		# sp1 = ax.transData.transform((xdata[ip], ydata[ip]))
		# sp2 = ax.transData.transform((xdata[ip-1], ydata[ip-1]))
		# ang = math.degrees(atan2(sp1[1]-sp2[1],sp1[0]-sp2[0]))
		# #Transform to screen co-ordinates
		# #pt = np.array([x,y]).reshape((1,2))
		# trans_angle = ang
		# print(f"Transformed Points: sp1={sp1}, sp2={sp2}")
		# print(f"Ang: {ang}")
		# Compute the slope
		dx = xdata[ip] - xdata[ip - 1]
		dy = ydata[ip] - ydata[ip - 1]
		ang = math.degrees(atan2(dy, dx))

		# Transform to screen co-ordinates
		pt = np.array([x, y]).reshape((1, 2))
		trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[0]

	elif align:
		#Transform to screen co-ordinates
		pt = np.array([x,y]).reshape((1,2))
		trans_angle = ang
	else:
		trans_angle = 0
	#print trans_angle

	#Set a bunch of keyword arguments
	if 'color' not in kwargs:
		kwargs['color'] = line.get_color()

	if 'alpha' not in kwargs:
		kwargs['alpha'] = line.get_alpha()

	if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
		kwargs['ha'] = 'left'

	if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
		kwargs['va'] = 'center'

	if 'backgroundcolor' not in kwargs:
		kwargs['backgroundcolor'] = ax.get_facecolor()

	if 'clip_on' not in kwargs:
		kwargs['clip_on'] = False

	# if 'zorder' not in kwargs:
	# 	kwargs['zorder'] = 2.5

	t=ax.text(x,y,label,rotation=trans_angle, rotation_mode='anchor',**kwargs)
	#t=ax.text(x,y*y_offset,label,rotation=trans_angle, **kwargs)
	t.set_bbox(dict(facecolor='white', alpha=0.1, edgecolor='white'))


def labelLines(lines, xvals=None, y_offset=OFFDEF, align=ALDEF, **kwargs):
	'''
	labelLines--Labels all of the lines in matplotlib plot


	Lines:list 


	xvals (Array-like or None) None--x-axis locations of line labels. If this is None (default), the x-positions 
	are evenly spaced. The number of elements in xvals must equal the number of lines. 
	y_offset (Float) 0.05 -- Offset between each label and line in axis coordinates (which go from 0 to 1 for both x and y)
	align (Boolean) True -- Whether to rotate label to be aligned with the line
	'''

	ax = lines[0].axes
	labLines = []
	labels = []
	axis_to_data = ax.transAxes + ax.transData.inverted()


	#Take only the lines which have labels other than the default ones
	for line in lines:
		label = line.get_label()
		if "_line" not in label:
			labLines.append(line)
			labels.append(label)

	##By default line-labels are evenly spaced in x.
	if xvals is None:
		xvals = np.linspace(0, 1,len(labLines)+2)[1:-1]
		dat = axis_to_data.transform([[xx,0] for xx in xvals])
		xvals = dat[:,0]


	y_offset=np.atleast_1d(y_offset)
	tmp=0
	for line,x,label in zip(labLines,xvals,labels):
		labelLine(line,x,label=label,align=align,y_offset=y_offset[tmp%len(y_offset)],**kwargs)
		tmp+=1
