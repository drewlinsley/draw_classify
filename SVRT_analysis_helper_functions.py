def scale_range(data,NewMax,NewMin):
	import numpy as np
	OldMax = np.max(data)
	OldMin = np.min(data)
	OldRange = (OldMax - OldMin)  
	NewRange = (NewMax - NewMin)  
	NewValue = (((data - OldMin) * NewRange) / OldRange) + NewMin
	return NewValue

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi 

def conv_coors(x,y):
	r = np.zeros(x.shape)
	p = np.zeros(x.shape)
	for row in range(0,x.shape[0]):
		for col in range(0,x.shape[1]):
			r[row,col], p[row,col] = cart2pol(x[row,col],y[row,col])
	return r, p

def img_grid(arr, rows, cols, global_scale=True):
    N, channels, height, width = arr.shape

    total_height = rows * height + 9
    total_width  = cols * width + 19

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((channels, total_height, total_width))
    I.fill(1)

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr[i])

        offset_y, offset_x = r*height+r, c*width+c
        I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = this
    
    I = (255*I).astype(np.uint8)
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I).astype(np.uint8)
    return Image.fromarray(out)

def atten_features(N_iter,batch_size,image_size,images,fun):
	x_arr = np.zeros((N_iter,batch_size))
	y_arr = np.zeros((N_iter,batch_size))
	delta = np.zeros((N_iter,batch_size))
	h_enc = np.zeros((N_iter,batch_size))
	for idx in range(0,batch_size):
		te,ty,tx,td = fun(images[idx].reshape(1,np.prod(image_size)))
		y_arr[:,idx] = ty.ravel()
		x_arr[:,idx] = tx.ravel()
		delta[:,idx] = td.ravel()
		h_enc[:,idx] = te[:,:,-1].ravel()

	return h_enc, y_arr, x_arr, delta

def hog_features(im,orientations,ppc):
	import numpy as np
	x_arr = np.zeros((im.shape[0],orientations**2*np.prod(ppc)))
	for idx in range(0,im.shape[0]):
		fd, hog_image = hog(train_image[idx,:,:], orientations=orientations, pixels_per_cell=(ppc),
		                    cells_per_block=(1, 1), visualise=True)
		x_arr[idx,:] = fd
	return x_arr

def svm_trainer(train_data,test_data,train_labels,test_labels,num_estimates,num_training,num_testing):
	import numpy as np
	from sklearn import svm
	pos_neg_samps = num_training / 2
	train_labels_pos = np.array([i for i, elem in enumerate(train_labels == 1, 1) if elem]) - 1
	train_labels_neg = np.array([i for i, elem in enumerate(train_labels == 0, 1) if elem]) - 1

	tr_idx = np.array(range(0,int(num_estimates*pos_neg_samps))).reshape(num_estimates,int(pos_neg_samps))
	#te_idx = np.repeat(range(0,num_estimates),num_testing,axis=0)
	acc = [];
	for idx in range(0,num_estimates):
		clf = svm.SVC()
		it_train = np.vstack((train_data[train_labels_pos[tr_idx[idx,:]],:],
			train_data[train_labels_neg[tr_idx[idx,:]],:]))
		it_lab = np.hstack((np.repeat(1,pos_neg_samps),np.repeat(0,pos_neg_samps)))
		clf.fit(it_train,it_lab)
		y_hat = clf.predict(test_data[idx,:].reshape(1, -1))
		it_acc = np.sum(test_labels[idx]==y_hat) / num_testing
		#Store stuff
		acc = np.hstack((acc,it_acc))
	return acc

def plot_ims(test_image_res,tcenter_x,tcenter_y,tdelta,which_ims,out_dir):
	from matplotlib.patches import Rectangle, Circle
	colors = mpl.cm.Reds((range(np.sum(which_ims))))
	a_range = np.linspace(.2,.8,np.sum(which_ims))
	for num in range(0,which_ims.shape[0]):
		if which_ims[num] == True:
			x=tcenter_x[:,num]
			y=tcenter_y[:,num]
			d=tdelta[:,num]
			plt.figure()
			plt.axis([0,32,0,32])
			currentAxis = plt.gca()
			it_im = test_image_res[num,:].reshape(image_size)
			currentAxis.imshow(it_im,cmap=plt.cm.binary)
			for idx in range(0,x.shape[0]):

				adj_delta = (d[idx]/2)
				d_hyp = np.sqrt(adj_delta**2 + adj_delta**2)

				lcx = np.min(((x[idx] - adj_delta),0))
				lcy = np.min(((y[idx] - adj_delta),0))
				rcx = np.min(((x[idx] + adj_delta),0))
				rcy = np.min(((y[idx] + adj_delta),0))
				#currentAxis.add_patch(Rectangle(((lcx,lcy)), adj_delta, adj_delta, ec='red',facecolor='none'))
				currentAxis.add_patch(Circle(((x[idx],y[idx])), adj_delta, ec='red',facecolor='none',alpha=a_range[idx]))


			#plt.imshow(it_im,cmap=plt.cm.binary)
			#c = np.linspace(1,lwidths.shape[0],lwidths.shape[0])
			#plt.scatter(x,y, s=lwidths, c=c, cmap=mpl.cm.Reds, alpha=0.5)
			#plt.show()
			#xnew = np.linspace(T.min(),T.max(),300)
			#power_smooth = spline(T,power,xnew)
			#plt.plot(x[:,num],y[:,num])

			fig_name = os.path.join(subdir,"{}_{}.png".format("image", num))
			plt.savefig(fig_name)
			plt.cla()
			plt.clf()
