from matplotlib import pyplot as plt
from numpy import array, ndarray
# Generate a confusion matrix with precision, sensibility and accuracy as a matplotlib figure

def plot_confusion_matrix(confusion_matrix, labels=None):
	mat = array(confusion_matrix)
	if mat.shape[0] != mat.shape[1]:
		raise IndexError("Matrix should be square")
	if labels is None:
		labels = generic_label(mat)

	gridspec_kw = {"wspace": 0, "hspace": 0}
	subplot_kw = {"xticks": [], "yticks": []}
	center_text= {"horizontalalignment": "center", "verticalalignment": "center"}
	fig, axs = plt.subplots(nrows=mat.shape[0]+1, ncols=mat.shape[1]+1, sharex=True, sharey=True, gridspec_kw=gridspec_kw, subplot_kw=subplot_kw, figsize=(6, 6))
	for idx in range(mat.shape[0]):
		for jdx in range(mat.shape[1]):
			axs[idx,jdx].text(0.5, 0.6, mat[idx][jdx], **center_text, color="black", fontweight="bold")
			percentage = "{:.02f}%".format(mat[idx][jdx]/mat.sum()*100)
			axs[idx,jdx].text(0.5, 0.4, percentage, **center_text, color="black")
			if idx == jdx:
				axs[idx,jdx].set_facecolor("xkcd:sea green")
			else:
				axs[idx,jdx].set_facecolor("xkcd:salmon")

	#Plot precision
	for idx in range(mat.shape[0]):
		precision = mat[idx][idx]/mat.sum(axis=1)[idx]*100
		positive = "{:.02f}%".format(precision)
		negative = "{:.02f}%".format(100-precision)
		axs[idx, mat.shape[1]].text(0.5, 0.6, positive, **center_text, color="green")
		axs[idx, mat.shape[1]].text(0.5, 0.4, negative, **center_text, color="red")
		axs[idx, mat.shape[1]].set_facecolor("xkcd:grey")

	#Plot sensitivity
	for jdx in range(mat.shape[1]):
		sensitivity = mat[jdx][jdx]/mat.sum(axis=0)[jdx]*100
		positive = "{:.02f}%".format(sensitivity)
		negative = "{:.02f}%".format(100-sensitivity)
		axs[mat.shape[0], jdx].text(0.5, 0.6, positive, **center_text, color="green")
		axs[mat.shape[0], jdx].text(0.5, 0.4, negative, **center_text, color="red")
		axs[mat.shape[0], jdx].set_facecolor("xkcd:grey")


	accuracy = mat.diagonal().sum()/mat.sum()*100
	positive = "{:.2f}%".format(accuracy)
	negative = "{:.2f}%".format(100-accuracy)
	axs[mat.shape[0], mat.shape[1]].text(0.5, 0.6, positive, **center_text, color="darkgreen")
	axs[mat.shape[0], mat.shape[1]].text(0.5, 0.4, negative, **center_text, color="red")
	axs[mat.shape[0], mat.shape[1]].set_facecolor("xkcd:periwinkle blue")

	for idx in range(len(labels)):
		axs[idx, 0].set_ylabel(labels[idx])
		axs[0, idx].set_xlabel(labels[idx])
		axs[0, idx].xaxis.set_label_position("top")
	fig.supylabel("True label")
	fig.suptitle("Predicted label")
	return fig

def generic_label(matrix: ndarray):
	return	[x+1 for x in range(matrix.shape[0])]