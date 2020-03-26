#!/usr/bin/python3

import pandas as pd
import numpy as np

from style import style


names = {"data-driven": "Data-Driven", "rdoc": "RDoC", "dsm": "DSM"}
suffix = {"data-driven": "", "rdoc": "_opsim", "dsm": "_opsim"}


def doc_mean_thres(df):

	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	
	return df_bin


def load_coordinates(path="../data", cerebellum="combo"):
	
	atlas_labels = pd.read_csv("{}/brain/labels_{}.csv".format(path, cerebellum))
	activations = pd.read_csv("{}/brain/coordinates_{}.csv".format(path, cerebellum), index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]]
	
	return activations


def load_lexicon(sources, path="../lexicon", tkn_filter=[]):
	
	lexicon = []
	for source in sources:
		file = "{}/lexicon_{}.txt".format(path, source)
		lexicon += [token.strip() for token in open(file, "r").readlines()]
	
	if len(tkn_filter) > 0:
		lexicon = sorted(list(set(lexicon).intersection(tkn_filter)))
	
	return sorted(lexicon)


def load_doc_term_matrix(version=190325, binarize=True, path="../data"):
	
	dtm = pd.read_csv("{}/text/dtm_{}.csv.gz".format(path, version), compression="gzip", index_col=0)
	
	if binarize:
		dtm = doc_mean_thres(dtm)
	
	return dtm


def load_framework(framework, suffix="", clf="", path="../ontology"):
	
	if path.endswith("/"):
		path = path[:-1]
	
	list_file = "{}/lists/lists_{}{}{}.csv".format(path, framework, suffix, clf)
	lists = pd.read_csv(list_file, index_col=None)
	
	circuit_file = "{}/circuits/circuits_{}{}.csv".format(path, framework, clf)
	circuits = pd.read_csv(circuit_file, index_col=0)
	
	return lists, circuits


def score_lists(lists, dtm_bin, label_var="DOMAIN"):

	from collections import OrderedDict
	
	labels = OrderedDict.fromkeys(lists[label_var])
	list_counts = pd.DataFrame(index=dtm_bin.index, columns=labels)
	
	for label in list_counts.columns:
		tkns = lists.loc[lists[label_var] == label, "TOKEN"]
		list_counts[label] = dtm_bin[tkns].sum(axis=1)
	list_scores = doc_mean_thres(list_counts)
	
	return list_scores


def transparent_background(file_name):
	
	from PIL import Image
	
	img = Image.open(file_name)
	img = img.convert("RGBA")
	data = img.getdata()
	
	newData = []
	for item in data:
		if item[0] == 255 and item[1] == 255 and item[2] == 255:
			newData.append((255, 255, 255, 0))
		else:
			newData.append(item)
	
	img.putdata(newData)
	img.save(file_name, "PNG")


def load_atlas(path="../data", cerebellum="combo"):

	import numpy as np
	from nilearn import image

	cer = "{}/brain/atlases/Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz".format(path)
	cor = "{}/brain/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz".format(path)
	sub = "{}/brain/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz".format(path)

	sub_del_dic = {1:0, 2:0, 3:0, 12:0, 13:0, 14:0}
	sub_lab_dic_L = {4:1, 5:2, 6:3, 7:4, 9:5, 10:6, 11:7, 8:8}
	sub_lab_dic_R = {15:1, 16:2, 17:3, 18:4, 19:5, 20:6, 21:7, 7:8}

	sub_mat_L = image.load_img(sub).get_data()[91:,:,:]
	sub_mat_R = image.load_img(sub).get_data()[:91,:,:]

	for old, new in sub_del_dic.items():
		sub_mat_L[sub_mat_L == old] = new
	for old, new in sub_lab_dic_L.items():
		sub_mat_L[sub_mat_L == old] = new
	sub_mat_L = sub_mat_L + 48
	sub_mat_L[sub_mat_L == 48] = 0

	for old, new in sub_del_dic.items():
		sub_mat_R[sub_mat_R == old] = new
	for old, new in sub_lab_dic_R.items():
		sub_mat_R[sub_mat_R == old] = new
	sub_mat_R = sub_mat_R + 48
	sub_mat_R[sub_mat_R == 48] = 0

	cor_mat_L = image.load_img(cor).get_data()[91:,:,:]
	cor_mat_R = image.load_img(cor).get_data()[:91,:,:]

	mat_L = np.add(sub_mat_L, cor_mat_L)
	mat_L[mat_L > 56] = 0
	mat_R = np.add(sub_mat_R, cor_mat_R)
	mat_R[mat_R > 56] = 0

	if cerebellum == "combo":
		mat_R = mat_R + 59
		mat_R[mat_R > 118] = 0
		mat_R[mat_R < 60] = 0

	elif cerebellum == "seg":
		mat_R = mat_R + 74
		mat_R[mat_R > 148] = 0
		mat_R[mat_R < 75] = 0

	cer_mat_L = image.load_img(cer).get_data()[91:,:,:]
	cer_mat_R = image.load_img(cer).get_data()[:91,:,:]
	
	if cerebellum == "combo":
		cer_mat_L[np.isin(cer_mat_L,[1,3,5,14,17,20,23,26])] = 57
		cer_mat_L[np.isin(cer_mat_L,[8,11])] = 58
		cer_mat_L[np.isin(cer_mat_L,[6,9,12,15,18,21,24,27])] = 59
		cer_mat_R[np.isin(cer_mat_R,[2,4,7,16,19,22,25,28])] = 116
		cer_mat_R[np.isin(cer_mat_R,[10,13])] = 117
		cer_mat_R[np.isin(cer_mat_R,[6,9,12,15,18,21,24,27])] = 118
		
		mat_L = np.add(mat_L, cer_mat_L)
		mat_L[mat_L > 59] = 0
		mat_R = np.add(mat_R, cer_mat_R)
		mat_R[mat_R > 118] = 0

	elif cerebellum == "seg":
		cer_mat_L[cer_mat_L == 1] = 57
		cer_mat_L[cer_mat_L == 3] = 58
		cer_mat_L[cer_mat_L == 5] = 59
		cer_mat_L[cer_mat_L == 6] = 69
		cer_mat_L[cer_mat_L == 8] = 65
		cer_mat_L[cer_mat_L == 9] = 67
		cer_mat_L[cer_mat_L == 11] = 66
		cer_mat_L[cer_mat_L == 12] = 68
		cer_mat_L[cer_mat_L == 14] = 60
		cer_mat_L[cer_mat_L == 15] = 70
		cer_mat_L[cer_mat_L == 17] = 61
		cer_mat_L[cer_mat_L == 18] = 71
		cer_mat_L[cer_mat_L == 20] = 62
		cer_mat_L[cer_mat_L == 21] = 72
		cer_mat_L[cer_mat_L == 23] = 63
		cer_mat_L[cer_mat_L == 24] = 73
		cer_mat_L[cer_mat_L == 26] = 64
		cer_mat_L[cer_mat_L == 27] = 74
		
		cer_mat_R[cer_mat_R == 2] = 131
		cer_mat_R[cer_mat_R == 4] = 132
		cer_mat_R[cer_mat_R == 6] = 143
		cer_mat_R[cer_mat_R == 7] = 133
		cer_mat_R[cer_mat_R == 9] = 141
		cer_mat_R[cer_mat_R == 10] = 139
		cer_mat_R[cer_mat_R == 12] = 142
		cer_mat_R[cer_mat_R == 13] = 140
		cer_mat_R[cer_mat_R == 15] = 144
		cer_mat_R[cer_mat_R == 16] = 134
		cer_mat_R[cer_mat_R == 18] = 145
		cer_mat_R[cer_mat_R == 19] = 135
		cer_mat_R[cer_mat_R == 21] = 146
		cer_mat_R[cer_mat_R == 22] = 136
		cer_mat_R[cer_mat_R == 24] = 147
		cer_mat_R[cer_mat_R == 25] = 137
		cer_mat_R[cer_mat_R == 27] = 148
		cer_mat_R[cer_mat_R == 28] = 138

		mat_L = np.add(mat_L, cer_mat_L)
		mat_L[mat_L > 75] = 0
		mat_R = np.add(mat_R, cer_mat_R)
		mat_R[mat_R > 148] = 0

	mat = np.concatenate((mat_R, mat_L), axis=0)
	atlas_image = image.new_img_like(sub, mat)

	return atlas_image


def map_plane(estimates, atlas, path, suffix="", plane="z", cbar=False, annotate=False,
			  vmin=None, vmax=None, cmaps=[], print_fig=True, verbose=False):
	
	from nilearn import image, plotting
	
	for f, feature in enumerate(estimates.columns):
		stat_map = image.copy_img(atlas).get_data()
		data = estimates[feature]
		if verbose:
			print("{:20s} Min: {:6.4f}  Mean: {:6.4f}  Max: {:6.4f}".format(
				  feature, min(data), np.mean(data), max(data)))
		if not verbose and print_fig:
			print("\n{}".format(feature))
		for i, value in enumerate(data):
			stat_map[stat_map == i+1] = value
		stat_map = image.new_img_like(atlas, stat_map)
		display = plotting.plot_stat_map(stat_map,
										 display_mode=plane, 
										 symmetric_cbar=False, colorbar=cbar,
										 cmap=cmaps[f], threshold=vmin, 
										 vmax=vmax, alpha=0.5,
										 annotate=annotate, draw_cross=False)
		file_name = "{}/{}{}.png".format(path, feature, suffix)
		display.savefig(file_name, dpi=250)
		transparent_background(file_name)
		if print_fig:
			plotting.show()
		display.close()


def compare_to_null(df_null, df, n_iter, alpha=0.001):
	
	from statsmodels.stats.multitest import multipletests
	
	pval = []
	for dom in df.index:
		dom_null = df_null.loc[dom].values
		dom_obs = float(df.loc[dom, "OBSERVED"])
		p = np.sum(dom_null >= dom_obs) / float(n_iter)
		pval.append(p)
		df.loc[dom, "P"] = p
	df["FDR"] = multipletests(pval, method="fdr_bh")[1]
	return df


def compare_bootstraps(stats, frameworks, n_iter=1000):

	from statsmodels.stats.multitest import multipletests

	p = np.empty((len(frameworks), len(frameworks)))
	for i, fw_i in enumerate(frameworks):
	    for j, fw_j in enumerate(frameworks):
	        boot_i = np.mean(stats["boot"][fw_i], axis=0)
	        boot_j = np.mean(stats["boot"][fw_j], axis=0)
	        p[i,j] = np.sum((boot_i - boot_j) <= 0.0) / float(n_iter)
	fdr = multipletests(p.ravel(), method="fdr_bh")[1].reshape(p.shape)
	fdr = pd.DataFrame(fdr, index=frameworks, columns=frameworks)

	return fdr


def load_stat_file(stats, stat_name, metric, stat, framework, suffix="", path=""):
	
	file = "{}data/{}_{}_{}{}.csv".format(path, metric, stat_name, framework, suffix)
	stats[stat_name][framework] = pd.read_csv(file, index_col=0, header=0)
	
	return stats


def load_partition_stats(stats, metric, framework, lists, dom2docs, n_iter=1000, alpha=0.001, clf="", path=""):

	stat_names = ["obs", "mean", "null", "boot"]
	iter_suffixes = ["", "", "_{}iter".format(n_iter), "_{}iter".format(n_iter)]

	for stat_name, iter_suffix in zip(stat_names, iter_suffixes):
		suffix = clf + iter_suffix
		stats = load_stat_file(stats, stat_name, metric, stat_name, framework, suffix=suffix, path=path)

	stats["null_comparison"][framework] = compare_to_null(stats["null"][framework], stats["mean"][framework], n_iter, alpha=alpha)

	return stats


def plot_violins(framework, domains, df_obs, df_null, df_stat, palette, metric="",
				 dx=[], dy=0.06, ds=0.115, interval=0.999, alphas=[0.01, 0.001, 0.0001],
				 ylim=[-0.1, 0.85], yticks=[0, 0.25, 0.5, 0.75], print_fig=True, 
				 font=style.font, path="", suffix=""):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams

	font_prop = font_manager.FontProperties(fname=font, size=20)
	rcParams["axes.linewidth"] = 1.5

	# Set up figure
	fig = plt.figure(figsize=(4.5, 2.1))
	ax = fig.add_axes([0,0,1,1])

	# Violin plot of observed values
	for i, dom in enumerate(domains):
		data = sorted(df_obs.loc[dom].dropna())
		obs = df_stat.loc[dom, "OBSERVED"]
		v = ax.violinplot(data, positions=[i], 
						  showmeans=False, showmedians=False, widths=0.85)
		for pc in v["bodies"]:
			pc.set_facecolor(palette[i])
			pc.set_edgecolor(palette[i])
			pc.set_linewidth(1.25)
			pc.set_alpha(0.4)
		for line in ["cmaxes", "cmins", "cbars"]:
			v[line].set_edgecolor("none")
		plt.plot([i-dx[i], i+dx[i]], [obs, obs], 
					c=palette[i], alpha=1, lw=2)

		# Comparison test
		dys = dy * np.array([0, 1, 2])
		for alpha, y in zip(alphas, dys):
			if df_stat["FDR"][i] < alpha:
				plt.text(i-ds, max(data) + y, "*", fontproperties=font_prop)

	# Confidence interval of null distribution
	n_iter = df_null.shape[1]
	lower = [sorted(df_null.loc[dom])[int(n_iter*(1.0-interval))] for dom in domains]
	upper = [sorted(df_null.loc[dom])[int(n_iter*interval)] for dom in domains]
	plt.fill_between(range(len(domains)), lower, upper, 
					 alpha=0.2, color="gray")
	plt.plot(df_null.values.mean(axis=1), linestyle="dashed", color="gray", linewidth=2)

	# Set plot parameters
	plt.xticks(range(len(domains)), [""]*len(domains))
	plt.yticks(yticks, fontproperties=font_prop)
	plt.xlim([-0.75, len(domains)-0.35])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	# Export figure
	plt.savefig("{}figures/{}_{}{}_{}iter.png".format(
				path, metric, framework, suffix, n_iter), dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()


def plot_framework_comparison(boot, obs, mean, n_iter=1000, print_fig=True, font=style.font, dx=0.38, 
							  ylim=[0.4,0.65], yticks=[], metric="mod", suffix="", path=""):
	
	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	
	font_lg = font_manager.FontProperties(fname=font, size=20)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=(2.1, 2.1))
	ax = fig.add_axes([0,0,1,1])

	i = 0
	labels = []
	for fw, dist in boot.items():
		labels.append(names[fw])
		dist_avg = np.mean(dist, axis=0)
		macro_avg = np.mean(mean[fw]["OBSERVED"])
		plt.plot([i-dx, i+dx], [macro_avg, macro_avg], 
				 c="gray", alpha=1, lw=2, zorder=-1)
		v = ax.violinplot(sorted(dist_avg), positions=[i], 
						  showmeans=False, showmedians=False, widths=0.85)
		for pc in v["bodies"]:
			pc.set_facecolor("gray")
			pc.set_edgecolor("gray")
			pc.set_linewidth(2)
			pc.set_alpha(0.5)
		for line in ["cmaxes", "cmins", "cbars"]:
			v[line].set_edgecolor("none")
		i += 1

	ax.set_xticks(range(len(boot.keys())))
	ax.set_xticklabels([], rotation=60, ha="right")
	plt.xticks(fontproperties=font_lg)
	plt.yticks(yticks, fontproperties=font_lg)
	plt.xlim([-0.75, len(boot.keys())-0.25])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	plt.savefig("{}figures/{}_{}_{}iter.png".format(path, metric, suffix, n_iter), 
				dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()



