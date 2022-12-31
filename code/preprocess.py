from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import utils as utils
import numpy as np
import h5py
import scipy as sp
import pandas as pd
import scanpy.api as sc
from sklearn.metrics.cluster import contingency_matrix


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=utils.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=utils.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def read_real_old(filename, batch = True):
    data_path = "../data/real_data/" + filename + "/data.h5"
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_ontology_class"])
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    if batch == True:
        batch_name = np.array(obs["dataset_name"])
        return X, cell_name, batch_name
    else:
        return X, cell_name


def read_real(filename, batch=True):
    data_path = "../data/real_data/" + filename + "/data.h5"
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_ontology_class"])
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    if batch == True:
        if "dataset_name" in obs.keys():
            batch_name = np.array(obs["dataset_name"])
        else:
            batch_name = np.array(obs["study"])
        return X, cell_name, batch_name
    else:
        return X, cell_name


def read_real_with_genes(filename, batch=True):
    data_path = "../data/real_data/" + filename + "/data.h5"
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_ontology_class"])
    gene_name = np.array(list(var.index))
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    if batch == True:
        if "dataset_name" in obs.keys():
            batch_name = np.array(obs["dataset_name"])
        else:
            batch_name = np.array(obs["study"])
        return X, cell_name, batch_name, gene_name
    else:
        return X, cell_name, gene_name


def class_splitting_new(filename, source_name, target_name):
    seen_classes = []
    novel_classes = []
    if filename == "ALIGNED_Homo_sapiens_Pancreas":
        if source_name == "Enge" and target_name == "Baron_human":
            seen_classes = ['pancreatic A cell', 'pancreatic acinar cell', 'pancreatic ductal cell', 'type B pancreatic cell']
            novel_classes = ['endothelial cell', 'pancreatic D cell', 'pancreatic PP cell', 'pancreatic stellate cell']
        if source_name == "Lawlor" and target_name == "Baron_human":
            seen_classes = ['pancreatic A cell', 'pancreatic D cell', 'pancreatic acinar cell', 'type B pancreatic cell']
            novel_classes = ['endothelial cell', 'pancreatic PP cell', 'pancreatic ductal cell', 'pancreatic stellate cell']
        if source_name == "Muraro" and target_name == "Baron_human":
            seen_classes = ['pancreatic A cell', 'pancreatic acinar cell', 'pancreatic ductal cell', 'type B pancreatic cell']
            novel_classes = ['endothelial cell', 'pancreatic D cell', 'pancreatic PP cell', 'pancreatic stellate cell']
        if source_name == "Xin_2016" and target_name == "Baron_human":
            seen_classes = ['pancreatic A cell', 'pancreatic D cell', 'pancreatic PP cell', 'type B pancreatic cell']
            novel_classes = ['endothelial cell', 'pancreatic acinar cell', 'pancreatic ductal cell', 'pancreatic stellate cell']
    if filename == "ALIGNED_Homo_sapiens_Placenta":
        if source_name == "Vento-Tormo_10x" and target_name == "Vento-Tormo_Smart-seq2":
            seen_classes = ['T cell', 'decidual natural killer cell', 'macrophage', 'monocyte']
            novel_classes = ['natural killer cell', 'placental villous trophoblast', 'stromal cell', 'trophoblast cell']
        if source_name == "Vento-Tormo_Smart-seq2" and target_name == "Vento-Tormo_10x":
            seen_classes = ['T cell', 'decidual natural killer cell', 'macrophage', 'monocyte']
            novel_classes = ['natural killer cell', 'placental villous trophoblast', 'stromal cell', 'trophoblast cell']
    if filename == "ALIGNED_Mus_musculus_Bladder":
        if source_name == "Quake_Smart-seq2_Bladder" and target_name == "Quake_10x_Bladder":
            seen_classes = ['bladder cell', 'bladder urothelial cell']
            novel_classes = ['endothelial cell', 'leukocyte']
    if filename == "ALIGNED_Mus_musculus_Limb_Muscle":
        if source_name == "Quake_10x_Limb_Muscle" and target_name == "Quake_Smart-seq2_Limb_Muscle":
            seen_classes = ['B cell', 'T cell', 'endothelial cell']
            novel_classes = ['macrophage', 'mesenchymal stem cell', 'skeletal muscle satellite cell']
        if source_name == "Quake_Smart-seq2_Limb_Muscle" and target_name == "Quake_10x_Limb_Muscle":
            seen_classes = ['B cell', 'T cell', 'endothelial cell']
            novel_classes = ['macrophage', 'mesenchymal stem cell', 'skeletal muscle satellite cell']
    if filename == "ALIGNED_Mus_musculus_Mammary_Gland":
        if source_name == "Quake_Smart-seq2_Mammary_Gland" and target_name == "Quake_10x_Mammary_Gland":
            seen_classes = ['basal cell', 'endothelial cell', 'luminal epithelial cell of mammary gland', 'stromal cell']
            novel_classes = ['B cell', 'T cell', 'macrophage']
    if filename == "ALIGNED_Mus_musculus_Retina":
        if source_name == "Shekhar" and target_name == "Macosko":
            seen_classes = ['Muller cell', 'amacrine cell', 'retinal bipolar neuron', 'retinal rod cell']
            novel_classes = ['blood vessel endothelial cell', 'retina horizontal cell', 'retinal cone cell', 'retinal ganglion cell']
    if filename == "ALIGNED_Mus_musculus_Small_Intestine":
        if source_name == "Haber_10x" and target_name == "Haber_Smart-seq2":
            seen_classes = ['brush cell', 'enterocyte of epithelium of small intestine']
            novel_classes = ['small intestine goblet cell', 'stem cell']
        if source_name == "Haber_Smart-seq2" and target_name == "Haber_10x":
            seen_classes = ['brush cell', 'enterocyte of epithelium of small intestine', 'enteroendocrine cell']
            novel_classes = ['paneth cell', 'small intestine goblet cell', 'stem cell']
        if source_name == "Haber_10x_largecell" and target_name == "Haber_10x_region":
            seen_classes = ['brush cell', 'enterocyte of epithelium of small intestine', 'enteroendocrine cell']
            novel_classes = ['paneth cell', 'small intestine goblet cell', 'stem cell']
        if source_name == "Haber_10x_region" and target_name == "Haber_10x_largecell":
            seen_classes = ['brush cell', 'enterocyte of epithelium of small intestine', 'enteroendocrine cell']
            novel_classes = ['paneth cell', 'small intestine goblet cell', 'stem cell']
    if filename == "ALIGNED_Mus_musculus_Spleen":
        if source_name == "Quake_Smart-seq2_Spleen" and target_name == "Quake_10x_Spleen":
            seen_classes = ['B cell', 'T cell']
            novel_classes = ['macrophage', 'natural killer cell']
    if filename == "ALIGNED_Mus_musculus_Trachea":
        if source_name == "Plasschaert" and target_name == "Montoro_10x":
            seen_classes = ['basal cell of epithelium of trachea', 'club cell']
            novel_classes = ['brush cell of trachea', 'ciliated columnar cell of tracheobronchial tree']
    return seen_classes, novel_classes


def class_splitting_single(dataname):
    class_set = []
    if dataname == "Quake_10x":
        class_set = ['B cell', 'T cell', 'alveolar macrophage', 'basal cell', 'basal cell of epidermis', 'bladder cell',
                     'bladder urothelial cell', 'blood cell', 'endothelial cell', 'epithelial cell', 'fibroblast',
                     'granulocyte', 'granulocytopoietic cell', 'hematopoietic precursor cell', 'hepatocyte',
                     'immature T cell', 'keratinocyte', 'kidney capillary endothelial cell', 'kidney collecting duct epithelial cell',
                     'kidney loop of Henle ascending limb epithelial cell', 'kidney proximal straight tubule epithelial cell',
                     'late pro-B cell', 'leukocyte', 'luminal epithelial cell of mammary gland', 'lung endothelial cell',
                     'macrophage', 'mesenchymal cell', 'mesenchymal stem cell', 'monocyte', 'natural killer cell',
                     'neuroendocrine cell', 'non-classical monocyte', 'proerythroblast', 'promonocyte', 'skeletal muscle satellite cell',
                     'stromal cell']
    if dataname == "Quake_Smart-seq2":
        class_set = ['B cell', 'Slamf1-negative multipotent progenitor cell', 'T cell', 'astrocyte of the cerebral cortex',
                     'basal cell', 'basal cell of epidermis', 'bladder cell', 'bladder urothelial cell', 'blood cell',
                     'endothelial cell', 'enterocyte of epithelium of large intestine', 'epidermal cell', 'epithelial cell',
                     'epithelial cell of large intestine', 'epithelial cell of proximal tubule', 'fibroblast', 'granulocyte',
                     'hematopoietic precursor cell', 'hepatocyte', 'immature B cell', 'immature T cell', 'keratinocyte',
                     'keratinocyte stem cell', 'large intestine goblet cell', 'late pro-B cell', 'leukocyte',
                     'luminal epithelial cell of mammary gland', 'lung endothelial cell', 'macrophage', 'mesenchymal cell',
                     'mesenchymal stem cell', 'mesenchymal stem cell of adipose', 'microglial cell', 'monocyte', 'myeloid cell',
                     'naive B cell', 'neuron', 'oligodendrocyte', 'oligodendrocyte precursor cell', 'pancreatic A cell',
                     'pro-B cell', 'skeletal muscle satellite cell', 'skeletal muscle satellite stem cell', 'stromal cell',
                     'type B pancreatic cell']
    if dataname == "Cao":
        class_set = ['GABAergic neuron', 'cholinergic neuron', 'ciliated olfactory receptor neuron', 'coelomocyte', 'epidermal cell',
                     'germ line cell', 'glial cell', 'interneuron', 'muscle cell', 'nasopharyngeal epithelial cell', 'neuron',
                     'seam cell', 'sensory neuron', 'sheath cell', 'socket cell (sensu Nematoda)', 'visceral muscle cell']
    if dataname == "Hochane":
        class_set = ['endothelial cell', 'epithelial cell of nephron', 'epithelial cell of proximal tubule',
                     'glomerular visceral epithelial cell', 'mesenchymal cell', 'stromal cell']
    if dataname == "Lake_2018":
        class_set = ['Purkinje cell', 'astrocyte', 'cerebellar granule cell', 'microglial cell', 'neuron',
                     'oligodendrocyte', 'oligodendrocyte precursor cell']
    if dataname == "Park":
        class_set = ['T cell', 'endothelial cell', 'epithelial cell of proximal tubule', 'fibroblast', 'kidney distal convoluted tubule epithelial cell',
                     'kidney loop of Henle epithelial cell', 'natural killer cell', 'renal intercalated cell', 'renal principal cell']
    if dataname == "Tosches_turtle":
        class_set = ['GABAergic inhibitory interneuron', 'ependymoglial cell', 'glutamatergic neuron', 'microglial cell',
                     'oligodendrocyte precursor cell']
    if dataname == "Wagner":
        class_set = ['early embryonic cell', 'ectodermal cell', 'embryonic cell', 'endodermal cell', 'epiblast cell', 'epidermal cell',
                     'erythroid progenitor cell', 'lateral mesodermal cell', 'mesodermal cell', 'midbrain dopaminergic neuron',
                     'neural crest cell', 'neurecto-epithelial cell', 'neuronal stem cell', 'spinal cord interneuron']
    if dataname == "Zeisel_2018":
        class_set = ['CNS neuron (sensu Vertebrata)', 'astrocyte', 'cerebellum neuron', 'dentate gyrus of hippocampal formation granule cell',
                     'endothelial cell of vascular tree', 'enteric neuron', 'ependymal cell', 'glial cell', 'inhibitory interneuron',
                     'microglial cell', 'neuroblast', 'oligodendrocyte', 'peptidergic neuron', 'pericyte cell', 'peripheral sensory neuron',
                     'perivascular macrophage', 'vascular associated smooth muscle cell']
    if dataname == "Zheng":
        class_set = ['B cell', 'T-helper 2 cell', 'cytotoxic T cell', 'hematopoietic precursor cell', 'memory T cell', 'monocyte',
                     'naive thymus-derived CD4-positive, alpha-beta T cell', 'naive thymus-derived CD8-positive, alpha-beta T cell',
                     'natural killer cell', 'regulatory T cell']
    if dataname == "Chen":
        class_set = ['GABAergic neuron', 'astrocyte', 'endothelial cell', 'ependymal cell', 'glutamatergic neuron', 'macrophage',
                     'microglial cell', 'oligodendrocyte', 'oligodendrocyte precursor cell', 'tanycyte']
    if dataname == "Guo":
        class_set = ['Leydig cell', 'endothelial cell', 'macrophage', 'male germ line stem cell', 'primary spermatocyte', 'smooth muscle cell',
                     'sperm', 'spermatid', 'spermatogonium']
    if dataname == "Tosches_lizard":
        class_set = ['GABAergic inhibitory interneuron', 'ependymoglial cell', 'glutamatergic neuron', 'microglial cell',
                     'oligodendrocyte', 'oligodendrocyte precursor cell']
    if dataname == "Young":
        class_set = ['vasa recta descending limb cell', 'vasa recta ascending limb cell', 'ureter urothelial cell', 'kidney loop of Henle epithelial cell',
                     'kidney collecting duct epithelial cell', 'glomerular visceral epithelial cell', 'glomerular endothelial cell',
                     'epithelial cell of proximal tubule']
    return class_set


def read_simu(data_path, cross=True):
    data_mat = h5py.File(data_path)
    x = np.array(data_mat["X"])
    y = np.array(data_mat["Y"])
    if cross:
        batch = np.array(data_mat["B"])
        return x, y, batch
    else:
        return x, y


def normalize(adata, highly_genes = None, size_factors=True, normalize_input=True, logtrans_input=True):
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def annotation(cellname_train, cellname_test, Y_pred_train, Y_pred_test):
    train_confusion_matrix = contingency_matrix(cellname_train, Y_pred_train)
    annotated_cluster = np.unique(Y_pred_train)[train_confusion_matrix.argmax(axis=1)]
    annotated_celltype = np.unique(cellname_train)
    annotated_score = np.max(train_confusion_matrix, axis=1) / np.sum(train_confusion_matrix, axis=1)
    annotated_celltype[(np.max(train_confusion_matrix, axis=1) / np.sum(train_confusion_matrix, axis=1)) < 0.5] = "unassigned"
    final_annotated_cluster = []
    final_annotated_celltype = []
    for i in np.unique(annotated_cluster):
        candidate_celltype = annotated_celltype[annotated_cluster == i]
        candidate_score = annotated_score[annotated_cluster == i]
        final_annotated_cluster.append(i)
        final_annotated_celltype.append(candidate_celltype[np.argmax(candidate_score)])
    annotated_cluster = np.array(final_annotated_cluster)
    annotated_celltype = np.array(final_annotated_celltype)

    succeed_annotated_train = 0
    succeed_annotated_test = 0
    test_annotation_label = np.array(["original versions for unassigned cell ontology types"] * len(cellname_test))
    for i in range(len(annotated_cluster)):
        succeed_annotated_train += (cellname_train[Y_pred_train == annotated_cluster[i]] == annotated_celltype[i]).sum()
        succeed_annotated_test += (cellname_test[Y_pred_test == annotated_cluster[i]] == annotated_celltype[i]).sum()
        test_annotation_label[Y_pred_test == annotated_cluster[i]] = annotated_celltype[i]
    annotated_train_accuracy = np.around(succeed_annotated_train / len(cellname_train), 4)
    total_overlop_test = 0
    for celltype in np.unique(cellname_train):
        total_overlop_test += (cellname_test == celltype).sum()
    annotated_test_accuracy = np.around(succeed_annotated_test / total_overlop_test, 4)
    test_annotation_label[test_annotation_label == "original versions for unassigned cell ontology types"] = "unassigned"
    return annotated_train_accuracy, annotated_test_accuracy, test_annotation_label


