import nichepca as npc
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def annotate(
    ad_target,
    ad_ref,
    label_key,
    label_out=None,
    min_counts=8,
    target_sum=1e3,
    genes=None,
    scaler="minmax",
    **kwargs,
):
    if genes is None:
        genes = ad_ref.var_names.intersection(ad_target.var_names).tolist()

    if label_out is None:
        label_out = label_key

    ad_train = ad_ref[:, genes].copy()
    ad_test = ad_target[:, genes].copy()

    # filter cells
    sc.pp.filter_cells(ad_train, min_counts=min_counts)
    sc.pp.filter_cells(ad_test, min_counts=min_counts)

    clf = LogisticRegression(random_state=42, max_iter=1000, **kwargs)

    encoder = LabelEncoder()

    pp = PreProcessor(scaler=scaler)

    encoder.fit(ad_train.obs[label_key])
    ad_train = pp.fit_transform(ad_train, target_sum=target_sum)

    # train data
    X_train = ad_train.X
    y_train = encoder.transform(ad_train.obs[label_key])

    # test data
    ad_test = pp.transform(ad_test)
    X_test = ad_test.X

    print("Fitting the model ...")
    clf.fit(X_train, y_train)
    print("Model fitted.")

    # inference
    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    labels_pred = encoder.inverse_transform(y_pred)

    # store predictions
    ad_target.obs[label_out] = "unknown"
    ad_target.obs[f"{label_out}_prob"] = 0.0

    ad_target.obs.loc[ad_test.obs_names, label_out] = labels_pred
    ad_target.obs.loc[ad_test.obs_names, f"{label_out}_prob"] = probs.max(axis=1)

    ad_target.obsm[f"{label_out}_probs"] = pd.DataFrame(
        0.0, index=ad_target.obs_names, columns=encoder.classes_
    )
    ad_target.obsm[f"{label_out}_probs"].loc[ad_test.obs_names, :] = probs


class PreProcessor:
    def __init__(self, scaler=None):
        if scaler is None:
            self.scaler = IdentityScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler}")

    def fit(self, adata, target_sum=None, restore=True):
        if restore:
            X_orig = adata.X.copy()

        adata.X = npc.utils.to_numpy(adata.X)

        # calc median
        if target_sum is None:
            self.target_sum = np.median(adata.X.sum(axis=1), axis=0)
        else:
            self.target_sum = target_sum

        sc.pp.normalize_total(adata, target_sum=self.target_sum)

        # log1p
        adata.X = np.log1p(adata.X)

        # fit standard scaler
        self.scaler.fit(adata.X)
        # restore old counts
        if restore:
            adata.X = X_orig

    def transform(self, adata):
        adata.X = npc.utils.to_numpy(adata.X)
        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        sc.pp.log1p(adata)
        adata.X = self.scaler.transform(adata.X)
        return adata

    def fit_transform(self, adata, **kwargs):
        self.fit(adata, restore=False, **kwargs)
        adata.X = self.scaler.transform(adata.X)
        return adata


class IdentityScaler:
    def __init__(self, **kwargs):
        pass

    def fit(self, adata):
        pass

    def transform(self, adata):
        return adata

    def fit_transform(self, adata):
        return self.transform(adata)


def median_expression(adata, layer=None):
    X = adata.X if layer is None else adata.layers[layer]
    return np.median(X.sum(axis=1), axis=0)
