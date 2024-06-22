import cv2
import torch
from torch import nn
import numpy as np

from scipy.stats import skew
from scipy.stats import pearsonr, spearmanr

from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler


def lcc(y, y_pred):
    """ Wrapping the pearsonr function to use it with GridSearchCV """
    corr = pearsonr(y_pred, y)[0]
    if np.isnan(corr):
        corr = 0.0
    return corr


def srocc(y, y_pred):
    """ Wrapping the spearmanr function to use it with GridSearchCV """
    corr = spearmanr(y_pred, y)[0]
    if np.isnan(corr):
        corr = 0.0
    return corr


class SSEQ:
    """ Spatial-Spectral Entropy-based Quality (SSEQ) index (Liu et al.) """
    def __init__(self,
                 block_size=8,
                 img_size=-1,
                 percentile=0.6,
                 scales=3,
                 eps=1e-5,
                 svr_regressor=None):
        self.block_size = block_size
        self.img_size = img_size
        self.percentile = percentile
        self.scales = scales
        self.eps = eps
        self.unfold = nn.Unfold(kernel_size=self.block_size, stride=self.block_size)
        self.svr_regressor = svr_regressor
        self.test_results = {'LCC': 0.0, 'SROCC': 0.0}

        self.m = self.make_dct_matrix()
        self.m_t = self.m.T

    def __call__(self, x):
        # Initial resizing
        x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x_gray = self.crop_input(x_gray)
        if self.img_size > 0:
            ratio = self.img_size / max(x_gray.shape)
            x_gray = cv2.resize(x_gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

        # Extracting the features at different scales
        spac_features, spec_features = self.extract_features(x_gray)
        for s in range(1, self.scales):
            ratio = 0.5**s
            x_scale = cv2.resize(x_gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            scale_features = self.extract_features(x_scale)
            spac_features.extend(scale_features[0])
            spec_features.extend(scale_features[1])
        features = np.array(spac_features + spec_features)

        # If we have loaded a SVR model, we predict the IQA score
        # The features are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(features.reshape(1, -1))
        else:
            return features

    def extract_features(self, x_gray):
        # Using Pytorch for extracting local image patches
        t = torch.from_numpy(x_gray).unsqueeze(0).unsqueeze(0).float()
        t = self.unfold(t).permute(0, 2, 1).squeeze()
        t = t.view(t.shape[0], self.block_size, self.block_size)

        # Spatial entropy
        # In order to compute it faster, I will use offsetting instead of computing row-wise entropy
        # from: https://discuss.pytorch.org/t/count-number-occurrence-of-value-per-row/137061/5
        t_flat = t.reshape(t.shape[0], -1).int()
        min_length = 256 * t_flat.shape[0]
        t_flat_offset = t_flat + 256 * torch.arange(t_flat.shape[0]).unsqueeze(1)
        counts = torch.bincount(t_flat_offset.flatten(), minlength=min_length).reshape(t_flat.shape[0], 256)
        mask = (counts > 0).float()
        p = counts / counts.sum(dim=1).unsqueeze(1)
        log_p = torch.log2(p).nan_to_num(posinf=0.0, neginf=0.0)
        se = np.sort(-1 * ((p * log_p * mask).sum(dim=1)).numpy())
        se_pooled = self.percentile_pooling(se)
        spatial_features = [se_pooled.mean(), skew(se)]

        # Spectral entropy
        m = torch.unsqueeze(torch.tensor(self.m), 0).repeat(t.shape[0], 1, 1)
        m_t = torch.unsqueeze(torch.tensor(self.m_t), 0).repeat(t.shape[0], 1, 1)
        t_dct = torch.bmm(torch.bmm(m, t), m_t)
        t_dct[:, 0, 0] = self.eps  # discarding the DC component
        p_sum = (t_dct ** 2).sum(axis=(1, 2)).unsqueeze(1).unsqueeze(1)
        p_i = (t_dct ** 2) / p_sum  # normalized spectral probability maps
        p_i[p_i == 0] = self.eps  # prevent NaNs
        fe = np.sort((p_i * torch.log2(p_i)).sum(axis=(1, 2)).numpy())  # entropy
        fe_pooled = self.percentile_pooling(fe)
        spectral_features = [fe_pooled.mean(), skew(fe)]

        return spatial_features, spectral_features

    def crop_input(self, x):
        """ We make sure the image is divisible into NxN tiles (N = block_size)
         If the image is not divisible, we crop it start from the top-left corner """
        h, w = x.shape
        h_cropped = h - (h % self.block_size)
        w_cropped = w - (w % self.block_size)
        return x[:h_cropped, :w_cropped]

    def make_dct_matrix(self):
        """ DCT can be computed as a matrix multiplication """
        m = np.zeros((self.block_size, self.block_size), dtype=np.float32)

        m[0, :] = np.sqrt(1 / self.block_size)
        for row in range(1, self.block_size):
            for col in range(self.block_size):
                k = np.sqrt(2 / self.block_size)
                m[row, col] = k * (np.cos((np.pi * (2 * col + 1) * row) / (2 * self.block_size)))

        return m

    def percentile_pooling(self, x):
        """ Percentile pooling, as explained in the paper """
        x_size = len(x)
        start = int(x_size * 0.5 * (1 - self.percentile))
        end = int(x_size - x_size * 0.5 * (1 - self.percentile))
        return x[start:end]

    def fit_svr(self, feature_db, n_jobs=4, test_size=0.3):
        """
        Fit an SVR model to a given dataset of features
        :param feature_db: dataframe with 14 columns: image name + 12 features + MOS
        :param n_jobs: number of threads for GridSearchCV
        :param test_size: test set size
        """

        X = feature_db.loc[:, feature_db.columns[1:-1]].values
        y = feature_db["MOS"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        params = {
            "svr__C": np.arange(1.0, 10, 0.5),
            "svr__epsilon": np.arange(0.1, 2.0, 0.1)
        }

        search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(), SVR()),
            param_grid=params,
            cv=5,
            n_jobs=n_jobs,
            verbose=1,
            scoring={
                "LCC": make_scorer(lcc),
                "SROCC": make_scorer(srocc)
            },
            error_score=0,
            refit="SROCC"
        )

        search.fit(X_train, y_train)
        self.svr_regressor = search.best_estimator_
        print(self.svr_regressor[1].C, self.svr_regressor[1].epsilon)

        # Test metrics
        y_pred = self.svr_regressor.predict(X_test)
        self.test_results = {
            'LCC': lcc(y_test, y_pred),
            'SROCC': srocc(y_test, y_pred)
        }

        return search.cv_results_

    def predict_score(self, f):
        """ Predicts the score from a set of features """
        score = self.svr_regressor.predict(f)
        return score
