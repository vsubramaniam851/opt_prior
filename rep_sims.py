import torch
import torch.nn.functional as F
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

class CKA(object):
    def __init__(self, device: torch.device, unbiased: bool = False, log_hsic: bool = False):
        self.device = device
        self.unbiased = unbiased
        self.log_hsic = log_hsic

    def centering(self, K: torch.Tensor) -> torch.Tensor:
        if K.dtype != torch.float32:
            K = K.to(dtype = torch.float32)
        n = K.size(0)
        I = torch.eye(n, device=self.device)
        H = I - torch.ones((n, n), device=self.device) / n
        return H @ K @ H

    def linear_HSIC(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.float()
        Y = Y.float()
        Lx = X @ X.T
        Ly = Y @ Y.T
        Kc = self.centering(Lx)
        Lc = self.centering(Ly)
        n = X.shape[0]
        return (Kc * Lc).sum() / ((n - 1) ** 2)

    def hsic_unbiased(self, K, L):
        m = K.shape[0]
        K_tilde = K.clone().fill_diagonal_(0)
        L_tilde = L.clone().fill_diagonal_(0)
        HSIC_value = (
            (torch.sum(K_tilde * L_tilde.T))
            + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
            - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
        )

        HSIC_value /= m * (m - 3)
        return HSIC_value

    def linear_unbiased_HSIC(self, X, Y):
        Lx = X @ X.T
        Ly = Y @ Y.T
        return self.hsic_unbiased(Lx, Ly) 

    @staticmethod
    def _global_rescale(Z, eps=1e-12):
        s = Z.abs().max().detach().clamp_min(1.0)
        return Z / s

    def linear_CKA(self, X, Y):
        if X.dim() != 2 or Y.dim() != 2 or X.size(0) != Y.size(0):
            raise ValueError('CKA expects two matrices with matching sample counts')
        n = X.size(0)
        if n < (4 if self.unbiased else 2):
            raise ValueError('Too few samples for the selected CKA estimator')
        with torch.autocast(device_type = X.device.type, enabled = False):
            X = X.double() if X.dtype == torch.float64 else X.float()
            Y = Y.double() if Y.dtype == torch.float64 else Y.float()
            X = X - X.mean(dim = 0, keepdim = True)
            Y = Y - Y.mean(dim = 0, keepdim = True)
            X = X / X.square().mean().sqrt().detach().clamp_min(1e-12)
            Y = Y / Y.square().mean().sqrt().detach().clamp_min(1e-12)
            K = X @ X.T / X.size(1)
            L = Y @ Y.T / Y.size(1)
            if self.unbiased:
                mask = ~torch.eye(n, device = X.device, dtype = torch.bool)
                K = K * mask
                L = L * mask
                K = (K - K.sum(dim = 0, keepdim = True) / (n - 2)
                     - K.sum(dim = 1, keepdim = True) / (n - 2)
                     + K.sum() / ((n - 1) * (n - 2))) * mask
                L = (L - L.sum(dim = 0, keepdim = True) / (n - 2)
                     - L.sum(dim = 1, keepdim = True) / (n - 2)
                     + L.sum() / ((n - 1) * (n - 2))) * mask
            K = K / torch.linalg.vector_norm(K).clamp_min(1e-12)
            L = L / torch.linalg.vector_norm(L).clamp_min(1e-12)
            return (K * L).sum().clamp(-1.0, 1.0)

    @staticmethod
    def _cka_cosine_cross_from_vectors(Va, Vb, eps=1e-12):
        G = Va @ Vb.t()
        na = torch.sqrt(torch.clamp((Va * Va).sum(dim=1), min=eps))
        nb = torch.sqrt(torch.clamp((Vb * Vb).sum(dim=1), min=eps))
        return G / (na[:, None] * nb[None, :] + eps)

    def token_aligned_CKA(self, Ha, Hb, exclude_cls = False, center_batch=True, eps=1e-12):
        if isinstance(Ha, tuple):
            Ha = Ha[0]
        if isinstance(Hb, tuple):
            Hb = Hb[0]
        if exclude_cls:
            Ha = Ha[:, 1:, :]
            Hb = Hb[:, 1:, :]
        Ha = self._global_rescale(Ha).float()
        Hb = self._global_rescale(Hb).float()
        if center_batch:
            Ha = Ha - Ha.mean(dim=0, keepdim=True)
            Hb = Hb - Hb.mean(dim=0, keepdim=True)
        T = min(Ha.shape[1], Hb.shape[1])
        Ha = Ha[:, :T, :]
        Hb = Hb[:, :T, :]
        Xa = Ha.transpose(0, 1).contiguous()
        Xb = Hb.transpose(0, 1).contiguous()
        Ka = torch.matmul(Xa, Xa.transpose(1, 2))
        Kb = torch.matmul(Xb, Xb.transpose(1, 2))
        Va = Ka.reshape(T, -1)
        Vb = Kb.reshape(T, -1)
        num = (Va * Vb).sum()
        den = torch.sqrt((Va * Va).sum().clamp_min(eps)) * torch.sqrt((Vb * Vb).sum().clamp_min(eps))
        return num / (den + eps)
    
class Procrustes(object):
    def __init__(self, device):
        self.device = device

    def svd_nuclear_norm(self, X, Y):
        Ux_r, Sx_r = torch.linalg.svd(X, full_matrices=False)[:2]
        Uy_r, Sy_r = torch.linalg.svd(Y, full_matrices=False)[:2]

        Sx_diag = torch.diag_embed(Sx_r)  # (bs x bs)
        Sy_diag = torch.diag_embed(Sy_r)  # (bs x bs)

        middle = Ux_r.transpose(0, 1).matmul(Uy_r)  # (bs x bs)
        A = Sx_diag @ middle @ Sy_diag               # (bs x bs)

        svals_A = torch.linalg.svdvals(A)
        nuc_norm = torch.sum(svals_A)

        return nuc_norm

    def orthogonal_procrustes_distance(self, X, Y, normalize=True):
        if normalize:
            X = X - X.mean()
            X = X / torch.norm(X, p='fro')
            Y = Y - Y.mean()
            Y = Y / torch.norm(Y, p='fro')
        xy_nuc = self.svd_nuclear_norm(X, Y)
        d = 2 - 2 * xy_nuc
        return d

    def orthogonal_procrustes_similarity(self, X, Y, normalize = True):
        d = self.orthogonal_procrustes_distance(X, Y, normalize)
        sim = 1.0 - d if normalize else 2.0 - d
        return sim
    
class DifferentiableKNN(object):
    def __init__(self, device, k, temperature = 500, metric = 'cosine', return_distributions = False, zscore_logits = True):
        self.device = device
        self.k = k
        self.temperature = temperature
        self.metric = metric
        self.return_distributions = return_distributions
        self.zscore_logits = zscore_logits

    def _sim(self, x):
        if self.metric == 'cosine':
            x = F.normalize(x, dim = 1)
            return x @ x.T 
        elif self.metric == 'euclidean':
            norms = x.square().sum(dim = 1, keepdim = True)
            return -(norms + norms.T - 2 * (x @ x.T)).clamp_min(0.0)
        else:
            raise NotImplementedError('metric must be `euclidean` or `cosine`')

    def soft_topk(self, logits, k, temperature):
        b, n = logits.shape
        topk_vals, topk_idx = torch.topk(logits, k, dim=1)

        if temperature == 0.0:
            probs = torch.zeros_like(logits)
            probs.scatter_(1, topk_idx, 1.0 / k)
            return probs
        scaled = topk_vals / temperature
        scaled = scaled - scaled.max(dim=1, keepdim=True).values
        weights = scaled.exp()
        weights /= weights.sum(dim=1, keepdim=True)

        probs = torch.zeros_like(logits)
        probs.scatter_(1, topk_idx, weights)
        return probs

    def _topk_indices(self, S):
        with torch.no_grad():
            B = S.size(0)
            S = S.clone()
            S.fill_diagonal_(-float('inf'))
        return S.topk(self.k, dim = 1).indices

    def _gather_rowwise(self, M, idx):
        return M.gather(dim = 1, index = idx)

    def _rowwise_standardize(self, z, eps = 1e-8):
        mean = z.mean(dim = 1, keepdim = True)
        std = z.std(dim = 1, keepdim = True)
        return (z - mean) / (std + eps)

    def soft_knn_alignment_topk(self, f, g):
        assert f.shape[0] == g.shape[0]
        b = f.size(0)
        device = f.device

        def _pairwise_sq_dists(x):
            xx = (x**2).sum(dim=1, keepdim=True)
            return xx + xx.T - 2 * x @ x.T

        if self.metric == 'euclidean':
            S_f = -_pairwise_sq_dists(f)
            S_g = -_pairwise_sq_dists(g)
        elif self.metric == 'cosine':
            S_f = (f @ f.T)
            S_g = (g @ g.T)
        else:
            raise ValueError('metric must be \'euclidean\' or \'cosine\'')
        diag_mask = torch.eye(b, device=device, dtype=torch.bool)
        S_f.masked_fill_(diag_mask, -1e8)
        S_g.masked_fill_(diag_mask, -1e8)

        P = self.soft_topk(S_f, self.k, self.temperature)
        Q = self.soft_topk(S_g, self.k, self.temperature)
        m_soft_per_anchor = (P * Q).sum(dim=1)
        m_soft = m_soft_per_anchor.mean()
        return (m_soft, P, Q) if self.return_distributions else m_soft
    
    def diff_knn_loss(self, f, g):
        B = f.size(0)
        S_f = self._sim(f)
        S_g = self._sim(g)

        topk_idx_g = self._topk_indices(S_g)
        z_f = self._gather_rowwise(S_f, topk_idx_g)
        z_g = self._gather_rowwise(S_g, topk_idx_g)

        if self.zscore_logits:
            z_f = self._rowwise_standardize(z_f)
            z_g = self._rowwise_standardize(z_g)

        tau = max(float(self.temperature), 1e-8)

        z_f = z_f / tau
        z_g = z_g / tau

        q = F.softmax(z_g, dim = 1).detach()
        B_, k = z_f.shape
        logits_expanded = z_f.unsqueeze(1).expand(B_, k, k).reshape(B_ * k, k)
        targets = torch.arange(k, device = self.device).unsqueeze(0).expand(B_, k).reshape(B_ * k)
        per_class_ce = F.cross_entropy(logits_expanded, targets, reduction = 'none')
        per_class_ce = per_class_ce.view(B_, k)

        loss_per_sample = (q * per_class_ce).sum(dim = 1)
        loss = loss_per_sample.mean()
        return loss

    def diff_knn_distribution_loss(self, f, g):
        if f.dim() != 2 or g.dim() != 2 or f.size(0) != g.size(0):
            raise ValueError('KNN expects two matrices with matching sample counts')
        n = f.size(0)
        if not 1 <= self.k < n or self.temperature <= 0:
            raise ValueError('KNN requires 1 <= k < batch size and a positive temperature')
        with torch.autocast(device_type = f.device.type, enabled = False):
            mask = ~torch.eye(n, device = f.device, dtype = torch.bool)

            def distribution(x):
                x = x.double() if x.dtype == torch.float64 else x.float()
                x = x - x.mean(dim = 0, keepdim = True)
                similarity = self._sim(x)[mask].reshape(n, n - 1)
                if self.zscore_logits:
                    similarity = similarity - similarity.mean(dim = 1, keepdim = True)
                    similarity = similarity / similarity.square().mean(dim = 1, keepdim = True).clamp_min(1e-8).sqrt()
                threshold = similarity.topk(self.k, dim = 1).values[:, -1:]
                log_weights = F.logsigmoid((similarity - threshold) / self.temperature)
                return log_weights - torch.logsumexp(log_weights, dim = 1, keepdim = True)

            log_p = distribution(f)
            with torch.no_grad():
                log_q = distribution(g)
                q = log_q.exp()
            loss = (q * (log_q - log_p)).sum(dim = 1).mean()
            return (loss, log_p.exp(), q) if self.return_distributions else loss

    def diff_knn_multilabel_loss(self, f, g, negative_weight=1.0):
        assert f.shape[0] == g.shape[0]
        B = f.size(0)
        device = f.device
        if self.k >= B:
            raise ValueError(f'k={self.k} must be < batch size B={B}')
        S_f = self._sim(f)
        with torch.no_grad():
            S_g = self._sim(g)
            topk_idx_g = self._topk_indices(S_g)

        diag_mask = torch.eye(B, device=device, dtype=torch.bool)
        valid_mask = (~diag_mask).float()
        targets = torch.zeros_like(S_f)
        targets.scatter_(1, topk_idx_g, 1.0)
        targets.masked_fill_(diag_mask, 0.0)

        logits = S_f.clone()
        if self.zscore_logits:
            count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (logits * valid_mask).sum(dim=1, keepdim=True) / count
            centered = (logits - mean) * valid_mask
            var = (centered ** 2).sum(dim=1, keepdim=True) / count
            std = var.sqrt().clamp_min(1e-8)
            logits = centered / std

        tau = max(float(self.temperature), 1e-8)
        logits = logits / tau
        logits = logits.masked_fill(diag_mask, 0.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pos_mask = targets * valid_mask
        neg_mask = (1.0 - targets) * valid_mask
        pos_loss = (bce * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1.0)
        neg_loss = (bce * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp_min(1.0)
        loss_per_row = (pos_loss + negative_weight * neg_loss) / (1.0 + negative_weight)
        loss = loss_per_row.mean()
        if self.return_distributions:
            probs = torch.sigmoid(logits) * valid_mask
            return loss, probs, targets
        return loss

    def token_aligned_diff_knn_multilabel_loss(self, Ha, Hb, exclude_cls = False, center_batch = True, negative_weight = 1.0, 
                                               eps = 1e-8, return_per_token = False):
        if isinstance(Ha, tuple):
            Ha = Ha[0]
        if isinstance(Hb, tuple):
            Hb = Hb[0]

        if exclude_cls:
            Ha = Ha[:, 1:, :]
            Hb = Hb[:, 1:, :]

        Ha = Ha.float()
        Hb = Hb.float()

        if center_batch:
            Ha = Ha - Ha.mean(dim=0, keepdim=True)
            Hb = Hb - Hb.mean(dim=0, keepdim=True)

        T = min(Ha.shape[1], Hb.shape[1])
        Ha = Ha[:, :T, :]
        Hb = Hb[:, :T, :]

        B = Ha.size(0)
        if self.k >= B:
            raise ValueError(f'k={self.k} must be < batch size B={B}')
        Xa = Ha.transpose(0, 1).contiguous()
        Xb = Hb.transpose(0, 1).contiguous()

        def _pairwise_token_sim(X):
            if self.metric == 'cosine':
                X = F.normalize(X, dim=-1, eps=1e-4)
                return X @ X.transpose(1, 2) 
            elif self.metric == 'euclidean':
                xx = (X ** 2).sum(dim=-1, keepdim=True)
                return -(xx + xx.transpose(1, 2) - 2 * (X @ X.transpose(1, 2)))
            else:
                raise ValueError("metric must be 'euclidean' or 'cosine'")

        S_a = _pairwise_token_sim(Xa)
        with torch.no_grad():
            S_b = _pairwise_token_sim(Xb)

        device = Ha.device
        diag_mask = torch.eye(B, device=device, dtype=torch.bool).unsqueeze(0)
        valid_mask = (~diag_mask).float()
        S_b_for_topk = S_b.masked_fill(diag_mask, -1e8)
        topk_idx_b = S_b_for_topk.topk(self.k, dim=2).indices
        targets = torch.zeros_like(S_a)
        targets.scatter_(2, topk_idx_b, 1.0)
        targets = targets.masked_fill(diag_mask, 0.0)
        logits = S_a.clone()
        if self.zscore_logits:
            zscore_eps = 1e-8
            count = valid_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            mean = (logits * valid_mask).sum(dim=2, keepdim=True) / count
            centered = (logits - mean) * valid_mask
            var = (centered ** 2).sum(dim=2, keepdim=True) / count
            std = torch.sqrt(var.clamp_min(zscore_eps * zscore_eps))
            logits = centered / std
        tau = max(float(self.temperature), eps)
        logits = logits / tau
        logits = logits.masked_fill(diag_mask, 0.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pos_mask = targets * valid_mask
        neg_mask = (1.0 - targets) * valid_mask
        pos_loss = (bce * pos_mask).sum(dim=2) / pos_mask.sum(dim=2).clamp_min(1.0)
        neg_loss = (bce * neg_mask).sum(dim=2) / neg_mask.sum(dim=2).clamp_min(1.0)
        loss_per_anchor = (pos_loss + negative_weight * neg_loss) / (1.0 + negative_weight)
        per_token_loss = loss_per_anchor.mean(dim=1)
        loss = per_token_loss.mean()
        if self.return_distributions:
            probs = torch.sigmoid(logits) * valid_mask
            if return_per_token:
                return loss, per_token_loss, probs, targets
            return loss, probs, targets
        if return_per_token:
            return loss, per_token_loss
        return loss