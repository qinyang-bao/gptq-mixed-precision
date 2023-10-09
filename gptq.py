import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                # inp: (batch*seq_length, hidden_dim)
                inp = inp.reshape((-1, inp.shape[-1]))
            # inp: (hidden_dim, batch*seq_length)
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        #  hmm, this and the 2/self.nsamples scale makes sure that the scale
        #  factor is always 2/N...But why don't we just apply the scale at the
        #  end?? Is it just to avoid having to add a line after all the add_batch calls?
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())

        #  H: (hidden_dim, hidden_dim)
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, get_saliency=False,
        outlier_relative_threshold=0.2
    ):
        # W: (hidden_dim_output, hidden_dim_input)
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            # Con1D from huggingface is like a linear layer, but with transposed weights
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        # for all rows, the dead columns => 0, cols = hidden_dim_input
        W[:, dead] = 0

        # quantize groups before permuting the columns
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        # if actorder:
        #     perm = torch.argsort(torch.diag(H), descending=True)
        #     W = W[:, perm]
        #     H = H[perm][:, perm]
        #     invperm = torch.argsort(perm)
            

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        _H = H.clone()
        _H[diag, diag] += damp
        _H = torch.linalg.cholesky(_H)
        _H = torch.cholesky_inverse(_H)
        _H = torch.linalg.cholesky(_H, upper=True)
        Hinv = _H

        self.quantizer.find_params(W, weight=True)
        Q = quantize(
            # w: (rows, 1), this does align with scale and zero dimension
            W, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
        )
        saliency = ((W - Q) ** 2 / torch.diag(Hinv).reshape(1, -1) ** 2)/2
        # sort by row

        mixed_percentage_losses = np.zeros(11)
        for mixed_percentage_idx in range(0, 11,1):
            _H = H.clone()
            _W = W.clone()

            print(f"Tring mixed precision high bitlength percentage {mixed_percentage_idx*10}:") 
            perm = torch.argsort(torch.mean(saliency, axis=1), descending=True)
            # torch.manual_seed(42)
            # perm = torch.randperm(saliency.size(0))
            bitlength = torch.ones_like(self.quantizer.scale) * 4
            bitlength[bitlength.size(0)//10*mixed_percentage_idx:]  = 3
            bitlength = bitlength[perm]
            self.quantizer.maxq = torch.pow(2, bitlength) - 1
            self.quantizer.find_params(W, weight=True)

            if actorder:
                perm = torch.argsort(torch.mean(saliency, axis=0), descending=True)
                # perm = torch.argsort(torch.diag(H), descending=True)
                _W = _W[:, perm]
                _H = _H[perm][:, perm]
                invperm = torch.argsort(perm) 

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            _H[diag, diag] += damp
            _H = torch.linalg.cholesky(_H)
            _H = torch.cholesky_inverse(_H)
            _H = torch.linalg.cholesky(_H, upper=True)
            Hinv = _H

            if get_saliency:
                            # mean(per column variance / (H diag)^2)
                            # This is later compared with (quant(w)-w) / (H diag)^2, so we are
                            # identifying outliers as the percentage of the quant_delta with respect to
                            # the variance of the weights.
                outlier_scale = (_W.var(dim=0) / torch.diag(_H).square()).mean().item()
                unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale
                saliency_wo_outliers = torch.zeros_like(W)
                unstructured_outlier_mask = torch.zeros_like(W, dtype=torch.bool)

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = _W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if get_saliency:
                    saliency_wo_outliers1 = torch.zeros_like(W1)
                    unstructured_outlier_mask1 = torch.zeros_like(W1, dtype=torch.bool)

                for i in range(count):
                    # w: (rows)
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q = quantize(
                        # w: (rows, 1), this does align with scale and zero dimension
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    if get_saliency and unstructured_outlier_threshold != float("inf"):

                        unstructured_outlier_mask1[:, i] = (
                            err1.square() > unstructured_outlier_threshold
                            # torch.abs(w) < 1e-3
                        )
                        # unstructured_outlier_mask1[:, i] = (
                        #     torch.rand(err1.size()) < 0.01
                        # )
                        
                        # re-quantize without outliers
                        is_outlier = unstructured_outlier_mask1[:, i].float()
                        weight_i_quantized_wo_outliers = quantize(
                            # 0 always stays as 0 after quantization
                            (w * (1 - is_outlier)).unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                        ).flatten()

                        q =  weight_i_quantized_wo_outliers * (1 - is_outlier) + w * is_outlier
                        Q1[:, i] = q

                        saliency_wo_outliers1[:, i] = ((w - q)*(1-is_outlier))**2 / d**2

                    err1 = (w - q) / d
                    # err1: (rows, 1), Hinv1: (1, block_size-1) => matmul: (rows, block_size-i)
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                if get_saliency:
                    saliency_wo_outliers[:, i1:i2] = saliency_wo_outliers1 / 2
                    unstructured_outlier_mask[:, i1:i2] = unstructured_outlier_mask1

                # (rows, block_size) @ (block_size, cols -i2) = (rows, cols-i2)
                _W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                if DEBUG:
                    self.layer.weight.data[:, :i2] = Q[:, :i2]
                    self.layer.weight.data[:, i2:] = W[:, i2:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

        # saliency = Losses
        # # sort by row
        # perm = torch.argsort(torch.mean(saliency, axis=1), descending=True)
        # # torch.manual_seed(42)
        # # perm = torch.randperm(saliency.size(0))
        # bitlength = torch.ones_like(self.quantizer.scale) * 4
        # bitlength[bitlength.size(0)//2:]  = 3
        # bitlength = bitlength[perm]
        # self.quantizer.maxq = torch.pow(2, bitlength) - 1
        # self.quantizer.find_params(W, weight=True)

        # if actorder:
        #     perm = torch.argsort(torch.mean(saliency, axis=0), descending=True)
        #     # perm = torch.argsort(torch.diag(H), descending=True)
        #     W = W[:, perm]
        #     H = H[perm][:, perm]
        #     invperm = torch.argsort(perm) 

        # damp = percdamp * torch.mean(torch.diag(H))
        # diag = torch.arange(self.columns, device=self.dev)
        # H[diag, diag] += damp
        # H = torch.linalg.cholesky(H)
        # H = torch.cholesky_inverse(H)
        # H = torch.linalg.cholesky(H, upper=True)
        # Hinv = H

        # for i1 in range(0, self.columns, blocksize):
        #     i2 = min(i1 + blocksize, self.columns)
        #     count = i2 - i1

        #     W1 = W[:, i1:i2].clone()
        #     Q1 = torch.zeros_like(W1)
        #     Err1 = torch.zeros_like(W1)
        #     Losses1 = torch.zeros_like(W1)
        #     Hinv1 = Hinv[i1:i2, i1:i2]

        #     if get_saliency:
        #         saliency_wo_outliers1 = torch.zeros_like(W1)
        #         unstructured_outlier_mask1 = torch.zeros_like(W1, dtype=torch.bool)

        #     for i in range(count):
        #         # w: (rows)
        #         w = W1[:, i]
        #         d = Hinv1[i, i]

        #         if groupsize != -1:
        #             if not static_groups:
        #                 if (i1 + i) % groupsize == 0:
        #                     self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
        #             else:
        #                 idx = i1 + i
        #                 if actorder:
        #                     idx = perm[idx]
        #                 self.quantizer = groups[idx // groupsize]

        #         q = quantize(
        #             # w: (rows, 1), this does align with scale and zero dimension
        #             w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
        #         ).flatten()
        #         Q1[:, i] = q
        #         Losses1[:, i] = (w - q) ** 2 / d ** 2

        #         err1 = (w - q) / d
        #         if get_saliency and unstructured_outlier_threshold != float("inf"):

        #             unstructured_outlier_mask1[:, i] = (
        #                 err1.square() > unstructured_outlier_threshold
        #                 # torch.abs(w) < 1e-3
        #             )
        #             # unstructured_outlier_mask1[:, i] = (
        #             #     torch.rand(err1.size()) < 0.01
        #             # )
                    
        #             # re-quantize without outliers
        #             is_outlier = unstructured_outlier_mask1[:, i].float()
        #             weight_i_quantized_wo_outliers = quantize(
        #                 # 0 always stays as 0 after quantization
        #                 (w * (1 - is_outlier)).unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
        #             ).flatten()

        #             q =  weight_i_quantized_wo_outliers * (1 - is_outlier) + w * is_outlier
        #             Q1[:, i] = q

        #             saliency_wo_outliers1[:, i] = ((w - q)*(1-is_outlier))**2 / d**2

        #         err1 = (w - q) / d
        #         # err1: (rows, 1), Hinv1: (1, block_size-1) => matmul: (rows, block_size-i)
        #         W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        #         Err1[:, i] = err1

        #     Q[:, i1:i2] = Q1
        #     Losses[:, i1:i2] = Losses1 / 2

        #     if get_saliency:
        #         saliency_wo_outliers[:, i1:i2] = saliency_wo_outliers1 / 2
        #         unstructured_outlier_mask[:, i1:i2] = unstructured_outlier_mask1

        #     # (rows, block_size) @ (block_size, cols -i2) = (rows, cols-i2)
        #     W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        #     if DEBUG:
        #         self.layer.weight.data[:, :i2] = Q[:, :i2]
        #         self.layer.weight.data[:, i2:] = W[:, i2:]
        #         print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        #         print(torch.sum(Losses))



            torch.cuda.synchronize()
            loss_sum = torch.sum(Losses).item()
            print('time %.2f' % (time.time() - tick))
            print('error', loss_sum)
            mixed_percentage_losses[mixed_percentage_idx] = loss_sum

        if actorder:
            Q = Q[:, invperm]

        # Huggingface Conv1D has transposed weights of a linear layer
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if get_saliency:
            return Losses, saliency_wo_outliers, Q, W, unstructured_outlier_mask, mixed_percentage_losses
        else:
            return Losses, Losses, Q, W, Hinv, mixed_percentage_losses

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
