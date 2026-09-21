import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
import datetime
import math
from sklearn.metrics import confusion_matrix
import warnings
import math
import torch

# --- In Trainer.py (or a new losses.py) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip # Make sure CLIP is installed: pip install git+https://github.com/openai/CLIP.git

class SemanticLowRankFusionLoss(nn.Module):
    def __init__(self, class_names, feature_dim, device, cls_num_list,
                 rank_k=16, num_similar_heads=5, svd_update_freq=10, warmup_epochs=20, head_threshold=300):
        super().__init__()
        self.device = device
        self.num_classes = len(class_names)
        self.feature_dim = feature_dim
        self.rank_k = rank_k
        self.num_similar_heads = num_similar_heads
        self.svd_update_freq = svd_update_freq
        self.warmup_epochs = warmup_epochs
        self.head_threshold = head_threshold

        if not class_names:
            raise ValueError("Class labels must be provided for semantic fusion.")

        # 1. Precompute semantic similarity matrix using CLIP
        print("Initializing SemanticLowRankFusionLoss...")
        print("Loading CLIP model...")
        self.clip_model, _ = clip.load("ViT-B/32", device=device) # Or choose another CLIP model
        self.clip_model.eval()
        print("Encoding class labels with CLIP...")
        with torch.no_grad():
            text_inputs = clip.tokenize(class_names).to(device)
            clip_text_embeddings = self.clip_model.encode_text(text_inputs) # [num_classes, clip_dim]
            clip_text_embeddings_norm = F.normalize(clip_text_embeddings, p=2, dim=1)
            # [num_classes, num_classes]
            self.semantic_sim_matrix = torch.matmul(clip_text_embeddings_norm, clip_text_embeddings_norm.t()).detach()
            # We might want similarity only between tail and head classes later
        print("Semantic similarity matrix computed.")

        # 2. Determine head and tail classes
        self.cls_num_list = np.array(cls_num_list)
        self.head_indices = np.where(self.cls_num_list >= self.head_threshold)[0]
        self.tail_indices = np.where(self.cls_num_list < self.head_threshold)[0] # Or use a specific tail threshold e.g., < 20
        print(f"Identified {len(self.head_indices)} head classes and {len(self.tail_indices)} tail classes.")

        if len(self.head_indices) < self.rank_k:
             print(f"Warning: Number of head classes ({len(self.head_indices)}) is less than target rank K ({self.rank_k}). Adjusting K.")
             self.rank_k = max(1, len(self.head_indices)) # Ensure K is at least 1 if there are head classes

        # 3. Initialize low-rank basis (V_k) - will be computed later
        self.V_k = None # Shape: [feature_dim, rank_k]
        self.last_svd_update_epoch = -1

    def _update_low_rank_basis(self, head_prototypes):
        """Performs SVD on head prototypes and updates self.V_k."""
        if head_prototypes.shape[0] < self.rank_k:
             print(f"Warning: Not enough head prototypes ({head_prototypes.shape[0]}) to compute rank {self.rank_k} SVD. Skipping update.")
             # Optionally, reduce rank_k dynamically or keep the old V_k
             # self.rank_k = head_prototypes.shape[0] # Reduce rank
             return # Keep old V_k or do nothing if it's the first time

        print(f"Updating low-rank basis (Rank K={self.rank_k})...")
        try:
            # Center the head prototypes before SVD (optional but recommended)
            # head_prototypes_centered = head_prototypes - head_prototypes.mean(dim=0, keepdim=True)
            # U, S, V = torch.linalg.svd(head_prototypes_centered, full_matrices=False)
            # Using torch.svd for potentially better compatibility/speed
            U, S, V = torch.svd(head_prototypes) # V here is V, not V^T

            # V_k contains the top K right singular vectors (principal components directions)
            self.V_k = V[:, :self.rank_k].detach() # Shape: [feature_dim, rank_k]
            print(f"Low-rank basis V_k updated (shape: {self.V_k.shape}).")
        except Exception as e: # Catch potential SVD errors (e.g., CUDA OOM)
            print(f"Error during SVD update: {e}. Skipping update.")
            # Keep the old V_k if available

    def forward(self, current_prototypes, current_epoch):
        """
        Calculates the semantic low-rank fusion loss for tail classes.
        Args:
            current_prototypes (torch.Tensor): All class prototypes [num_classes, feature_dim]
            current_epoch (int): The current training epoch.
        Returns:
            torch.Tensor: The calculated loss value.
        """
        # Check if warmup period is over
        if current_epoch < self.warmup_epochs:
            return torch.tensor(0.0, device=self.device)

        # Check if it's time to update SVD basis
        if self.V_k is None or \
           (self.svd_update_freq > 0 and (current_epoch - self.last_svd_update_epoch) >= self.svd_update_freq):
            # Get current head prototypes (ensure they are on the correct device)
            head_prototypes = current_prototypes[self.head_indices].detach().to(self.device)
            if len(head_prototypes) > 0:
                 self._update_low_rank_basis(head_prototypes)
                 self.last_svd_update_epoch = current_epoch
            else:
                 print("Warning: No head classes found, cannot compute SVD basis.")
                 return torch.tensor(0.0, device=self.device) # Cannot proceed without basis

        # If V_k is still None after trying to update (e.g., not enough heads), return 0 loss
        if self.V_k is None:
             print("Warning: Low-rank basis V_k not available. Skipping fusion loss.")
             return torch.tensor(0.0, device=self.device)

        total_fusion_loss = 0.0
        num_tail_classes_processed = 0

        V_k_device = self.V_k.to(current_prototypes.device) # Ensure V_k is on correct device

        for t_idx in self.tail_indices:
            m_t = current_prototypes[t_idx] # Tail prototype

            # Find K most semantically similar head classes
            sim_with_heads = self.semantic_sim_matrix[t_idx, self.head_indices]
            if len(sim_with_heads) == 0: continue # Skip if no head classes

            # Get indices of top K similar heads within the `head_indices` array
            num_k = min(self.num_similar_heads, len(self.head_indices))
            if num_k <= 0: continue

            top_k_sim_vals, top_k_rel_indices = torch.topk(sim_with_heads, k=num_k)

            # Get the actual indices in the full class list
            top_k_head_indices = self.head_indices[top_k_rel_indices.cpu().numpy()] # Ensure indices are on CPU for numpy indexing

            # Get corresponding head prototypes
            M_h = current_prototypes[top_k_head_indices] # Shape: [num_k, feature_dim]

            # Project to low-rank space
            m_t_prime = torch.matmul(m_t, V_k_device) # Shape: [rank_k]
            M_h_prime = torch.matmul(M_h, V_k_device) # Shape: [num_k, rank_k]

            # Semantic weighted fusion in low-rank space
            # Normalize similarity weights for the top K heads
            weights_h = F.softmax(top_k_sim_vals, dim=0) # Shape: [num_k]
            # Weighted average of head prototypes in low-rank space
            fused_h_prime = torch.sum(weights_h.unsqueeze(1) * M_h_prime, dim=0) # Shape: [rank_k]

            # Combine tail prototype with fused head prototypes (e.g., simple average or weighted)
            # Option 1: Just use fused heads as target
            # m_t_fused_prime = fused_h_prime
            # Option 2: Weighted average with original tail (weight w_t for tail)
            w_t = 0.5 # Example weight for the tail prototype itself
            m_t_fused_prime = w_t * m_t_prime + (1 - w_t) * fused_h_prime

            # Reconstruct back to original feature space
            m_t_fused_lowrank = torch.matmul(m_t_fused_prime, V_k_device.t()) # Shape: [feature_dim]

            # Calculate MSE loss between original tail and fused/reconstructed version
            # Use .detach() on the target to prevent gradients flowing back through the fusion process itself
            loss_t = F.mse_loss(m_t, m_t_fused_lowrank.detach())
            total_fusion_loss += loss_t
            num_tail_classes_processed += 1

        if num_tail_classes_processed > 0:
            avg_fusion_loss = total_fusion_loss / num_tail_classes_processed
            return avg_fusion_loss
        else:
            return torch.tensor(0.0, device=self.device)

class CustomScheduler:
    def __init__(self, optimizer, max_epochs, stage2_lr_factor=0.1, stage3_lr_factor=0.01, stage2_start_epoch=160, stage3_start_epoch=180):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.stage2_lr_factor = stage2_lr_factor  
        self.stage3_lr_factor = stage3_lr_factor  
        self.stage2_start_epoch = stage2_start_epoch  
        self.stage3_start_epoch = stage3_start_epoch 
        self.lr_lambda = self.get_lr_lambda()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

    def get_lr_lambda(self):
        def lr_lambda(epoch):
            if epoch < self.stage2_start_epoch:
                
                return 0.5 * (1 + math.cos(math.pi * epoch / self.max_epochs))
            elif epoch < self.stage3_start_epoch:
           
                cos_val_stage2 = 0.5 * (1 + math.cos(math.pi * self.stage2_start_epoch / self.max_epochs))
                linear_factor = (self.stage3_start_epoch - epoch) / (self.stage3_start_epoch - self.stage2_start_epoch)  # 线性下降到0
                return cos_val_stage2 * self.stage2_lr_factor * linear_factor
            else:
 
                cos_val_stage2 = 0.5 * (1 + math.cos(math.pi * self.stage2_start_epoch / self.max_epochs))
                linear_factor = (self.max_epochs - epoch) / (self.max_epochs - self.stage3_start_epoch)
                return cos_val_stage2 * self.stage3_lr_factor * linear_factor  # stage2_lr_factor * stage3_lr_factor
        return lr_lambda




    def step(self):
        self.scheduler.step()

# 使用示例
# 假设优化器为optimizer，总训练轮数为total_epochs
# scheduler = CustomScheduler(optimizer, total_epochs)
# 在每个epoch开始时调用 scheduler.step()
class Trainer(object):
    def __init__(self, args, model=None,train_loader=None, val_loader=None,weighted_train_loader=None,per_class_num=[],log=None,class_names = None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.label_weighting = args.label_weighting
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader
        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr,weight_decay=args.weight_decay)
        # self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.train_scheduler = CustomScheduler(self.optimizer,self.epochs)
        self.log = log
        self.beta = args.beta
        self.fusion_loss_calculator = None
        if args.fusion_gamma > 0 and class_names:
            # Determine feature dimension dynamically from model if possible
            # Example: Assuming fc_cb exists and has weight
            try:
                 feature_dim = 512# Get feature dim from the prototype layer
                 print(f"Feature dimension for fusion loss: {feature_dim}")
            except AttributeError:
                 raise ValueError("Cannot determine feature dimension for fusion loss. Ensure model.fc_cb.weight exists.")

            self.fusion_loss_calculator = SemanticLowRankFusionLoss(
                class_names=class_names,
                feature_dim=feature_dim,
                device=self.device,
                cls_num_list=self.cls_num_list,
                rank_k=args.fusion_rank_k,
                num_similar_heads=args.fusion_num_heads,
                svd_update_freq=args.fusion_svd_freq,
                warmup_epochs=args.fusion_warmup,
                head_threshold=args.head_threshold
            )
        elif args.fusion_gamma > 0:
            print("Warning: fusion_gamma > 0 but class_names were not provided. Semantic fusion loss disabled.")
        self.update_weight()

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            alpha = 1 - (epoch / self.epochs) ** 2
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            # switch to train mode
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            for i, (inputs, targets,_) in enumerate(self.train_loader):

                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                try:
                    input_invs, target_invs,_ = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs,_ = next(weighted_train_loader)

                input_invs_1 = input_invs[0][:input_org_1.size()[0]]
                input_invs_2 = input_invs[1][:input_org_2.size()[0]]

                one_hot_org = torch.zeros(target_org.size(0), self.num_classes).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1), 1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = self.per_cls_weights.cpu() * one_hot_invs

                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # Data augmentation
                lam = np.random.beta(self.beta, self.beta)

                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1, org2=input_org_2,
                                                                                        invs1=input_invs_1,
                                                                                        invs2=input_invs_2,
                                                                                        label_org=one_hot_org,
                                                                                        label_invs=one_hot_invs,
                                                                                        label_org_w=one_hot_org_w,
                                                                                        label_invs_w=one_hot_invs_w)


                output_1, output_cb_1, z1, p1,class_prototypes = self.model(mix_x, train=True)
                output_2, output_cb_2, z2, p2,_ = self.model(cut_x, train=True)

                prototype_contrast_weight = 1.0
                prototype_contrast_weight_epoch = 0
                if epoch>100:
                    prototype_contrast_weight_epoch =2

                prototypes_norm = F.normalize(class_prototypes, p=2, dim=1) 
                similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.transpose(0, 1)) 

                loss_matrix = -torch.diag(similarity_matrix) + torch.logsumexp(similarity_matrix, dim=1) # 计算 InfoNCE Loss

                per_cls_weights = self.per_cls_weights 
                weight_matrix = (per_cls_weights.unsqueeze(0) + per_cls_weights.unsqueeze(1)) / 2.0 
              
                weighted_loss_matrix = loss_matrix 
                prototype_contrastive_loss = torch.mean(weighted_loss_matrix)

        
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w
                semantic_fusion_loss = torch.tensor(0.0, device=self.device)
                if self.fusion_loss_calculator is not None and self.args.fusion_gamma > 0:

                    semantic_fusion_loss = self.fusion_loss_calculator(class_prototypes, epoch)
                
                loss = alpha* balance_loss + (1 - alpha) * rebalance_loss + (self.contrast_weight) * contrastive_loss +prototype_contrastive_loss*prototype_contrast_weight_epoch+self.args.fusion_gamma * semantic_fusion_loss
 
                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))  # TODO
                    print(output)
                    # evaluate on validation set
            acc1 = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1,  best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            # ,self.optimizer.param_groups[0]['lr']
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)

    def validate(self,epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        eps = np.finfo(np.float64).eps

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            end = time.time()
            for i, (input, target,_) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = self.model(input, train=False)

                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1 , flag='val', top1=top1, top5=top5))

            self.log.info(output)
            out_cls_acc = '%s Class Accuracy: %s' % (
            'val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))

            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
        return top1.avg

    def SimSiamLoss(self,p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def paco_adjust_learning_rate(self,optimizer, epoch, args):
        warmup_epochs = 10
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



