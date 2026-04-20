from pathlib import Path
import torch
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from GNT.transformer_network import GNT
from GNT.feature_network import ResUNet
from GNT.sample_ray import RaySamplerSingleImage
from GNT.render_ray import render_rays
from GNT.render_image import render_single_image
from GNT.Criterion import Criterion
from GNT.projection import Projector
from nerf.TorchNeRFWrapper import TorchNeRFWrapper
from GNT.data_loaders.create_training_dataset import create_training_dataset


class GNTModel(TorchNeRFWrapper):
    """
    GNT Implementation utilizing the agnostic TorchNeRFWrapper.
    """

    def __init__(self, args, model_name: str, dataset_name: str, **kwargs):
        super().__init__(model_name=model_name, dataset_name=dataset_name, **kwargs)
        
        self.args = args
        if hasattr(self.args, "local_rank"):
            self.device = torch.device(f"cuda:{self.args.local_rank}")

        self.criterion = Criterion()
        self.projector = Projector(device=self.device)

    # ---------------------------------------------------------
    # Compatibility Layer for legacy GNT render functions
    # ---------------------------------------------------------
    @property
    def net_coarse(self):
        return self.models["net_coarse"]

    @property
    def feature_net(self):
        return self.models["feature_net"]

    @property
    def net_fine(self) -> Optional[torch.nn.Module]:
        return self.models.get("net_fine", None)

    # ---------------------------------------------------------
    # Architecture Setup
    # ---------------------------------------------------------
    def build_models(self) -> dict[str, torch.nn.Module]: 
        models = {
            "net_coarse": GNT(
                self.args,
                in_feat_ch=self.args.coarse_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=self.args.N_importance > 0,
            ).to(self.device),
            
            "feature_net": ResUNet(
                coarse_out_ch=self.args.coarse_feat_dim,
                fine_out_ch=self.args.fine_feat_dim,
                single_net=self.args.single_net,
            ).to(self.device)
        }

        if not self.args.single_net:
            models["net_fine"] = GNT(
                self.args,
                in_feat_ch=self.args.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            ).to(self.device)

        if getattr(self.args, "distributed", False):
            for name, model in models.items():
                models[name] = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                )

        return models

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]: 
        param_groups = [
            {"params": self.net_coarse.parameters()},
            {"params": self.feature_net.parameters(), "lr": self.args.lrate_feature},
        ]
        
        if self.net_fine is not None:
            param_groups.append({"params": self.net_fine.parameters()})

        optimizer = torch.optim.Adam(param_groups, lr=self.args.lrate_gnt)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.args.lrate_decay_steps,
            gamma=self.args.lrate_decay_factor,
        )

        return {"gnt_optimizer": optimizer}

    # ---------------------------------------------------------
    # Training & Evaluation Loops
    # ---------------------------------------------------------
    def train_epoch(self, data_loader) -> Dict[str, float]: 
        """Executes a single epoch of ray sampling and network optimization."""
        # Ensure networks are in training mode
        for mod in self.models.values(): mod.train()
        
        optimizer = self.optimizers["gnt_optimizer"]
        epoch_loss = 0.0

        for train_data in data_loader:
            # 1. Ray Sampling
            ray_sampler = RaySamplerSingleImage(train_data, self.device)
            N_rand = int(1.0 * self.args.N_rand * self.args.num_source_views / train_data["src_rgbs"][0].shape[0])
            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=self.args.sample_mode,
                center_ratio=self.args.center_ratio,
            )

            # 2. Feature Extraction & Rendering
            featmaps = self.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            
            ret = render_rays(
                ray_batch=ray_batch,
                model=self,
                projector=self.projector,
                featmaps=featmaps,
                N_samples=self.args.N_samples,
                inv_uniform=self.args.inv_uniform,
                N_importance=self.args.N_importance,
                det=self.args.det,
                white_bkgd=self.args.white_bkgd,
                ret_alpha=self.args.N_importance > 0,
                single_net=self.args.single_net,
            )

            # 3. Optimization
            optimizer.zero_grad()
            loss, _ = self.criterion(ret["outputs_coarse"], ray_batch, {})
            
            if ret["outputs_fine"] is not None:
                fine_loss, _ = self.criterion(ret["outputs_fine"], ray_batch, {})
                loss += fine_loss

            loss.backward()
            optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()

        # Average loss across the epoch batches
        avg_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        return {"gnt_loss": avg_loss}

    @torch.no_grad()
    def _generate_eval_images(self, val_loader) -> Tuple[list, list]:
        """
        Samples validation views and forwards them through the network.
        These outputs feed directly into the base class's FID/PSNR evaluators.
        """
        for mod in self.models.values(): mod.eval()
        
        rendered_images = []
        gt_images = []

        for val_data in val_loader:
            ray_sampler = RaySamplerSingleImage(val_data, self.device, render_stride=self.args.render_stride)
            ray_batch = ray_sampler.get_all()
            
            if self.feature_net is not None:
                featmaps = self.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            else:
                featmaps = [None, None]

            ret = render_single_image(
                ray_sampler=ray_sampler,
                ray_batch=ray_batch,
                model=self,
                projector=self.projector,
                chunk_size=self.args.chunk_size,
                N_samples=self.args.N_samples,
                inv_uniform=self.args.inv_uniform,
                det=True,
                N_importance=self.args.N_importance,
                white_bkgd=self.args.white_bkgd,
                render_stride=self.args.render_stride,
                featmaps=featmaps,
                ret_alpha=self.args.N_importance > 0,
                single_net=self.args.single_net,
            )

            # Extract the final RGB prediction (fine if available, else coarse)
            pred_rgb = ret["outputs_fine"]["rgb"] if ret["outputs_fine"] is not None else ret["outputs_coarse"]["rgb"]
            pred_rgb = torch.clip(pred_rgb, 0.0, 1.0)
            
            # Reconstruct GT Image
            H, W = ray_sampler.H, ray_sampler.W
            gt_img = ray_sampler.rgb.reshape(H, W, 3)
            if self.args.render_stride != 1:
                gt_img = gt_img[::self.args.render_stride, ::self.args.render_stride]

            # Send back to CPU for metrics processing in the base class
            rendered_images.append(pred_rgb.detach().cpu())
            gt_images.append(gt_img.detach().cpu())
            
            # Break early if you only want to validate on 1 image per epoch to save time
            # break 

        return rendered_images, gt_images

    def build_dataloaders(self, data_path: Path):
        """
        Implements the GNT-specific training and validation datasets.
        """
        train_dataset, train_sampler = create_training_dataset(self.args)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=self.args.workers,
            pin_memory=True,
            sampler=train_sampler,
            shuffle=True if train_sampler is None else False,
        )

        # 2. Validation Dataset
        # Assumes self.args.eval_dataset and self.args.eval_scenes are set
        val_dataset = dataset_dict[self.args.eval_dataset](
            self.args, "validation", scenes=self.args.eval_scenes
        )
        val_loader = DataLoader(val_dataset, batch_size=1)

        return train_loader, val_loader
